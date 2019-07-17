from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
import os
import os
import sys
import time

from skimage.color import label2rgb
import SimpleITK as sitk
import openslide
import skimage

from keras.applications import VGG16
from keras.applications.resnet50 import ResNet50
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.imagenet_utils import decode_predictions
%pylab inline


from scipy.spatial import distance as dist
import copy
from scipy.io import loadmat
from keras.optimizers import SGD
from keras.callbacks import *
from skimage import data, img_as_float
from skimage.measure import compare_ssim as ssim


"""
reg_with_roi = cv2.imread('E:\\transfer_roi\\result\\registered\\marked\\43849A_sec1.tif')
ir_img_name = '43849A_sec1.tif'

reg_unmarked = cv2.imread('E:\\transfer_roi\\result\\registered\\unmarked\\43849A_sec1.tif')
ir_unmarked =  cv2.imread('E:\\transfer_roi\\ir\\unmarked_resized\\43849A_sec1.png')
print(reg_with_roi.shape, reg_unmarked.shape, ir_unmarked.shape)

"""

def transfer_roi(im_src, im_dst, dst_img):
    """
    Transfer the region of interest from source image to destination image and save image to dst_img
    
    Parameters
    ----------
    im_src: ndarray
             3-channel source image containing region of interest with a particular pixel value
             
    im_dst: ndarray
             2-channel/3-channel destnation image where roi is to be transfered from source image,
             same shape as of src_image
             
    dst_img: str
             Filename of the infrared image with roi
           
    Returns
    -------
    3-channel image with the roi
    
    """
    
    # get the pixel value and the location of the roi
    roi_pos = np.where(im_src[:,:,2] == 255)
    
    #get the list of the x and y coordinates of the location of roi
    x = list(roi_pos[0])
    y = list(roi_pos[1])
    
    dst_with_roi = copy.deepcopy(im_dst)
    dst_with_roi[x, y] = [0,0,255]
    
    marked_folder_roi =  'E:\\transfer_roi\\ir\\marked'
    marked_path = os.path.join(marked_folder_roi, dst_img)
    
    if not os.path.exists(marked_path):
        cv2.imwrite(marked_path, dst_with_roi)
    
    return dst_with_roi


def extract_roi(reg_with_roi, ir_with_roi, reg_unmarked, ir_unmarked):
    
    """
    Extract the roi from registered and infrared image and store them in the seperate list to further
    process them using differet methods - cnn, histogram, chisquared_distance
    
    Parameters
    ----------
    reg_with_roi: ndarray
                  3-channel registered image containing region of interest with a particular pixel value
             
    ir_with_roi: ndarray
                 3-channel infra-red image containing region of interest transfered from registered image
                 
    reg_unmarked: ndarray
                  3-channel/2-channel registered image not containing region of interest with a particular pixel value
             
    ir_unmarked: ndarray
                 3-channel/2-channel infra-red image not containing region of interest transfered from registered image
             
             
    Returns
    -------
    reg_roi_list - Lists of extracted roi - 3-channel roi from registered image
    reg_roi_list - Lists of extracted roi - 3-channel roi from infrared image
    cntrs - list of contours
    """
    roi_pos = np.where( reg_with_roi[:,:,2] == 255 ) 
    
    x = list(roi_pos[0])
    y = list(roi_pos[1])
    
    #make a 2-d mask
    
    mask = np.zeros_like(reg_with_roi[:,:,1])
    mask[x,y] = 255
    
    _, cntrs = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[:2]

    reg_roi_list = []
    ir_roi_list = []
    
    #masks = []
    for  cnt in cntrs:
        
        if reg_unmarked.ndim == 3:
            reg_unmarked = cv2.cvtColor(reg_unmarked, cv2.COLOR_BGR2GRAY)
            
        if ir_unmarked.ndim == 3:
            ir_unmarked = cv2.cvtColor(ir_unmarked, cv2.COLOR_BGR2GRAY)
            
        temp_mask = np.zeros_like(reg_unmarked)
        cv2.fillPoly(temp_mask, [cnt], (255,255,255))
        #masks.append(temp_mask)
        
        reg_roi = cv2.bitwise_and(temp_mask, reg_unmarked)
        ir_roi = cv2.bitwise_and(temp_mask, ir_unmarked)
        
        x, y, w, h = cv2.boundingRect(cnt)
        reg_roi = reg_roi[y:y+h, x:x+w]
        ir_roi =  ir_roi[y:y+h, x:x+w]
        
        reg_roi_list.append(reg_roi)
        ir_roi_list.append(ir_roi)
        
    return reg_roi_list, ir_roi_list, cntrs


def update_model(train_dir, image_size = 224, batch_size = 8, epochs = 2):
    
    """ Update the weight of the model after new images are being added to the datasets"""
    
    # Create a data generator and specify
    # the parameters for augmentation
    train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
    
    # create an iterator for data generator
    # and autment the images
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size= batch_size,
        class_mode='categorical')
    
    #load pretrained model
    model = models.load_model('vgg16_finetuned.h5')
    
    # Compile the pretrained model in order to update its weight
    model.compile(loss='categorical_crossentropy',
                  optimizer = optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    
    # use keras checkpoint to update the model weight
    file_path = 'vgg16_finetuned.h5'
    checkpoint = ModelCheckpoint(file_path)
    callbacks_list = [checkpoint]
    
    # Train the model to update model weight
    history = model.fit_generator(
          train_generator,
          steps_per_epoch = train_generator.samples/train_generator.batch_size,
          epochs = epochs,
          callbacks = callbacks_list)


#train_dir = "E:\\transfer_roi\\cnn_rejection\\train"
#update_model(train_dir, epochs = 5)

def reject_and_save_roi_cnn(reg_with_roi, reg_unmarked,  ir_with_roi, ir_unmarked):
    
    """
    Reject unclear and distorted roi's from IR image using cnn and save them to 
    cnn-training forlder for further learning process based on their class
    
    Parameters
    ----------
    reg_with_roi: ndarray
             3-channel registered image containing region of interest with a particular pixel value
             
    ir_with_roi: ndarray
                 3-channel infra-red image containing region of interest transfered from registered image   
                 
    reg_unmarked: ndarray
                  3-channel/2-channel registered image not containing region of interest 
             
    ir_unmarked: ndarray
                 3-channel/2-channel infra-red image not containing region of interest 
             
                    
    cnn_folder: str
                path for the directory where extracted images is to be saved
             
    Returns
    -------
    IR image free from distorted roi 
    
    """
    reg_list, ir_list, cntrs = extract_roi(reg_with_roi, ir_with_roi, reg_unmarked, ir_unmarked)
    
    model = models.load_model('vgg16_finetuned.h5')
    
    img = copy.deepcopy(ir_unmarked)
    idx = 0
    for ir_img, reg_img, cnt in zip(ir_list, reg_list, cntrs):
        ir_img = cv2.resize(ir_img, (224,224), cv2.INTER_NEAREST)
        ir_img1 = skimage.color.grey2rgb(ir_img)
        ir_img = np.reshape(ir_img1, (1, 224,224,3))
        
        #make prediction from model
        pred1 = model.predict(ir_img)
        pred = np.argmax(pred1)
        
        #print(pred1)
        # pred = 0 -> deformed image, pred = 1 -> undeformed ir or h&e image
        
        if(pred == 1):
            cv2.drawContours(img, [cnt], 0, (0,0, 255), 10)
        if(pred == 0):
            print('Transfered roi with index {0} on IR image is deformed'.format(idx))
            #fn = os.path.join(cnn_folder, '{}.png'.format(idx))
            #cv2.imwrite(fn, ir_img1)
        idx += 1
    return img

