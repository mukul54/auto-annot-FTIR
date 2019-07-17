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


def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    
    # return the MSE, the lower the error, the more "similar"
    # the two images 
    return err


def reject_and_save_roi_ssim(reg_with_roi, reg_unmarked,  ir_with_roi, ir_unmarked, thresh = 0.25):
    """
    Reject the ROI with SSIM greater than thresh value
    
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
    IR image-  free from distorted roi and list of ssim for each ROI
    ssim_hist - list of ssim values for all pairs of ROI for IR and registered images
    """
    
    reg_list, ir_list, cntrs = extract_roi(reg_with_roi, ir_with_roi, reg_unmarked, ir_unmarked)
    
    ss_hist = []
    mse_val = []
    
    idx = 1
    #plt.figure(figsize = (8,48))
    
    img = copy.deepcopy(ir_unmarked)
    
    for reg, ir, cnt in zip(reg_list, ir_list, cntrs):
        reg_equalized = cv2.equalizeHist(reg)
        ir_equalized = cv2.equalizeHist(ir)
        
        ssim_value  = ssim(reg_equalized, ir_equalized)
        ss_hist.append(ssim_value)
        

        #mse1 = mse(reg_equalized, ir_equalized)
        #mse_val.append(mse1)
        
        #print(mse1, ssim_value)
        if ssim_value > thresh:
            cv2.drawContours(img, [cnt], 0, (0,0, 255), 10)
        
    return ss_hist, img

def calc_hist(reg_img_list, ir_img_list):
    """ Returns the list of histogram of registered and infrared image patches"""
    h_ir = []
    h_reg = []
    for r, i in zip(reg_img_list, ir_img_list):
        h1 = cv2.calcHist(r, [0], None, [256], [0,256])
        h1 = cv2.normalize(h1, h1).flatten()
        h_reg.append(h1)

        #i1 = cv2.bitwise_not(r)
        h11 = cv2.calcHist(i, [0], None, [256], [0,256])
        h11 = cv2.normalize(h11, h11).flatten()
        h_ir.append(h11)
    return h_reg, h_ir


def reject_and_save_roi_hist(reg_with_roi, reg_unmarked,  ir_with_roi, ir_unmarked, method, thresh = 0.25, cmp_mode = 'l'):

    """
    Reject the ROI with SSIM greater than thresh value
    
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
    
    cmp_mode: char
              mode for comparison, 'l' -> less is good
              'm' -> more is good
    Returns
    -------
    IR image free from distorted roi and list of similarity values for each ROI
    
    """
    
    reg_list, ir_list, cntrs = extract_roi(reg_with_roi, ir_with_roi, reg_unmarked, ir_unmarked)
    h_reg, h_ir = calc_hist(reg_list, ir_list)
    
    results = []
    
    #idx = 1
    #plt.figure(figsize = (8,48))
    
    img = copy.deepcopy(ir_unmarked)
    
    for h1, h2, cnt in zip(h_reg, h_ir, cntrs):
        
        h1 = cv2.equalizeHist(h1)
        h2 = cv2.equalizeHist(h2)
        
        dist  = cv2.compareHist(h1, h2, method)
        results.append(dist)

        if dist > thresh and cmp_mode == 'm':
            cv2.drawContours(img, [cnt], 0, (0,0, 255), 10)
            
        if dist < thresh and cmp_mode == 'l':
            cv2.drawContours(img, [cnt], 0, (0,0, 255), 10)
        
    return results, img

#methods = [cv2.HISTCMP_CORREL, cv2.HISTCMP_CHISQR, cv2.HISTCMP_INTERSECT, cv2.HISTCMP_BHATTACHARYYA ]
# cv2.HISTCMP_CHISQR and cv2.HISTCMP_BHATTACHARYYA -  minimum is the better other two reverse
#method = cv2.HISTCMP_BHATTACHARYYA
#results, img = reject_and_save_roi_hist(reg_with_roi, reg_unmarked,  ir_with_roi, 
#                                        ir_unmarked, method = cv2.HISTCMP_BHATTACHARYYA, thresh = 0.25, cmp_mode = 'l')

