# Automatic Annotation of FTIR tissue images

## Introduction

This repository contains some of the early work for the automated annotations of Fourier Transform InfraRed Images based on the annotations of the H&E images. The steps for this tasks include:

- Registration of FTIR and H&E image: We tried various methods for this task, including some of them based on deep learning methods, but finally Mutual Information based method turns out to be giving best result. `regis.m` file contains the matlab code for the registration.

- Region of Interest(ROI) Transfer: After image registration we get the the transformation matrix which align the H&E images to the FTIR images. We can use that matrix on the ROI mask of the H&E image to get the ROI of the FTIR image.

- Rejection of the distorted ROI: FTIR and H&E image may not be from the same layer of the tissue section, hence there can be the cases where ROI may not match or ROI in FTIR for some of the frequencies can be distorted, hence it is important to not consider them in our final annotations. Directory `reject_roi`, contains various methods for rejecting the bad or distorted ROI from the final annotations. Notebook `final_notebook_stage_1.ipynb` contains all the details and code for step2 and step3.

## Problems and Future Work
- This is not an end-to-end framework and require two different languages- matlab and python. One of the first future task for this work would be to remove the dependency from one of the language and make it an end-to-end framework.

- Another big issue with this task is the time required for image registration. Registration for one image takes from 15-20 minutes and looking at the image size the reason for this looks obvious. Still we can think of some other methods to reduce the time required for registration.

## Authors
 - [Mukul Ranjan](https://github.com/mukul54)
 
 * Mentors*
 - [Luke Pfister](https://github.com/lukepfister)
 - [Sachi Mittal](https://www.linkedin.com/in/shachi-mittal-8379b645/)
