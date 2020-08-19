# Automatic Annotation of FTIR tissue images

This repository contains some of the early work for the automated annotations of Fourier Transform InfraRed Images based on the annotations of the H&E images. The steps for this tasks include:

- Registration of FTIR and H&E image: We tried various methods for this task, including some of them based on deep learning methods, but finally Mutual Information based method turns out to be giving best result. `regis.m` file contains the matlab code for the registration.

- Region of Interest(ROI) Transfer: After image registration we get the the transformation matrix which align the H&E images to the FTIR images. We can use that matrix on the ROI mask of the H&E image to get the ROI of the FTIR image.

- Rejection of the distorted ROI: FTIR and H&E image may not be from the same layer of the tissue section, hence there can be the cases where ROI may not match or ROI in FTIR for some of the frequencies can be distorted, hence it is important to not consider them in our final annotations. Directory `reject_roi`, contains various methods for rejecting the bad or distorted ROI from the final annotations. Notebook `final_notebook_stage_1.ipynb` contains all the details and code for step2 and step3.

