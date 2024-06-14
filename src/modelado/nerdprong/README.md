# MLInstanceSegmentation
Instance Segmentation and Clustering Scripts

Modified from Matterport's implementation of Mask R-CNN https://github.com/matterport/Mask_RCNN

MRCNN folder contains the base code architecture, model, and training functions.
Scripts folder has some helpful wrappers for training on the Wilson Cluster.
Inspect has some jupyter notebooks for validating sensibility of input training set and output models.

nova_basic.py is the baseline script for calling training and evaluation functions. Default config is based on what was trained for evaluation in Prod5.1.