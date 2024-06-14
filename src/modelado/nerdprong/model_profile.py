#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install memory_profiler line_profiler


#mprof run profiling_nerdprong.py

#mprof plot -o memory_profile.png


# In[2]:



import os
import sys

import tensorflow as tf


# Root directory of the project
ROOT_DIR = os.path.abspath("drive/MyDrive/nova/m-rcnn/")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

import mrcnn.model as modellib



# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
import nova_basic

import time

# ## Load Dataset

# In[3]:


@profile
def load_dataset():
    dataset = nova_basic.novaDataset()
    files=['/wclustre/nova/users/rafaelma/dataset/trimmed_FD_nominal_FHC_nonswap.999_of_2000.h5']
    dataset.load_nova(files)
    dataset.prepare()
    
    return dataset


# ## Load Model

# In[4]:


@profile
def load_model():
    class inferNovaConfig(nova_basic.novaConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BATCH_SIZE=1
    config = inferNovaConfig()
    # Device to load the neural network on.
    # Useful if you're training a model on the same
    # machine, in which case use CPU and leave the
    # GPU for training.
    DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

    # Inspect the model in training or inference modes
    # values: 'inference' or 'training'
    # TODO: code for 'training' test mode not ready yet
    TEST_MODE = "inference"
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,config=config)
    model.load_weights('mask_rcnn_nova_0249.h5', by_name=True)
    return model


# ## Run Detection

# In[5]:


#@profile
def load_sample_image(dataset,image_id):
    class inferNovaConfig(nova_basic.novaConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        BATCH_SIZE=1
    config = inferNovaConfig()
    image, _,_ ,_ ,_  = modellib.load_image_gt(dataset, config, image_id)
    return image

@profile
def detect(model,image):
    results = model.mi_detect([image])
    return results

@profile
def run_detect(dataset,model):

    for image_id in range(20):

        image =load_sample_image(dataset,image_id)

        # Run object detection
        detect(model,image)






# In[6]:


if __name__ == "__main__":
    dataset=load_dataset()
    model=load_model()
    run_detect(dataset,model)


# In[ ]:

