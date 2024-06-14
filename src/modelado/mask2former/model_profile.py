#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install memory_profiler line_profiler


# mprof run profiling_model.py

# mprof plot -o memory_profile.png


# In[2]:


# ## Load Dataset

# In[3]:
from src.datasets import ImageSegmentationDataset
from src.models import Mask2FormerNova

from torch import load,device,no_grad
import sys



import os

os.environ['CURL_CA_BUNDLE'] = ''
@profile
def load_dataset():
    dataset = ImageSegmentationDataset(['../../../data/processed/dataset/trimmed_FD_nominal_FHC_nonswap.999_of_2000.h5'])

    return dataset


# ## Load Model

# In[4]:


@profile
def load_model():
    #ckpt_path='../../../models/mask2former_nova_.ckpt'
    ckpt_path = 'modelo_bruto.pt'
    model = Mask2FormerNova()
    checkpoint = load(ckpt_path, map_location=device('cpu'))  # Carga el checkpoint en CPU
    model.load_state_dict(checkpoint)
    model = model.half()
    return model



# ## Run Detection

# In[5]:


@profile
def load_sample_image(dataset, image_id):
    image = dataset[image_id]["pixel_values"].unsqueeze(0)
    return image


@profile
def detect(model, image):
    outputs = model(image,None,None)
    return outputs


@profile
def run_detect(dataset, model):
    # Inference
    model.eval()
    with no_grad():
        for image_id in range(10):
            image = load_sample_image(dataset, image_id)

            # Run object detection

            detect(model, image)


# In[6]:


if __name__ == "__main__":
    dataset = load_dataset()
    model = load_model()
    run_detect(dataset, model)

# In[ ]: