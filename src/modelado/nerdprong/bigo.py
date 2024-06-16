import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import nova_basic

from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import big_o

os.environ['CURL_CA_BUNDLE'] = ''

class inferNovaConfig(nova_basic.novaConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = inferNovaConfig()


# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library




dataset = nova_basic.novaDataset()
dataset.load_nova(['../../../datos/procesados/dataset/trimmed_FD_nominal_FHC_nonswap.999_of_2000.h5'])
dataset.prepare()
# Root directory of the project




# Import Mask RCNN
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
# Create model in inference mode
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)
weights_path = '/wclustre/nova/users/rafaelma2/models/mask_rcnn_nova_0249.h5'
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

def predict(batch_size):
    print(" prediciendo batch_size=",batch_size )
    model.config.BATCH_SIZE=batch_size
    imagenes=next(modellib.data_generator(dataset,config,batch_size=batch_size))[0][0]
    model.detect(imagenes, verbose=0)


best, others = big_o.big_o(predict, big_o.datagen.n_,n_repeats=10,min_n=1,max_n=500, n_measures=20)
print(best)