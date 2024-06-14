#######################################################################
#
# Contains a config and dataset for running, testing, and training
# on NOvA data contained in hdf5 files.
#
# Also included are examples for viewing data, testing a model, and
# training.
#
#######################################################################

import os
import numpy as np
import h5py

import itertools

from .utils import *


# A class for loading nova images from numpy arrays in an hdf5 file
class novaDataset():
    def __init__(self, files):
        self._image_ids = []
        self.image_info = []
        self.load_nova(files)
        
    # Load the nova classes, and set the the image ids
    def load_nova(self, filelist):
        n = 0  # keep tracking of the total number of events
        
        for f in filelist:
            h5 = h5py.File(f, 'r')
            labs = h5['label'][:]
            Es = h5['energy'][:]
            object_arrays = h5['cvnobjmap'][:]
            

            for i, (lab, E,object_array) in enumerate(zip(labs, Es,object_arrays)):
                if lab == 15:  # omit cosmics
                    continue
                if E > 10:  # omit real crazy events
                    continue
                if np.max(object_array)>5:
                    continue
                # use both x and y views together
                self.add_image('NOvA', n, f, idx=i, view='X')
                n += 1
                self.add_image('NOvA', n, f, idx=i, view='Y')
                n += 1
            h5.close()

        print('Loaded', n, 'images.')


    # Transform the 1D pixelmap array to one of the views
    def transform_pm(self, pm, view):
        # if pixel map geometry changes so will this
        newpm = pm.reshape(2, 100, 80)
        if view == 'X':
            return newpm[0]
        else:
            return newpm[1]
    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)
    
    # Useful for visualizing results
    def load_image_BonW(self, image_id):
        info = self.image_info[image_id]
        hf = h5py.File(info['path'], 'r')
        pm = hf['cvnmap'][info['idx']]
        pm = self.transform_pm(pm, info['view'])
        hf.close()

        image = np.zeros(pm.shape + (3,))

        image[pm > 0] = [255, 255, 255]

        return image

    # Pixel map of the hits
    def load_image(self, image_id):
        info = self.image_info[image_id]
        hf = h5py.File(info['path'], 'r')
        pm = hf['cvnmap'][info['idx']]

        pm = self.transform_pm(pm, info['view'])
        hf.close()

        # second channel is boolean for hit or no hit
        blue = np.zeros_like(pm)
        blue[pm > 0] = 255

        image = np.zeros(pm.shape + (3,))
        image[:, :, 0] = pm
        image[:, :, 1] = blue
        image[:, :, 2] = blue

        return self.pm_resize(image)

    # Turn each cell into 9 (3x3) and remove the last 15 planes.
    # Keeps input size under 256x256
    def pm_resize(self, image, n=3):
        return image[:-15, :].repeat(n, axis=0).repeat(n, axis=1)

    # Create the binary mask from the label and object instance of each pixel
    def load_mask(self,image_id):
        info = self.image_info[image_id]
        hf = h5py.File(info['path'],'r')
        object_array = hf['cvnobjmap'][info['idx']]
        object_array = self.transform_pm(object_array, info['view'])
        label_array = hf['cvnlabmap'][info['idx']]
        label_array = self.transform_pm(label_array, info['view'])
        hf.close()

        # Some objects never make it as a max contributor to any hits
        # Remove them
        max_object = np.max(object_array)
        for i in range(max_object, 0, -1):
            if object_array[object_array==i].shape[0]==0:
                object_array[object_array >= i] -= 1

        max_object = np.max(object_array)
        mask = to_categorical(object_array, num_classes=max_object+1).astype(dtype=np.bool)
        mask = mask[:,:,1:]
        
        
        #eliminar 
        #for i in range(max_object-1, 0, -1):
        #    if np.sum(object_array == i) < 2:
        #        object_array[object_array >= i] -= 1        
        #        mask=np.delete(mask, i, axis=2)
        #max_object = np.max(object_array)   
        
        label = np.zeros(max_object,dtype=np.int32)
        
        for i in range(max_object):
            # Sometimes we get an object with different labels at each hit
            labs = label_array[object_array == i+1]
            # Use the most common
            id = np.argmax(np.bincount(labs))

            # Redefine labels
            if id==7:
                id=4
            elif id>7 or id==4:
                id=6
            label[i] = id

        return self.pm_resize(mask), label

    def __len__(self):
        return len(self.image_info)

#
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import MaskFormerImageProcessor

class ImageSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(self, files_list):
        
        self.dataset = novaDataset(files_list)
        self.processor = MaskFormerImageProcessor(reduce_labels=True, ignore_index=255, do_resize=False, do_rescale=False,do_normalize=False)
        #self.transform = ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = self.dataset.load_image(idx)

        mask, class_ids = self.dataset.load_mask(idx)
        
        mask = torch.from_numpy(mask.transpose((2, 0, 1)))

        inputs = self.processor([image], return_tensors="pt")
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        inputs["class_labels"] = torch.from_numpy(class_ids)
        inputs["mask_labels"] = mask
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float32)
        inputs["pixel_mask"] = inputs["pixel_mask"].to(dtype=torch.int64)
        inputs["mask_labels"] = inputs["mask_labels"].to(dtype=torch.float32)
        inputs["class_labels"] = inputs["class_labels"].to(dtype=torch.int64)

        return inputs


    
