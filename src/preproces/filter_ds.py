#!/usr/bin/env python
# coding: utf-8

# In[55]:


import os
import h5py
import numpy as np
from tqdm import tqdm


# In[56]:


dataset_path='/wclustre/novapro/R19-11-18-Prod5_fullset/mbaird42-trimmed-H5s-for-CVN-training/FD_nominal/FHC_nonswap/'
files=os.listdir(dataset_path)
files = [dataset_path+f for f in files]
#files=files[0:2]
#files
def load_nova(filelist):
    
    for filename in tqdm(filelist):
        with h5py.File(filename, 'r') as h5_file, \
                h5py.File(os.path.join('/wclustre/nova/users/rafaelma/dataset', os.path.basename(filename)), 'w') as hf:

            labels = h5_file['label']
            energies = h5_file['energy']
            cvn_maps = h5_file['cvnmap']
            cvn_obj_maps = h5_file['cvnobjmap']
            cvn_lab_maps = h5_file['cvnlabmap']

            num_events = labels.shape[0]
            events_to_keep = np.ones(num_events, dtype=bool)

            # Filtrar los eventos que no cumplen las condiciones
            events_to_keep &= labels[:] != 15# no cosmics
            events_to_keep &= energies[:] <= 10# no crazy events
            events_to_keep &= np.max(cvn_obj_maps[:], axis=1) <= 5# no more of 5 particles
            
            events_to_keep &= np.sum(np.isin(cvn_lab_maps[:], [5]), axis=1) >= 5 #no < 5 hits pions
            events_to_keep &= np.sum(np.isin(cvn_lab_maps[:], [4]), axis=1) == 0 #no id 4
            events_to_keep &= np.sum(cvn_lab_maps[:] > 7, axis=1) ==0 #no id > 7
            
            events_to_keep &= (np.any(np.sum(np.isin(cvn_obj_maps[:], [1]), axis=1) >= 5)) or (np.any(np.sum(np.isin(cvn_obj_maps[:], [1]), axis=1) == 0))
            events_to_keep &= (np.any(np.sum(np.isin(cvn_obj_maps[:], [2]), axis=1) >= 5)) or (np.any(np.sum(np.isin(cvn_obj_maps[:], [2]), axis=1) == 0))
            events_to_keep &= (np.any(np.sum(np.isin(cvn_obj_maps[:], [3]), axis=1) >= 5)) or (np.any(np.sum(np.isin(cvn_obj_maps[:], [3]), axis=1) == 0))
            events_to_keep &= (np.any(np.sum(np.isin(cvn_obj_maps[:], [4]), axis=1) >= 5)) or (np.any(np.sum(np.isin(cvn_obj_maps[:], [4]), axis=1) == 0))
            events_to_keep &= (np.any(np.sum(np.isin(cvn_obj_maps[:], [5]), axis=1) >= 5)) or (np.any(np.sum(np.isin(cvn_obj_maps[:], [5]), axis=1) == 0))

             # Crear matrices filtradas de eventos
            labels = labels[events_to_keep]
            energies = energies[events_to_keep]
            cvn_maps = cvn_maps[events_to_keep]
            cvn_obj_maps = cvn_obj_maps[events_to_keep]
            cvn_lab_maps = cvn_lab_maps[events_to_keep]

            # Guardar los datos filtrados en un archivo de salida
            hf.create_dataset('label', data=labels, compression='gzip')
            hf.create_dataset('energy', data=energies, compression='gzip')
            hf.create_dataset('cvnmap', data=cvn_maps, compression='gzip')
            hf.create_dataset('cvnobjmap', data=cvn_obj_maps, compression='gzip')
            hf.create_dataset('cvnlabmap', data=cvn_lab_maps, compression='gzip')
            
load_nova(files)