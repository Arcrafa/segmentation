import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from transformers import (Mask2FormerForUniversalSegmentation,
                          MaskFormerImageProcessor)

from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from collections import defaultdict

id2label = {
                    0: 'BG',
                    1: 'Electron',
                    2: 'Muon',
                    3: 'Proton',
                    4: 'Photon',
                    5: 'Pion',
                    6: 'Other'
                }
from PIL import Image
import io
from torchvision import transforms

def plot_to_image(figure):
    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    img_tensor = transforms.ToTensor()(img)
    return img_tensor     

def create_cmap(n):
    return cm.get_cmap('inferno', n)


def draw_pred_segmentation(ax,  pred, id2label):
    ax.set_title('Segmentación predicha')
    
    cmap = create_cmap(len(pred['segments_info'])+1)
    #plt.imshow([[i for i in range(10)]], cmap=cmap)
    #plt.show()
    ax.imshow(pred['segmentation'], cmap=cmap)
    instances_counter = defaultdict(int)
    handles = []
    for segment in pred['segments_info']:
        segment_id = segment['id']
        segment_label_id = segment['label_id']
        segment_label = id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id] + 1}"
        instances_counter[segment_label_id] += 1
        color = cmap(segment_id + 1)
        handles.append(mpatches.Patch(color=color, label=label))
    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')


def draw_real_segmentation(ax,  target, id2label,index):
    
    ax.set_title('Segmentación real')
    mask_list = target['mask_labels'][index]
    class_labels = target['class_labels'][index]
    
    cmap = create_cmap(len(class_labels)+1)
    output_matrix = np.zeros(mask_list[0].shape)
    instances_counter = defaultdict(int)
    handles = []
    segments_ids = np.arange(len(class_labels))

    for mask, label, segment_id in zip(mask_list, class_labels, segments_ids):
        output_matrix[mask.to(dtype=torch.bool)] = np.full((np.count_nonzero(mask.to(dtype=torch.bool)),),
                                                           segment_id + 1)

    ax.imshow(output_matrix, cmap=cmap)
    for mask, label, segment_id in zip(mask_list, class_labels, segments_ids):
        segment_label_id = int(label)
        segment_label = id2label[segment_label_id]
        label = f"{segment_label}-{instances_counter[segment_label_id] + 1}"
        instances_counter[segment_label_id] += 1
        color = cmap(segment_id + 1)

        handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=handles, bbox_to_anchor=(1.5, 1), loc='upper right')


def draw_segmentation(pred, target, id2label,batch_size=1):
    
    for i in range(batch_size):
        # Configura la figura y los ejes
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)  # Ajusta el tamaño de la figura según tus necesidades
        fig.subplots_adjust(wspace=0.1)  # Ajusta el espacio entre las subfiguras

        draw_pred_segmentation(ax1, pred[i], id2label)
        draw_real_segmentation(ax2, target, id2label,i)

        fig.tight_layout()

        
        plt.savefig('comparacion-'+str(i+1)+'.png')

        plt.show()

# Llama a la función draw_segmentation con los argumentos adecuados
# draw_segmentation(results, real, id2label,file='fig.png')
