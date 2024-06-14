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


def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection) / torch.sum(union)
    return iou


def confusion(outputs, labels, threshold):
    confusion_matrix = np.zeros((7, 7))
    # f1_score, precision, recall=0,0,0
    TP = 0
    FP = 0
    FN = 0
    try:
        labels_pred = [d['label_id'] for d in outputs['segments_info']]
        labels_true = labels['class_labels'][0].tolist()
        masks_true = torch.tensor(labels['mask_labels'][0])
        masks_pred = torch.stack([torch.tensor((outputs['segmentation'].numpy() == i) * 1) for i in
                                  range(int(torch.max(outputs['segmentation'])) + 1)])

        for i in range(len(masks_true)):
            ious = []
            for j in range(len(masks_pred)):
                iou = calculate_iou(masks_true[i], masks_pred[j])
                ious.append(iou)
            max_iou = max(ious)
            if max_iou >= threshold and labels_true[i] == labels_pred[ious.index(max_iou)]:
                TP += 1
                confusion_matrix[labels_true[i]][labels_pred[ious.index(max_iou)]] += 1
            else:
                FN += 1
                confusion_matrix[labels_true[i]][labels_pred[ious.index(max_iou)]] += 1

        for i in range(len(masks_pred)):
            ious = []
            for j in range(len(masks_true)):
                iou = calculate_iou(masks_pred[i], masks_true[j])
                ious.append(iou)
            max_iou = max(ious)
            if max_iou < threshold:
                FP += 1

                confusion_matrix[labels_true[ious.index(max_iou)]][labels_pred[i]] += 1
        # try:
        #    precision = TP / (TP + FP)+0.01
        #    recall = TP / (TP + FN)+0.01
        #    f1_score = 2 * precision * recall / (precision + recall)+0.01
        # except:
        #    pass
    except:
        pass

    return TP, FP, FN, confusion_matrix


def mask_iou_calc(masks1, masks2):
    """
    Return intersection-over-union (Jaccard index) of masks.
    Both sets of masks are expected to have the same shape.
    Arguments:
        masks1 (Array[N, H, W])
        masks2 (Array[M, H, W])
    Returns:
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in masks1 and masks2
    """
    area1 = np.sum(masks1, axis=(1, 2))  # [N]
    area2 = np.sum(masks2, axis=(1, 2))  # [M]

    # Compute intersection between all masks combinations
    intersection = np.einsum('nij,mij->nm', masks1, masks2)  # [N, M]

    union = area1[:, np.newaxis] + area2 - intersection
    iou = intersection / union

    return iou


class ConfusionMatrix:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.1):
        self.matrix = np.zeros((num_classes + 1, num_classes + 1))
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD

    def process_batch(self, outputs, real,batch_size):
        
        for i in range(batch_size):
            try:
                pred_scores = torch.tensor([d['score'] for d in outputs[i]['segments_info']]).numpy()
                pred_labels = torch.tensor([d['label_id'] for d in outputs[i]['segments_info']]).numpy()


                pred_masks = torch.stack([torch.tensor((outputs[i]['segmentation'].numpy() == x) * 1, dtype=torch.uint8) for x in range(int(torch.max(outputs[i]['segmentation'])) + 1)]).numpy()

                target_labels = torch.tensor(real['class_labels'][i].tolist()).numpy()
                target_masks = torch.tensor(real['mask_labels'][i], dtype=torch.uint8).numpy()
            
            except:
                continue

            all_ious = mask_iou_calc(target_masks, pred_masks)
            want_idx = np.where(all_ious > self.IOU_THRESHOLD)

            all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                           for i in range(want_idx[0].shape[0])]

            all_matches = np.array(all_matches)
            if all_matches.shape[0] > 0:  # if there is match
                all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

                all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

                all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

                all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

            for i, label in enumerate(target_masks):
                gt_class = target_labels[i]
                if all_matches.shape[0] > 0 and all_matches[all_matches[:, 0] == i].shape[0] == 1:
                    detection_class = pred_labels[int(all_matches[all_matches[:, 0] == i, 1][0])]
                    self.matrix[detection_class, gt_class] += 1
                else:
                    self.matrix[self.num_classes, gt_class] += 1
                    pass
            for i, detection in enumerate(pred_masks):
                if all_matches.shape[0] and all_matches[all_matches[:, 1] == i].shape[0] == 0:
                    detection_class = pred_labels[i]
                    self.matrix[detection_class, self.num_classes] += 1

    def return_matrix(self):
        return self.matrix

    def print_matrix(self):
        for i in range(self.num_classes + 1):
            print(' '.join(map(str, self.matrix[i])))