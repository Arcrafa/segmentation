import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import *
from datasets import *
from postprocess import *

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {
        "pixel_values": pixel_values, 
        "pixel_mask": pixel_mask, 
        "class_labels": class_labels,
        "mask_labels": mask_labels
    }

def get_test_dataloader(dataset_path, batch_size=60):
    files = [os.path.join(dataset_path, f) for f in sorted(os.listdir(dataset_path))]
    test_dataset = ImageSegmentationDataset(files)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

def process_batch(outputs, real, ds, image_id, num_classes, CONF_THRESHOLD, IOU_THRESHOLD):
    for im_id_batch in range(len(outputs)):
        image_id += 1
        
        try:
            pred_scores = torch.tensor([d['score'] for d in outputs[im_id_batch]['segments_info']]).numpy()
            pred_labels = torch.tensor([d['label_id'] for d in outputs[im_id_batch]['segments_info']]).numpy()
            pred_probs = [d['softmax'].tolist() for d in outputs[im_id_batch]['segments_info']]
            pred_masks = outputs[im_id_batch]['instance_maps']
            target_labels = torch.tensor(real['class_labels'][im_id_batch].tolist()).numpy()
            target_masks = torch.tensor(real['mask_labels'][im_id_batch], dtype=torch.uint8).numpy()
        except:
            continue
        
        try:
            all_ious = mask_iou_calc(target_masks, pred_masks)
        except:
            #print(f'la imagen: {image_id} genera pred_masks de forma invalida : {pred_masks}')
            continue

        want_idx = np.where(all_ious > IOU_THRESHOLD)
        all_matches = np.array([[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]] for i in range(want_idx[0].shape[0])])
        
        if all_matches.shape[0] > 0:
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]
            all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]
            all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

        for i, label in enumerate(target_labels):
            gt_class = target_labels[i]
            if all_matches.shape[0] > 0:
                matching_rows = all_matches[all_matches[:, 0] == i]
                if matching_rows.shape[0] > 0:
                    best_match_index = np.argmax(matching_rows[:, 2])
                    best_match = matching_rows[best_match_index, 2]
                    probs = pred_probs[int(matching_rows[best_match_index, 1])]
                    ds['clase'].append(gt_class)
                    ds['probs'].append(probs)
                    ds['image_id'].append(image_id)
                    ds['conf_score'].append(pred_scores[int(matching_rows[best_match_index, 1])])
                    ds['iou_score'].append(matching_rows[best_match_index, 2])
                else:
                    ds['clase'].append(gt_class)
                    ds['probs'].append([0, 0, 0, 0, 0])
                    ds['image_id'].append(image_id)
                    ds['conf_score'].append(0)
                    ds['iou_score'].append(0)
            else:
                print(f'no detecto ninguna particula del: {image_id}')

    return ds, image_id

def predict_and_save(plmodel, dataloader, num_classes, CONF_THRESHOLD, IOU_THRESHOLD, output_file='predicciones.csv'):
    ds = {'clase': [], 'probs': [], 'image_id': [], 'conf_score': [], 'iou_score': []}
    image_id = -1

    for idx, batch in enumerate(tqdm(dataloader)):
        plmodel.model.eval()
        with torch.no_grad():
            outputs = plmodel.model(batch["pixel_values"])
        outputs = post_process_nova(outputs, batch["pixel_values"])
        ds, image_id = process_batch(outputs, batch, ds, image_id, num_classes, CONF_THRESHOLD, IOU_THRESHOLD)

    df = pd.DataFrame(ds)
    expanded_probs = df['probs'].apply(pd.Series)
    df_expanded = pd.concat([df['clase'], expanded_probs, df['image_id'], df['conf_score'], df['iou_score']], axis=1)
    df_expanded.to_csv(output_file, index=False)


dataset_path ='/wclustre/nova/users/rafaelma2/NOvA-Clean/datos/procesados/dataset_test/'
ckpt_path='/wclustre/nova/users/rafaelma2/NOvA-Clean/modelos/tb_logs/maskformernova_ag/version_0/checkpoints/epoch=0-step=7.ckpt'
ckpt_path='/wclustre/nova/users/rafaelma/main/tb_logs/Mask2FormerNova_tunedv2_ag/version_3/checkpoints/epoch=29-step=35760.ckpt'

test_dataloader = get_test_dataloader(dataset_path)
segmenter = Mask2FormerNova.load_from_checkpoint(ckpt_path)

predict_and_save(segmenter, test_dataloader, num_classes=5, CONF_THRESHOLD=0.5, IOU_THRESHOLD=0.1, output_file='predicciones.csv')
