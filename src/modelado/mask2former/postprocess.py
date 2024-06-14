import pytorch_lightning as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import *
from datasets import *
import os 
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import pandas as pd
def post_process_instance_segmentation(
        outputs,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
        overlap_mask_area_threshold: float = 0.8,
        target_sizes: Optional[List[Tuple[int, int]]] = None,
        return_coco_annotation: Optional[bool] = False,
        return_binary_maps: Optional[bool] = False,
    ) -> List[Dict]:
        """
        Converts the output of [`Mask2FormerForUniversalSegmentationOutput`] into instance segmentation predictions.
        Only supports PyTorch.

        Args:
            outputs ([`Mask2FormerForUniversalSegmentation`]):
                Raw outputs of the model.
            threshold (`float`, *optional*, defaults to 0.5):
                The probability score threshold to keep predicted instance masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.8):
                The overlap mask area threshold to merge or discard small disconnected parts within each binary
                instance mask.
            target_sizes (`List[Tuple]`, *optional*):
                List of length (batch_size), where each list item (`Tuple[int, int]]`) corresponds to the requested
                final size (height, width) of each prediction. If left to None, predictions will not be resized.
            return_coco_annotation (`bool`, *optional*, defaults to `False`):
                If set to `True`, segmentation maps are returned in COCO run-length encoding (RLE) format.
            return_binary_maps (`bool`, *optional*, defaults to `False`):
                If set to `True`, segmentation maps are returned as a concatenated tensor of binary segmentation maps
                (one per detected instance).
        Returns:
            `List[Dict]`: A list of dictionaries, one per image, each dictionary containing two keys:
            - **segmentation** -- A tensor of shape `(height, width)` where each pixel represents a `segment_id` or
              `List[List]` run-length encoding (RLE) of the segmentation map if return_coco_annotation is set to
              `True`. Set to `None` if no mask if found above `threshold`.
            - **segments_info** -- A dictionary that contains additional information on each segment.
                - **id** -- An integer representing the `segment_id`.
                - **label_id** -- An integer representing the label / semantic class id corresponding to `segment_id`.
                - **score** -- Prediction score of segment with `segment_id`.
        """
        if return_coco_annotation and return_binary_maps:
            raise ValueError("return_coco_annotation and return_binary_maps can not be both set to True.")

        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits, size=(384, 384), mode="bilinear", align_corners=False
        )

        device = masks_queries_logits.device
        num_classes = class_queries_logits.shape[-1] - 1
        num_queries = class_queries_logits.shape[-2]

        # Loop over items in batch size
        results: List[Dict[str, TensorType]] = []

        for i in range(class_queries_logits.shape[0]):
            mask_pred = masks_queries_logits[i]
            mask_cls = class_queries_logits[i]
            
            scores = torch.nn.functional.softmax(mask_cls, dim=-1)[:, :-1]
            scores = torch.round(scores , decimals=3)
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_queries, sorted=False)
            labels_per_image = labels[topk_indices]
            
            
            
            
            topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
            scores=scores[topk_indices]
            
            mask_pred = mask_pred[topk_indices]
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores_per_image * mask_scores_per_image
            pred_classes = labels_per_image

            segmentation = torch.zeros((384, 384)) - 1
            if target_sizes is not None:
                segmentation = torch.zeros(target_sizes[i]) - 1
                pred_masks = torch.nn.functional.interpolate(
                    pred_masks.unsqueeze(0), size=target_sizes[i], mode="nearest"
                )[0]

            instance_maps, segments = [], []
            current_segment_id = 0
            for j in range(num_queries):
                score = pred_scores[j].item()

                if not torch.all(pred_masks[j] == 0) and score >= threshold:
                    segmentation[pred_masks[j] == 1] = current_segment_id
                    segments.append(
                        {
                            "id": current_segment_id,
                            "label_id": pred_classes[j].item(),
                            "was_fused": False,
                            "score": round(score, 6),
                            "softmax":scores[j]
                        }
                    )
                    current_segment_id += 1
                    instance_maps.append(pred_masks[j])
                    # Return segmentation map in run-length encoding (RLE) format
                    if return_coco_annotation:
                        segmentation = convert_segmentation_to_rle(segmentation)

            # Return a concatenated tensor of binary instance maps
            if return_binary_maps and len(instance_maps) != 0:
                segmentation = torch.stack(instance_maps, dim=0)
            
            

            results.append({"segmentation": segmentation, "segments_info": segments,"instance_maps":instance_maps})
        return results
 

def reverse_subsample(mask, bbox=None, image=None):
    
    # Invertir la reducción de la máscara
    mask = mask.repeat(3,axis=0).repeat(3,axis=1)

    # Invertir la reducción de la imagen
    if image is not None:
        image =image.repeat(3,axis=0).repeat(3,axis=1)

    # Invertir la reducción de las cajas delimitadoras (bbox)
    if bbox is not None:
        bbox = bbox * 3
        bbox = np.asarray(list(map(lambda x: [x[0], x[1], x[2] - 0.5, x[3] - 0.5], bbox)))

    return mask, bbox, image
def subsample(mask, bbox=None, image=None):
    rows, cols, _ = mask.shape
    mask = mask.reshape(rows//3,3,cols//3,3,-1)
    mask = mask.any(axis=(1,3))

    if image is not None:
        image = image.reshape(rows//3,3,cols//3,3,-1)
        image = image.sum(axis=(1,3))/9

    if bbox is not None:
        bbox = bbox/3
        bbox = np.asarray(list(map(lambda x: [int(x[0]), int(x[1]), int(x[2]+0.5), int(x[3]+0.5)], bbox)))

    return mask, bbox, image

def post_process_nova(outputs_model,images):
    def dist(r,c,mask):
        row,col = np.where(mask>0)
        return min( map(lambda pt: np.sqrt((pt[0]-r)**2 + (pt[1]-c)**2), zip(row,col)) )
        
    outputs_post_process = post_process_instance_segmentation(outputs_model,target_sizes=[[255,240]]*60,threshold=0.5,return_binary_maps=False)
    outputs_post_process_nova=[]
    for idx,output in enumerate(outputs_post_process):
        try:
            output['instance_maps']=np.array(torch.stack(output['instance_maps'], dim=2))
            # Suppress Background
            #output['segmentation'][images[idx][1,:,:] == 0] = -1
            output['instance_maps'][images[idx][1,:,:] == 0] = np.zeros(output['instance_maps'].shape[2])
            
            # Downsample to hit level
            output['instance_maps'], _, image = subsample(output['instance_maps'], None, images[idx].permute(1, 2, 0))
            
            
            # Remove false positives
            bad = []
            for i,j in itertools.combinations(np.arange(output['instance_maps'].shape[2]),2):
                # Grab the pair of masks and compute area
                m1,m2  = output['instance_maps'][:,:,i], output['instance_maps'][:,:,j]
                a1,a2 = np.sum(m1), np.sum(m2)
                ac = np.sum(m1 & m2)
        
                # If cluster is direct subset of another and of sufficient size...
                if (ac>=a1-1 and ac>=0.1*a2) or (ac>=a2-1 and ac>=0.1*a1):
                    if a1>=a2-1 and a2>=a1-1:
                        # Grab the largest score of identical clusters
                        if output['segments_info'][i]['score'] > output['segments_info'][j]['score']:
                            bad.append(j)
                        else:
                            bad.append(i)
                    else:
                        # or the larger of two clusters
                        if a1 > a2:
                            bad.append(j)
                        else:
                            bad.append(i)
            
            # and delete them
            output['instance_maps'] = np.delete(output['instance_maps'],bad,axis=2)
            output['segments_info'] = np.delete(output['segments_info'],bad,axis=0)
            #r['rois']      = np.delete(r['rois'],bad,axis=0)
            #r['class_ids'] = np.delete(r['class_ids'],bad)
            #r['scores']    = np.delete(r['scores'],bad)
    
            # Identify unclustered hits
    
            #print('imagen condicion ',image.shape)
            
            row,col = np.where((image[:,:,2] > 0) & ~output['instance_maps'].any(axis=2))
            #print(row,col)
            #plt.imshow((image[:,:,2] > 0) & ~output['instance_maps'].any(axis=2))
            #plt.show()
            
            
            bbox=generar_cuadros_delimitadores(torch.tensor(output['instance_maps']).permute(2, 0, 1))
            
            
            for row,col in zip(row,col):
                cont=[]
                for n, [y1, x1, y2, x2] in enumerate(bbox):
                    if row>=y1 and row<=y2 and col>=x1 and col<=x2:
                        cont.append(n)
                if len(cont)==1:
                    
                    
                    if (dist(row,col,output['instance_maps'][:,:,cont[0]]) < 10 and output['segments_info'][cont[0]]['label_id'] in (1,7)) or \
                        dist(row,col,output['instance_maps'][:,:,cont[0]]) < 3.5:
                        
                        output['instance_maps'][row,col,cont[0]] = 1
                elif len(cont)>1:
                    m=256
                    for id in cont:
                        d = dist(row,col,output['instance_maps'][:,:,id])
                        if d < m:
                            m = d
                            i = id
                    output['instance_maps'][row,col,i] = 1
    
            
            output['instance_maps'], _, _ = reverse_subsample(output['instance_maps'], None, None)
            
            output['instance_maps'] = np.transpose(output['instance_maps'], (2, 0, 1))
        except:
            outputs_post_process_nova.append(output)
            continue
        outputs_post_process_nova.append(output)
    return outputs_post_process
