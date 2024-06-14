
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from tqdm import tqdm
from models import *
from datasets import *
from postprocess import * 
import os 
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
import pandas as pd

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels,"mask_labels": mask_labels}

dataset_path ='/wclustre/nova/users/rafaelma2/NOvA-Clean/datos/procesados/dataset_test/'

files=os.listdir(dataset_path)
files = [dataset_path+f for f in files]
files.sort()
test_dataset = ImageSegmentationDataset(files)
test_dataloader=DataLoader(test_dataset, batch_size=60, shuffle=False, num_workers=4,collate_fn=collate_fn)
    

ckpt_path='/wclustre/nova/users/rafaelma2/NOvA-Clean/modelos/tb_logs/maskformernova_ag/version_0/checkpoints/epoch=0-step=7.ckpt'

print(ckpt_path)


segmenter=Mask2FormerNova.load_from_checkpoint(ckpt_path)





class DsPredict:
    def __init__(self, num_classes: int, CONF_THRESHOLD=0.3, IOU_THRESHOLD=0.1):
        
        self.ds={'clase':[],'probs':[],'image_id':[],'conf_score':[],'iou_score':[]}
        self.num_classes = num_classes
        self.CONF_THRESHOLD = CONF_THRESHOLD
        self.IOU_THRESHOLD = IOU_THRESHOLD
        self.image_id=-1

    def process_batch(self, outputs, real,batch_size,batch_id):
        
        for im_id_batch in range(batch_size):
            self.image_id=self.image_id+1
            
            try:
                pred_scores = torch.tensor([d['score'] for d in outputs[im_id_batch]['segments_info']]).numpy()
                pred_labels = torch.tensor([d['label_id'] for d in outputs[im_id_batch]['segments_info']]).numpy()
                pred_probs = [d['softmax'].tolist() for d in outputs[im_id_batch]['segments_info']]
                
                #pred_masks = torch.stack([torch.tensor((outputs[im_id_batch]['segmentation'].numpy() == x) * 1, dtype=torch.uint8) for x in range(int(torch.max(outputs[im_id_batch]['segmentation'])) + 1)]).numpy()
                pred_masks=outputs[im_id_batch]['instance_maps']

                
                target_labels = torch.tensor(real['class_labels'][im_id_batch].tolist()).numpy()
                target_masks = torch.tensor(real['mask_labels'][im_id_batch], dtype=torch.uint8).numpy()
                
            except:
                
                continue
            
            

            try:
                all_ious = mask_iou_calc(target_masks, pred_masks)
            except:
                print('la imagen: ',self.image_id,' genera pred_masks de forma invalida : ',pred_masks)
                continue
            want_idx = np.where(all_ious > self.IOU_THRESHOLD)

            all_matches = [[want_idx[0][i], want_idx[1][i], all_ious[want_idx[0][i], want_idx[1][i]]]
                           for i in range(want_idx[0].shape[0])]

            all_matches = np.array(all_matches)
            if all_matches.shape[0] > 0:  # if there is match
                all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

                all_matches = all_matches[np.unique(all_matches[:, 1], return_index=True)[1]]

                all_matches = all_matches[all_matches[:, 2].argsort()[::-1]]

                all_matches = all_matches[np.unique(all_matches[:, 0], return_index=True)[1]]

            for i, label in enumerate(target_labels):
                
                gt_class = target_labels[i] 
                if all_matches.shape[0] > 0:
                        
                     # Buscar coincidencias para el índice i
                    matching_rows = all_matches[all_matches[:, 0] == i]
    
                    if matching_rows.shape[0] > 0:
                        
                        # Encontrar la coincidencia con el mayor IoU
                        best_match_index = np.argmax(matching_rows[:, 2])
                        best_match = matching_rows[best_match_index,2]
                        
                        #probs = pred_probs[int(best_match[1])]
                        probs = pred_probs[int(matching_rows[best_match_index,1])]
                        #probs = pred_probs[int(all_matches[all_matches[:, 0] == i, 1][0])]
                        
                        self.ds['clase'].append(gt_class)
                        self.ds['probs'].append(probs)
                        self.ds['image_id'].append(self.image_id)
                        
                        self.ds['conf_score'].append(pred_scores[int(matching_rows[best_match_index,1])])
                        self.ds['iou_score'].append(matching_rows[best_match_index, 2])
                    else:
                        #probs = pred_probs[int(all_matches[all_matches[:, 0] == i, 1][0])]
                        self.ds['clase'].append(gt_class)
                        self.ds['probs'].append([0, 0, 0, 0, 0])
                        self.ds['image_id'].append(self.image_id)
                        self.ds['conf_score'].append(0)
                        self.ds['iou_score'].append(0)
                        

                else:
                    print('no detecto ninguna particula del: ', self.image_id)
                    

    
    def calc_ds(self,plmodel,bach_size,test_dataloader):
        #test_dataloader=plmodel.test_dataloader()
        for idx, batch in enumerate(tqdm(test_dataloader)):
            
            # Inference
            plmodel.model.eval()
            with torch.no_grad():
                outputs = plmodel.model(batch["pixel_values"])
            # you can pass them to processor for postprocessing
            #outputs = post_process_instance_segmentation(outputs,target_sizes=[[255,240]]*bach_size,threshold=0.5)
            
            outputs=post_process_nova(outputs,batch["pixel_values"])
            self.process_batch(outputs,batch,bach_size,idx)
            
    
    def save_ds(self,filename='predicciones_m2f.csv'):
        df=pd.DataFrame(ds.ds)
        # Aplicar pd.Series sobre la columna "probs" y expandir en columnas
        expanded_probs = df['probs'].apply(pd.Series)

        # Concatenar las nuevas columnas al DataFrame original
        df_expanded = pd.concat([df['clase'], expanded_probs,df['image_id'],df['conf_score'],df['iou_score']], axis=1)
        df_expanded.to_csv(filename)
    
ds=DsPredict(5,0.5,0.1)  

ds.calc_ds(segmenter,60,test_dataloader)

pd.DataFrame(ds.ds)

ds.save_ds(filename='predicciones.csv')











