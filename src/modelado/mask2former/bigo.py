import os
import torch
from torch.utils.data import DataLoader, Subset
from models import Mask2FormerNova
from datasets import ImageSegmentationDataset
import pytorch_lightning as pl
import big_o
os.environ['CURL_CA_BUNDLE'] = ''

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels, "mask_labels": mask_labels}

# Cargar el dataset completo
dataset = ImageSegmentationDataset(['../../../datos/procesados/dataset/trimmed_FD_nominal_FHC_nonswap.999_of_2000.h5'])
len_dataset=len(dataset)
print(len_dataset)
# Seleccionar solo el primer batch

# Cargar el modelo desde el checkpoint
ckpt_path = '/wclustre/nova/users/rafaelma2/models/m2f_final_epoch=30.ckpt'
model = Mask2FormerNova.load_from_checkpoint(ckpt_path)

# Crear el entrenador de PyTorch Lightning
trainer = pl.Trainer(accelerator="gpu", devices=1, strategy="ddp")

def predict(indices):
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=len(indices), shuffle=False, num_workers=20, collate_fn=collate_fn, pin_memory=False)
    predictions = trainer.predict(model, dataloader, return_predictions=False)

positive_int_generator = lambda n: big_o.datagen.integers(n, 0, len_dataset-1)
best, others = big_o.big_o(predict, positive_int_generator, n_repeats=10,min_n=1,max_n=500, n_measures=20)

print(best)