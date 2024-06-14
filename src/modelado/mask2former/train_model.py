import multiprocessing
import torch
from torch.utils.data import DataLoader
from models import Mask2FormerNova
from datasets import ImageSegmentationDataset
import pytorch_lightning as pl
import os 

def load_train_val_test_list_files(dataset_path,val_size=0.1,test_size=0.1,num_files=None):
    # Lista todos los archivos en el directorio especificado
    files = os.listdir(dataset_path)
    
    # Une la ruta del directorio a cada nombre de archivo
    files = [os.path.join(dataset_path, f) for f in files]
    
    # Si se especifica, limita el número de archivos a cargar
    if num_files is not None:
        files = files[:num_files]

    # Calcula las longitudes de los conjuntos de validación y prueba
    val_len = int(len(files) * val_size)
    test_len = int(len(files) * test_size)
    
    # Divide la lista de archivos en conjuntos de entrenamiento, validación y prueba
    train_files = files[:-val_len-test_len]
    val_files = files[-val_len-test_len:-test_len]
    test_files = files[-test_len:]
    
    return train_files, val_files, test_files
    
def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels,"mask_labels": mask_labels}

dataset_path = '../../../datos/procesados/dataset/'

ftrain, fval, ftest = load_train_val_test_list_files(dataset_path,num_files=10)

train_dataset = ImageSegmentationDataset(ftrain, augmentation=True)
val_dataset = ImageSegmentationDataset(fval)

batch_size = 60
num_workers = multiprocessing.cpu_count()


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn, pin_memory=True)

#ckpt_path='/wclustre/nova/users/rafaelma2/NOvA-Clean/modelos/tb_logs/maskformernova_ag/version_1/checkpoints/epoch=29-step=35640.ckpt'
#print(ckpt_path)
#model=Mask2FormerNova.load_from_checkpoint(ckpt_path)

model=Mask2FormerNova()


trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp",logger=pl.loggers.TensorBoardLogger('/wclustre/nova/users/rafaelma2/NOvA-Clean/modelos/tb_logs/', name='maskformernova_ag'),max_epochs=1)


trainer.fit(model, train_dataloader, val_dataloader)



