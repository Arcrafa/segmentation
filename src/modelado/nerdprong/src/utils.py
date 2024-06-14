


import os
import numpy as np
import torch
def train_val_test_split(data, val_size=0.1, test_size=0.1):

    
    # determina el tamaño del conjunto de validación y prueba en función del tamaño total de los datos y los tamaños deseados de los conjuntos de validación y prueba
    val_len = int(len(data) * val_size)
    test_len = int(len(data) * test_size)
    
    # divide los datos en conjuntos de entrenamiento, validación y prueba
    train_data = data[:-val_len-test_len]
    val_data = data[-val_len-test_len:-test_len]
    test_data = data[-test_len:]
    
    return train_data, val_data, test_data

def load_train_val_test_list_files(dataset_path,num_files=None):
    files=os.listdir(dataset_path)
    files = [dataset_path+f for f in files]
    print(len(files))
    if num_files!=None: files=files[0:num_files]

    ftrain,fval,ftest = train_val_test_split(files)
    return ftrain,fval,ftest
    
def to_categorical(y, num_classes=None, dtype="float32"):
    y = np.array(y, dtype="int")
    input_shape = y.shape

    # Shrink the last dimension if the shape is (..., 1).
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])

    y = y.reshape(-1)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def collate_fn(batch):
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    pixel_mask = torch.stack([example["pixel_mask"] for example in batch])
    class_labels = [example["class_labels"] for example in batch]
    mask_labels = [example["mask_labels"] for example in batch]
    return {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "class_labels": class_labels,"mask_labels": mask_labels}