import os
from src.datasets import ImageSegmentationDataset
from src.models import Mask2FormerNova
from torch import load, device, no_grad
import sys
import numpy as np

os.environ['CURL_CA_BUNDLE'] = ''

def load_dataset():
    dataset = ImageSegmentationDataset(['../../../data/processed/dataset/trimmed_FD_nominal_FHC_nonswap.999_of_2000.h5'])
    return dataset

def load_model():
    ckpt_path = 'modelo_bruto.pt'
    model = Mask2FormerNova()
    checkpoint = load(ckpt_path, map_location=device('cpu'))  # Load the checkpoint on CPU
    model.load_state_dict(checkpoint)
    model = model.half()
    return model

def load_sample_image(dataset, image_ids):
    images = []
    for image_id in image_ids:
        image = dataset[image_id]["pixel_values"].unsqueeze(0)
        images.append(image)
    return images

def detect(model, images):
    outputs = []
    for image in images:
        output = model(image, None, None)
        outputs.append(output)
    return outputs

def run_detect(dataset, model, max_images):
    model.eval()
    with no_grad():
        for i in range(1, max_images + 1):
            image_ids = np.random.choice(len(dataset), i, replace=False)
            images = load_sample_image(dataset, image_ids)
            detections = detect(model, images)
            print(f"Processed {i} images.")

if __name__ == "__main__":
    dataset = load_dataset()
    model = load_model()
    max_images = 20  # Maximum number of images to process
    run_detect(dataset, model, max_images)
