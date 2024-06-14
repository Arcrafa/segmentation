from .datasets import *
from .utils import *
import pytorch_lightning as pl

from transformers import Mask2FormerForUniversalSegmentation

class Mask2FormerNova(pl.LightningModule):
    def __init__(self, dataset_path,num_files=300, batch_size=20, num_workers=4, lr=5e-5, CONF_THRESHOLD=0.9, IOU_THRESHOLD=0.15):
        super().__init__()
        
        ftrain,fval,ftest = load_train_val_test_list_files(dataset_path,num_files=num_files)
        self.train_dataset = ImageSegmentationDataset(ftrain)
        self.val_dataset = ImageSegmentationDataset(fval)
        self.test_dataset = ImageSegmentationDataset(ftest)
        
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.id2label = {
                    0: 'BG',
                    1: 'Electron',
                    2: 'Muon',
                    3: 'Proton',
                    4: 'Photon',
                    5: 'Pion',
                    6: 'Other'
                }
        
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-small-coco-instance",
            id2label=self.id2label,
            ignore_mismatched_sizes=True)

    def forward(self, pixel_values, mask_labels, class_labels):
        return self.model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        outputs = self(pixel_values=batch["pixel_values"], mask_labels=[labels for labels in batch["mask_labels"]],
                       class_labels=[labels for labels in batch["class_labels"]])
        loss = outputs.loss
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(pixel_values=batch["pixel_values"], mask_labels=[labels for labels in batch["mask_labels"]],
                       class_labels=[labels for labels in batch["class_labels"]])
        loss = outputs.loss
        
        self.log('val_loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True)
        
        
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,collate_fn=collate_fn, pin_memory=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,collate_fn=collate_fn, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,collate_fn=collate_fn, pin_memory=True)