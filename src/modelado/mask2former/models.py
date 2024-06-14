from torch.optim import Adam
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig
import pytorch_lightning as pl



import pytorch_lightning as pl

class Mask2FormerNova(pl.LightningModule):
    def __init__(self, lr=5e-5):
        super().__init__()
        self.lr = lr
        self.id2label = {
            0: 'Electron',
            1: 'Muon',
            2: 'Proton',
            3: 'Photon',
            4: 'Pion'
        }

        configuration = Mask2FormerConfig()
        configuration.id2label = self.id2label
        configuration.backbone_config.depths = [1, 1, 1, 1]
        configuration.num_queries = 30
        configuration.encoder_layers = 2
        configuration.decoder_layers = 3
        configuration.num_hidden_layers = 3
        model_path = "/wclustre/nova/users/rafaelma2/NOvA-Clean/modelos/m2fpre"
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            model_path,
            ignore_mismatched_sizes=True,
            config=configuration
        )
    def forward(self, pixel_values, mask_labels, class_labels):
        return self.model(pixel_values=pixel_values, mask_labels=mask_labels, class_labels=class_labels)

    def training_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"], mask_labels=batch["mask_labels"],
                             class_labels=batch["class_labels"])
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"], mask_labels=batch["mask_labels"],
                             class_labels=batch["class_labels"])
        loss = outputs.loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        return optimizer




