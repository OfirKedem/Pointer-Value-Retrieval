import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torchmetrics import Accuracy

optimizers = {
    'SGD': torch.optim.SGD,
    'Adam': torch.optim.Adam
}


class Classifer(pl.LightningModule):
    def __init__(self, model: nn.Module,
                 optimizer_type: str,
                 optimizer_params: dict):
        super().__init__()

        self.model = model
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(logits, y)
        self.train_acc(preds, y)

        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(logits, y)
        self.val_acc(preds, y)

        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc)

        return loss

    def configure_optimizers(self):
        optimizer = optimizers[self.optimizer_type](self.parameters(), **self.optimizer_params)

        return optimizer
