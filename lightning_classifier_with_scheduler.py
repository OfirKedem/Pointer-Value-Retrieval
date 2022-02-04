import math

import torch
import pytorch_lightning as pl

from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
from torchmetrics import Accuracy


# taken from: https://github.com/huggingface/transformers/blob/v4.15.0/src/transformers/optimization.py#L103
def get_cosine_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5,
        last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


optimizers = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam}


class ClassiferWithScheduler(pl.LightningModule):
    def __init__(self, model: nn.Module,
                 optimizer_type: str,
                 optimizer_params: dict,
                 scheduler_param: dict):
        super().__init__()

        self.automatic_optimization = False

        self.model = model
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params
        self.scheduler_param = scheduler_param

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        opt.zero_grad()

        x, y = batch
        logits = self(x)
        preds = torch.argmax(logits, dim=-1)
        loss = F.cross_entropy(logits, y)
        self.manual_backward(loss)

        opt.step()
        sch.step()

        self.train_acc(preds, y)

        # log to wandb
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
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    self.scheduler_param['num_warmup_steps'],
                                                    self.scheduler_param['num_training_steps'])

        return [optimizer], [scheduler]
