from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl


class LoggerCallback(Callback):
    def __init__(self, ):
        super().__init__()

    def on_batch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # f_step is the fraction of 'Step' the appears in wandb
        # f_step = (pl_module.global_step + 1) / trainer.log_every_n_steps
        # pl_module.log("f_step", f_step, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        # fraction of current epoch
        f_epoch = trainer.global_step / trainer.num_steps_in_epoch
        pl_module.log("f_epoch", f_epoch, on_step=True, on_epoch=False, prog_bar=True, logger=True)

