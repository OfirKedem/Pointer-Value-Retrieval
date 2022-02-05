from pytorch_lightning.callbacks import Callback
import wandb


class LoggerCallback(Callback):
    def __init__(self, ):
        super().__init__()

    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # f_step is the fraction of 'Step' the appears in wandb
        # log only to progress bar
        pl_module.log("f_step", (pl_module.global_step + 1) / trainer.log_every_n_steps,
                      on_step=True, on_epoch=False, prog_bar=True, logger=False)
