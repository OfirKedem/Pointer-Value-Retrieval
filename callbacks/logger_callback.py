from pytorch_lightning.callbacks import Callback


class LoggerCallback(Callback):
    def on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.log("log_step", (pl_module.global_step + 1) / trainer.log_every_n_steps, prog_bar=True)
