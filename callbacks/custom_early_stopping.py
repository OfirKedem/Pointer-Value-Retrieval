from pytorch_lightning.callbacks import Callback
import wandb


class CustomEarlyStoppingCallback(Callback):

    def __init__(self, hard_patience=1, soft_patience=1, min_epochs=2, verbose=False):
        super().__init__()
        self.hard_patience = hard_patience  # checked every train epoch
        self.soft_patience = soft_patience  # check every val epoch
        self.min_epochs = min_epochs  # start checking after min_epochs
        self.verbose = verbose

        self.metrics = {
            "train_acc_epoch": 0.0,
            "val_acc": 0.0,
            "val_loss": float('inf')
        }

        self.hard_counter = 0
        self.soft_counter = 0
        self.prev_val_loss = float('inf')

    def _hard_stopping_condition(self):
        train_acc_epoch = self.metrics["train_acc_epoch"]
        val_acc = self.metrics["val_acc"]

        # reached full accuracy on both train and val
        return train_acc_epoch > 0.99 and val_acc > 0.99

    def _soft_stopping_condition(self):
        train_acc_epoch = self.metrics["train_acc_epoch"]
        val_loss = self.metrics["val_loss"]

        is_val_loss_decreasing = val_loss < self.prev_val_loss

        # train reached full accuracy but val is not improving
        return train_acc_epoch > 0.99 and not is_val_loss_decreasing

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # track train metrics
        self.metrics["train_acc_epoch"] = trainer.logged_metrics["train_acc_epoch"]

        # don't check until min_epochs is passed
        if trainer.current_epoch < self.min_epochs:
            return

        if self._hard_stopping_condition():
            self.hard_counter += 1

            if self.verbose:
                print(f"\n+++ hard stopping conditions were met. hard_counter: {self.hard_counter}")

            # check patience and stop if reached
            if self.hard_counter >= self.hard_patience:
                print(f"\n\t *** hard stopping condition reached - stopping training!")
                trainer.should_stop = True
        else:
            self.hard_counter = 0

            if self.verbose:
                print(f"\n--- hard stopping conditions were NOT met. hard_counter: {self.hard_counter}")

        # log the counts left until stopping
        pl_module.log("hard_patience", float(self.hard_patience - self.hard_counter),
                      prog_bar=True, logger=True)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # track train metrics
        self.metrics["val_acc"] = trainer.logged_metrics["val_acc"]
        self.metrics["val_loss"] = trainer.logged_metrics["val_loss"]

        # don't check until min_epochs is passed
        if trainer.current_epoch < self.min_epochs:
            return

        # update soft counter
        if self._soft_stopping_condition():
            self.soft_counter += 1

            if self.verbose:
                print(f"\n+++ soft stopping conditions were met. soft_counter: {self.soft_counter}\n")

            # check patience and stop if reached
            if self.soft_counter >= self.soft_patience:
                print(f"\n\t *** soft stopping condition reached - stopping training!")
                trainer.should_stop = True
        else:
            self.soft_counter = 0

            if self.verbose:
                print(f"\n--- soft stopping conditions were NOT met. soft_counter: {self.soft_counter}\n")

        # save prev value for next epoch
        val_loss = self.metrics["val_loss"]
        self.prev_val_loss = val_loss

        # log the counts left until stopping
        pl_module.log("soft_patience", float(self.soft_patience - self.soft_counter),
                      prog_bar=True, logger=True)


