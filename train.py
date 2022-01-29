import math

import torch
import wandb
import yaml

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from models import models_registry
from datasets import tasks_registry
from lightning_classifier import Classifer
from lightning_classifier_with_scheduler import ClassiferWithScheduler

import multiprocessing

AVAIL_GPUS = min(1, torch.cuda.device_count())
AVAIL_CPUS = multiprocessing.cpu_count()


def setup_loaders(config: dict):
    data_cfg = config['data']
    train_cfg = config['training']

    train_ds = tasks_registry[data_cfg['task']](**data_cfg['train_params'])
    train_loader = DataLoader(train_ds, batch_size=train_cfg['batch_size'],
                              pin_memory=True,
                              num_workers=AVAIL_CPUS-1)

    val_ds = tasks_registry[data_cfg['task']](**data_cfg['val_params'])
    val_loader = DataLoader(val_ds, batch_size=train_cfg['batch_size'],
                            pin_memory=True,
                            num_workers=AVAIL_CPUS-1)

    return train_loader, val_loader


def steps_in_epochs(config: dict):
    dataset_size = config['data']['train_params']['size']
    batch_size = config['training']['batch_size']

    return math.ceil(dataset_size / batch_size)


def train(config: dict):
    # Init our model
    train_cfg = config['training']

    net = models_registry[config['model']]()
    cls = Classifer(
        net,
        train_cfg['optimizer_type'],
        train_cfg['optimizer_params']
    )

    scheduler_params = train_cfg.get('scheduler')
    if scheduler_params is not None:
        num_steps_in_epoch = steps_in_epochs(config)

        scheduler_params['num_warmup_steps'] = scheduler_params['num_warmup_epochs'] * num_steps_in_epoch
        scheduler_params['num_training_steps'] = train_cfg['epochs'] * num_steps_in_epoch
        cls = ClassiferWithScheduler(
            net,
            train_cfg['optimizer_type'],
            train_cfg['optimizer_params'],
            scheduler_params
        )

    # Init DataLoader from Dataset
    train_loader, val_loader = setup_loaders(config)

    # setup WandB logger
    wandb_logger = WandbLogger(save_dir=None,
                               project="pointer-value-retrieval",
                               entity="deep-learning-course-project")

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Initialize a trainer
    trainer = Trainer(
        val_check_interval=train_cfg['val_check_interval'],
        gpus=AVAIL_GPUS,
        precision=16 if AVAIL_GPUS > 0 and train_cfg['mixed_precision'] else 32,
        strategy='ddp' if AVAIL_GPUS > 1 else None,
        max_epochs=train_cfg['epochs'],
        logger=wandb_logger,
        callbacks=[lr_monitor]
    )

    # Train the model ⚡
    trainer.fit(cls, train_loader, val_loader)

    wandb.config.update(config)


def main():
    print(f'CPUS: {AVAIL_CPUS}, GPUS: {AVAIL_GPUS}')

    config_file = 'configs/vector_test.yaml'
    with open(config_file, 'r') as stream:
        cfg = yaml.safe_load(stream)
    train(cfg)


if __name__ == "__main__":
    main()
