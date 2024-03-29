import math
import os

import pytorch_lightning
import torch
import yaml
import time

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from callbacks import CustomEarlyStoppingCallback, LoggerCallback

from models import models_registry
from datasets import tasks_registry
from lightning_classifier import Classifer
from lightning_classifier_with_scheduler import ClassiferWithScheduler

from argparse import ArgumentParser

import multiprocessing

AVAIL_GPUS = min(1, torch.cuda.device_count())
AVAIL_CPUS = multiprocessing.cpu_count()


def _get_new_int_seed():
    return int(time.time() * 10000000) % (2 ** 32 - 1)


def setup_loaders(config: dict):
    torch.manual_seed(config['random_seed'])

    data_cfg = config['data']
    train_cfg = config['training']

    num_workers = (4 * AVAIL_GPUS) if (AVAIL_GPUS > 0) else AVAIL_CPUS

    shuffle = train_cfg['shuffle'] if 'shuffle' in train_cfg else False

    train_ds = tasks_registry[data_cfg['task']]('train_ds', **data_cfg['train_params'])
    train_batch_size = train_cfg['train_batch_size']
    train_loader = DataLoader(train_ds, batch_size=train_batch_size,
                              pin_memory=True,
                              num_workers=num_workers,
                              shuffle=shuffle)

    eval_batch_size = train_cfg['eval_batch_size'] if 'eval_batch_size' in train_cfg else train_batch_size
    val_ds = tasks_registry[data_cfg['task']]('val_ds', **data_cfg['val_params'])
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size,
                            pin_memory=True,
                            num_workers=num_workers)

    return train_loader, val_loader


def steps_in_epochs(config: dict):
    dataset_size = config['data']['train_params']['size']
    train_batch_size = config['training']['train_batch_size']

    return math.ceil(dataset_size / train_batch_size)


def setup_callbacks(callback_cfg, logger):
    # create callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger_callback = LoggerCallback()
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(logger.experiment.dir, 'Checkpoints'),
                                          filename='{epoch}-val_acc={val_acc:.4f}',
                                          monitor='val_acc',
                                          mode='max',
                                          **callback_cfg['checkpoint_saving'])

    callbacks = [lr_monitor,
                 logger_callback,
                 checkpoint_callback]

    # append early stopping callback if found in config
    if 'early_stopping' in callback_cfg:
        early_stopping_cfg = callback_cfg['early_stopping']
        verbose = early_stopping_cfg['verbose'] if 'verbose' in early_stopping_cfg else False
        earlyStoppingCallback = CustomEarlyStoppingCallback(hard_patience=early_stopping_cfg['hard_patience'],
                                                            soft_patience=early_stopping_cfg['soft_patience'],
                                                            verbose=verbose)
        callbacks.append(earlyStoppingCallback)

    return callbacks


def train(config: dict):
    print(f'CPUS: {AVAIL_CPUS}, GPUS: {AVAIL_GPUS}')

    # set random seed
    # notice: seed myst be (int) for seed_everything, but the default seed is (long)
    is_manual_seed = 'random_seed' in config and config['random_seed'] is not None
    random_seed = int(config['random_seed']) if is_manual_seed else _get_new_int_seed()
    pytorch_lightning.seed_everything(random_seed)
    config['random_seed'] = random_seed
    print(f'Random seed: {random_seed} {"(Manual)" if is_manual_seed else "(Auto)"}')

    # Init our model
    train_cfg = config['training']

    net = models_registry[config['model']]()
    cls = Classifer(
        net,
        train_cfg['optimizer_type'],
        train_cfg['optimizer_params']
    )

    scheduler_params = train_cfg.get('scheduler')
    num_steps_in_epoch = steps_in_epochs(config)
    if scheduler_params is not None:
        scheduler_params['num_warmup_steps'] = scheduler_params['num_warmup_epochs'] * num_steps_in_epoch
        scheduler_params['num_training_steps'] = train_cfg['epochs'] * num_steps_in_epoch
        cls = ClassiferWithScheduler(
            net,
            train_cfg['optimizer_type'],
            train_cfg['optimizer_params'],
            scheduler_params
        )

    # Init DataLoader from Dataset
    t0 = time.time()
    train_loader, val_loader = setup_loaders(config)
    t1 = time.time()
    print(f"setup_loaders() time: {t1 - t0:.3f} (sec)")

    # setup WandB logger
    wandb_logger = WandbLogger(project="pointer-value-retrieval",
                               entity="deep-learning-course-project",
                               name=config["name"],
                               group=config["group"])
    # update experiment name if it was auto-generated
    config["name"] = wandb_logger.experiment.name
    
    print(f'Experiment group: {config["group"]}')
    print(f'Experiment name: {config["name"]}')

    # log the config before training starts
    wandb_logger.experiment.config.update(config)

    callbacks = setup_callbacks(train_cfg['callbacks'], wandb_logger)

    # determine validation frequency
    if 'val_check_interval' in train_cfg and 'check_val_every_n_epoch' in train_cfg:
        raise Exception("'val_check_interval' and 'check_val_every_n_epoch' shouldn't be used simultaneously")
    val_check_interval = train_cfg['val_check_interval'] if 'val_check_interval' in train_cfg else 1.0
    check_val_every_n_epoch = train_cfg['check_val_every_n_epoch'] if 'check_val_every_n_epoch' in train_cfg else 1

    # Initialize a trainer
    trainer = Trainer(
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
        log_every_n_steps=train_cfg['log_every_n_steps'],
        gpus=AVAIL_GPUS,
        precision=16 if (AVAIL_GPUS > 0 and train_cfg['mixed_precision']) else 32,
        max_epochs=train_cfg['epochs'],
        logger=wandb_logger,
        callbacks=callbacks,
    )
    # add num_steps_in_epoch to trainer, used in logger_callback
    trainer.num_steps_in_epoch = num_steps_in_epoch

    # Train the model ⚡
    t0 = time.time()
    trainer.fit(cls, train_loader, val_loader)
    t1 = time.time()
    print(f"train.fit() time: {t1 - t0:.3f} (sec)")


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default='./configs/vector_test.yaml',
                        help="path to yaml config file")
    parser.add_argument("-s", "--seed", type=int, default=None, help="set manual random seed")
    parser.add_argument("-g", "--group", type=str, default=None, help="WandbLogger group")
    parser.add_argument("-n", "--name", type=str, default=None, help="WandbLogger name")
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("Missing config file path. add it with -c or --config.")

    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)

    # override config with console parameters
    if args.seed is not None:
        config['random_seed'] = args.seed
    if args.name is not None:
        config["name"] = args.name
    if args.group is not None:
        config["group"] = args.group

    train(config)


if __name__ == "__main__":
    main()
