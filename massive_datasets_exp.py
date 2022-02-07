import yaml
from train import train
import copy
from argparse import ArgumentParser
from math import ceil
from pytorch_lightning.callbacks import EarlyStopping

CONFIG_PATH = "configs/massive_datasets_fig12.yaml"


def get_exp_name(train_ds_size, complexity, holdout):
    ds_size_for_name = train_ds_size if train_ds_size < 1e5 else f"{train_ds_size:.1e}"
    return f"md-fig12_ds={ds_size_for_name}_m={complexity}_ho={holdout}"


def single_run(config, train_ds_size, complexity, holdout):
    # deep copy config so changes will not affect other runs
    config = copy.deepcopy(config)

    data_cfg = config['data']
    train_cfg = config['training']

    train_batch_size = train_cfg["train_batch_size"]
    log_every_n_steps = train_cfg["log_every_n_steps"]
    val_check_interval = train_cfg["val_check_interval"]
    epochs = train_cfg["epochs"]

    exp_name = get_exp_name(train_ds_size, complexity, holdout)

    # set config params
    config["name"] = exp_name
    data_cfg["train_params"] = {}
    data_cfg["train_params"]["size"] = train_ds_size
    # set complexity
    data_cfg["train_params"]["complexity"] = complexity
    data_cfg["val_params"]["complexity"] = complexity
    # set holdout
    data_cfg["train_params"]["holdout"] = holdout
    data_cfg["val_params"]["holdout"] = holdout
    # set val adversarial if holdout > 0
    data_cfg["val_params"]["adversarial"] = holdout > 0

    # trim train batch size for smaller datasets
    modified_train_batch_size = min(train_batch_size, train_ds_size)
    train_cfg["train_batch_size"] = modified_train_batch_size

    # log every epoch for small datasets
    steps_in_epoch = train_ds_size / modified_train_batch_size
    train_cfg["log_every_n_steps"] = min(log_every_n_steps, steps_in_epoch)

    # modify val_check_interval if it's not enough steps
    # float = fraction of train epoch, int = number of steps
    val_check_interval_steps = \
        val_check_interval if isinstance(val_check_interval, int) else steps_in_epoch * val_check_interval

    # modify val_check_interval
    if val_check_interval_steps < 50:
        if 'val_check_interval' in train_cfg:
            del train_cfg["val_check_interval"]
        train_cfg["check_val_every_n_epoch"] = ceil(50.0 / steps_in_epoch)
    else:
        train_cfg['val_check_interval'] = val_check_interval

    # modify epochs according to minimum steps requirement
    if epochs * steps_in_epoch < 800:
        train_cfg['epochs'] = ceil(800.0 / steps_in_epoch)

    # print experiment details
    str_to_print = f"*** {exp_name} | Size: {train_ds_size:.1e} ({train_ds_size}), Complexity: {complexity}, Holdout: {holdout} ***"
    str_to_print = (' ' * 5) + str_to_print + (' ' * 5)
    print('\n' + '-' * len(str_to_print))
    print(str_to_print)
    print('-' * len(str_to_print) + '\n')

    # Train ⚡
    train(config)


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=None, help="config path", required=True)
    parser.add_argument("-s", "--size", type=float, default=None, help="train dataset size", required=True)
    parser.add_argument("-m", "--complexity", type=int, default=None, help="complexity (m)", required=True)
    parser.add_argument("-ho", "--holdout", type=int, default=None, help="holdout", required=True)
    args = parser.parse_args()

    print('\n' + '>' * 15)
    print(f'\tnow running massive_datasets_exp.py --size {int(args.size)} --complexity {args.complexity} --holdout {args.holdout}')
    print(f'\t\t--config {args.config}')
    print('>' * 15 + '\n')

    try:
        # load common config
        with open(args.config, 'r') as stream:
            config = yaml.safe_load(stream)

        single_run(config, int(args.size), args.complexity, args.holdout)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")


if __name__ == "__main__":
    main()
