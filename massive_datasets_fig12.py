import yaml
from train import train
import copy
from argparse import ArgumentParser

CONFIG_PATH = "configs/massive_datasets_fig12.yaml"


def get_exp_name(train_ds_size, complexity):
    ds_size_for_name = train_ds_size if train_ds_size < 1e5 else f"{train_ds_size:.1e}"
    return f"md-fig12_ds={ds_size_for_name}_m={complexity}"


def single_run(config, train_ds_size, complexity):
    # deep copy config so changes will not affect other runs
    config = copy.deepcopy(config)

    data_cfg = config['data']
    train_cfg = config['training']

    train_batch_size = train_cfg["train_batch_size"]
    log_every_n_steps = train_cfg["log_every_n_steps"]
    val_check_interval = train_cfg["val_check_interval"]

    exp_name = get_exp_name(train_ds_size, complexity)

    # set name, dataset size & complexity
    config["name"] = exp_name
    data_cfg["train_params"]["size"] = train_ds_size
    data_cfg["train_params"]["complexity"] = complexity
    data_cfg["val_params"]["complexity"] = complexity

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

    if val_check_interval_steps < 50:
        if 'val_check_interval' in train_cfg:
            del train_cfg["val_check_interval"]
        train_cfg["check_val_every_n_epoch"] = int(50.0 / steps_in_epoch)
    else:
        train_cfg['val_check_interval'] = val_check_interval

    # print experiment details
    str_to_print = f"*** {exp_name} | Size: {train_ds_size:.1e} ({train_ds_size}), Complexity: {complexity} ***"
    str_to_print = (' ' * 5) + str_to_print + (' ' * 5)
    print('\n' + '-' * len(str_to_print))
    print(str_to_print)
    print('-' * len(str_to_print) + '\n')

    # Train ⚡
    train(config)


def loop_all_runs(config):
    train_dataset_sizes = [64, 128, 1024, 1e4, 5e4, 1e5, 2e5, 3e5, 4e5, 5e5, 1e6, 1e7, 5e7]
    MIN_COMPLEXITY = 0
    MAX_COMPLEXITY = 5

    for i, train_ds_size in enumerate(train_dataset_sizes):
        for complexity in range(MIN_COMPLEXITY, MAX_COMPLEXITY + 1):
            single_run(config, train_ds_size, complexity)


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--size", type=float, default=None, help="train dataset size", required=True)
    parser.add_argument("-m", "--complexity", type=int, default=None, help="complexity (m)", required=True)
    args = parser.parse_args()

    print('\n' + '>' * 15)
    print(f'\tnow running massive_datasets_fig12.py -size {int(args.size)} -complexity {args.complexity}')
    print('>' * 15 + '\n')

    try:
        # load common config
        with open(CONFIG_PATH, 'r') as stream:
            config = yaml.safe_load(stream)

        single_run(config, int(args.size), args.complexity)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")


if __name__ == "__main__":
    main()
