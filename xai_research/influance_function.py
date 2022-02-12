import os
import yaml
from argparse import ArgumentParser
from typing import Union
from pathlib import Path

import wandb
import torch
import pytorch_influence_functions as ptif

from models import MLP
from train import setup_loaders

WANDB_PATH = Path("../wandb")


def remove_model_from_keys(state_dict: dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k[6:]] = v

    return new_state_dict


def load_model(path):
    state_dict = torch.load(path)['state_dict']
    state_dict = remove_model_from_keys(state_dict)
    model = MLP()

    model.load_state_dict(state_dict)
    model.eval()

    return model


def get_config(run_dir: Union[str, Path]) -> dict:
    config_path = os.path.join(run_dir, 'files/config.yaml')
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream)

    original_config = {}
    # strip weird wandb parts
    for k, v in config.items():
        if 'wandb' in k:
            continue

        original_config[k] = v['value']

    return original_config


def download_run_from_wandb(run_id: str):
    api = wandb.Api()
    run = api.run(f"deep-learning-course-project/pointer-value-retrieval/{run_id}")

    for file in run.files():
        try:
            file.download(f'wandb/{run_id}/files')
        except Exception as e:
            pass

    return Path(f'wandb/{run_id}/files')


def get_run(run_id):
    run_path = list(WANDB_PATH.glob(f'*{run_id}*'))

    # check if run doesn't exist locally
    if len(run_path) == 0:
        run_path = download_run_from_wandb(run_id)
    else:
        run_path = run_path[0]

    return run_path


def main(run_id: str, epoch: int, use_cuda=False):
    """

    Args:
        run_id: WandB run ID
        use_cuda:
    """
    run_dir = get_run(run_id)
    config = get_config(run_dir)

    ckpt_path = list(run_dir.glob(f'files/Checkpoints/*epoch={epoch}*'))
    if len(ckpt_path) == 0:
        raise ValueError(f'No checkpoint in epoch {epoch}')
    else:
        ckpt_path = ckpt_path[0]

    model = load_model(ckpt_path)

    if use_cuda:
        model.cuda()

    config['data']['train_params']['size'] = 20
    trainloader, testloader = setup_loaders(config)

    ptif.init_logging()
    if_config = ptif.get_default_config()

    if_config['outdir'] = os.path.join('influance_results', run_id)
    if_config['gpu'] = 1 if use_cuda else -1
    if_config['dataset'] = config['data']['task']
    if_config['num_classes'] = 1
    if_config['test_sample_num'] = 1

    influences = ptif.calc_img_wise(if_config, model, trainloader, testloader)
    pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--run", type=str, default='m21q11cw',
                        help="run identifier in WandB")

    parser.add_argument("-e", "--epoch", type=int, default=75,
                        help="epoch checkpoint")

    parser.add_argument("-c", "--cuda", action='store_true',
                        help="run on GPU or not")

    args = parser.parse_args()

    main(run_id=args.run, epoch=args.epoch, use_cuda=args.cuda)
