# Pointer-Value-Retrieval

An *unofficial* PyTorch implementation of the 2021 paper [Pointer Value Retrieval: A new benchmark for understanding the limits of neural network generalization](https://arxiv.org/abs/2107.12580).

Both the datasets and the models are pure pytorch modules. 

For training we used Pytorch-Lightning. 

In order to train a model run the following command in the terminal
`python train.py -c <path_to_config_file> -s <seed value>`

Example configs can be found in the `configs` folder.

To recreate figures from the paper run the appropriate bash script:
`massive_datasets_fig12.sh`
`massive_datasets_fig13_m1.sh`
`massive_datasets_fig13_m2.sh`

The code requires integration with WandB, but it can be easily edited out.
