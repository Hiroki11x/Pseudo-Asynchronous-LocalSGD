# Code for TinyStories on GPT-NEO Experiments

## Overview

This repository contains the code for the experiments on the TinyStories dataset using GPT-NEO.
As a backend, we use torch distributed data parallel for DDP training.

```sh
└── gptneo
    ├── README.md
    ├── __init__.py
    ├── main.py
    ├── util.py
    ├── algorithm
    │   ├── __init__.py
    │   ├── adaptive_localsgd.py
    │   ├── base_algorithm.py
    │   ├── ddp.py
    │   ├── experimental
    │   ├── localsgd.py
    │   ├── post_localsgd.py
    │   ├── post_localsgd_decoupled.py
    │   ├── postadaptive_localsgd.py
    │   └── postadaptive_localsgd_decoupled.py
    ├── config
    │   ├── __init__.py
    │   ├── init_config.py
    │   └── models
    ├── data
    │   ├── __init__.py
    │   └── data_module.py
    └── requirements.txt

```

## Dataset

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

## Requirements

```sh
pip install -r requirements.txt
```

## Example of a command to run a experiment

```sh
torchrun --nnodes 1 --nproc-per-node 4 main.py --precision 16-mixed --num_gpus 4 --bs_per_gpu 32 --grad_accu_step 16 --algorithm postadaptive_localsgd --clip_gradnorm True --optimizer adamw --lr 0.01 --outer_optimizer nesterov --outer_lr 0.1 --dataset_name roneneldan/TinyStories --num_dataload_workers 16 --model_config_path config/models/gptneo_8m.yaml --is_decoupled True --is_debug_mode True --sync_gradient False --sync_optim_state False --sync_interval 32 --local_sgd_start_iter 32 --prob_no_grad 0.2 --eta 1 --wandb_project YOUR_WANDB_PROJECT_NAME --wandb_entity YOUR_WANDB_ENTITY --wandb_expname YOUR_WANDB_EXPNAME --is_grad_clear False --is_intended_impl False --is_multi_outer_step False
```

## Arguments

- `--precision`: Precision of the model. Default is `16-mixed`. (Actually, it is not used in the current implementation.)
- `--num_gpus`: Number of GPUs to use. Default is `4`.
- `--bs_per_gpu`: Batch size per GPU. Default is `32`.
- `--grad_accu_step`: Number of gradient accumulation steps. Default is `16`.
- `--algorithm`: Algorithm to use.
- `--clip_gradnorm`: Whether to clip the gradient norm. Default is `True`.
- `--optimizer`: Optimizer to use. Default is `adamw`.
- `--lr`: Learning rate. Default is `0.01`.
- `--outer_optimizer`: Outer optimizer to use. Default is `nesterov`.
- `--outer_lr`: Learning rate of the outer optimizer. Default is `0.1`.
- `--dataset_name`: Name of the dataset. Default is `roneneldan/TinyStories`.
- `--num_dataload_workers`: Number of dataloader workers. Default is `16`.
- `--model_config_path`: Path to the model configuration file. Default is `config/models/gptneo_8m.yaml`.
- `--is_decoupled`: Whether to use the decoupled version of the algorithm. Default is `True`.
- `--is_debug_mode`: Whether to use the debug mode. Default is `True`. If `True`, the model will be trained on a small subset of the dataset.
- `--sync_gradient`: Whether to synchronize the gradients. Default is `False`.
- `--sync_optim_state`: Whether to synchronize the optimizer state. Default is `False`.
- `--sync_interval`: Synchronization interval. Default is `32`.
- `--local_sgd_start_iter`: Start iteration of the local SGD. Default is `32`.
- `--prob_no_grad`: Probability of not computing the gradient. Default is `0.2`.
- `--eta`: Eta parameter of the algorithm. Default is `1`.
- `--wandb_project`: Name of the Weights & Biases project.
- `--wandb_entity`: Name of the Weights & Biases entity.
- `--wandb_expname`: Name of the Weights & Biases experiment.
- `--is_grad_clear`: Whether to clear the gradient for outer model. Default is `False`.
- `--is_intended_impl`: Whether to use the intended implementation. Default is `False`.
- `--is_multi_outer_step`: Whether to use multiple outer steps. Default is `False`.