import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataset import Subset
import torch.distributed as dist
import os
import numpy as np

from config.init_config import load_config
from data.data_module import DataModule

from algorithm import get_algorithm
import wandb

from datetime import timedelta

from util import (
    print_gpu_utilization, 
    print_model_paramsize, 
    print_library_versions, 
    print_hparams, 
    print_dist_info,
    load_model_config,
    set_seed
)

def cleanup():
    dist.destroy_process_group()

def main(config, model_config):
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=7200000))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(rank)
    set_seed(config.seed)

    print(f'Rank {rank} is initializing model...')
    algorithm = get_algorithm(config=config, model_config=model_config, device=device).to(device)

    config.wandb_expname = f'rank{rank}_{config.wandb_expname}'
    if rank == 0: 
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project, 
            name=config.wandb_expname,
            config=config)
    
    print_library_versions()
    print_hparams(vars(config))
    print_gpu_utilization()
    print_model_paramsize(algorithm)
    print_dist_info(config)
    
    print(f'Rank {rank} is initializing dataloader...')
    datamodule = DataModule(config)
    datamodule.setup()

    if config.is_debug_mode:
        num_train_samples = len(datamodule.trn_dset) // 100
        num_val_samples = len(datamodule.val_dset) // 100

        indices_train = np.random.choice(len(datamodule.trn_dset), num_train_samples, replace=False).tolist()
        indices_val = np.random.choice(len(datamodule.val_dset), num_val_samples, replace=False).tolist()
        train_dataset = Subset(datamodule.trn_dset, indices_train)
        val_dataset = Subset(datamodule.val_dset, indices_val)
    else:
        train_dataset = datamodule.trn_dset
        val_dataset = datamodule.val_dset

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.bs_per_gpu,
        sampler=train_sampler,
        pin_memory=True,
        shuffle=False,
        collate_fn=datamodule.collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.bs_per_gpu,
        shuffle=False,
        collate_fn=datamodule.collate_fn
    )

    print(f'Rank {rank} is starting training loop...')
    for epoch in range(config.epoch):
        print(f'Rank {rank} is training epoch {epoch}...')
        torch.cuda.synchronize() 
        
        train_sampler.set_epoch(epoch)
        train_loss = []
        algorithm.train()
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss = algorithm.train_step(batch, batch_idx)
            train_loss.append(loss)

        print(f'Rank {rank} is evaluating...')
        val_loss = 0
        val_n = 0
        algorithm.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                output = algorithm(**val_batch)
                val_loss += output.loss.item()
                val_n += 1
        val_loss /= val_n
        train_loss = sum(train_loss) / len(train_loss)
        print(f'Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}')

        if rank == 0:
            wandb.log({
                'epoch': epoch,
                'global_step': algorithm.global_step,
                'train_loss': train_loss,
                'val_loss': val_loss})

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    cluster = os.environ["CLUSTER_NAME"]

    config = load_config()
    config.rank = rank
    config.world_size = world_size
    config.cluster = cluster

    config.bs_local = config.bs_per_gpu * config.grad_accu_step
    config.bs_global = config.bs_local * config.num_gpus 
    config.bs_effective = config.bs_global * config.sync_interval

    model_config = load_model_config(config.model_config_path)
    
    main(config, model_config)
    cleanup()