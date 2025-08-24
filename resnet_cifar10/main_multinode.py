import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import os
import numpy as np

from config.init_config import load_config
from algorithm import get_algorithm
import wandb

from datetime import timedelta

from util import (
    print_gpu_utilization, 
    print_model_paramsize, 
    print_library_versions, 
    print_hparams, 
    print_dist_info,
    set_seed
)

def cleanup():
    dist.destroy_process_group()

def main(config):
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = rank % torch.cuda.device_count()
    print(f'Rank: {rank}')
    print(f'Local Rank: {local_rank}')
    print(f'World Size: {world_size}')

    dist.init_process_group(
        backend='nccl',
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=7200000)
    )

    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    set_seed(config.seed)

    print(f'Rank {rank} is initializing model...')
    algorithm = get_algorithm(config=config, device=device).to(device)

    config.wandb_expname = f'rank{rank}_{config.wandb_expname}'
    if rank == 0: 
        if config.wandb_offline:
            os.environ["WANDB_MODE"] = "dryrun"
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

    if config.is_debug_mode:
        train_dataset = datasets.FakeData(50000, (3, 32, 32), 10, transforms.ToTensor())
        val_dataset = datasets.FakeData(10000, (3, 32, 32), 10, transforms.ToTensor())
    else:
        if config.arch == 'vit_b_16':
            transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                    (0.229, 0.224, 0.225)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                    (0.2023, 0.1994, 0.2010)),
            ])

        if rank == 0:
            train_dataset = datasets.CIFAR10(
                root='./data',
                train=True,
                download=True,
                transform=transform_train
            )
            val_dataset = datasets.CIFAR10(
                root='./data',
                train=False,
                download=True,
                transform=transform_test
            )
        dist.barrier()
        if rank != 0:
            train_dataset = datasets.CIFAR10(
                root='./data',
                train=True,
                download=False,
                transform=transform_train
            )
            val_dataset = datasets.CIFAR10(
                root='./data',
                train=False,
                download=False,
                transform=transform_test
            )

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
        num_workers=4
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.bs_per_gpu,
        shuffle=False,
        num_workers=4
    )

    scheduler = CosineAnnealingLR(algorithm.optimizer, T_max=config.epoch, eta_min=0)

    print(f'Rank {rank} is starting training loop...')
    for epoch in range(config.epoch):
        print(f'Rank {rank} is training epoch {epoch}...')
        torch.cuda.synchronize() 
        
        train_sampler.set_epoch(epoch)
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        train_n = 0
        algorithm.train()
        for batch_idx, (images, target) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss, acc = algorithm.train_step(images, target, batch_idx)
            train_loss_sum += loss
            train_acc_sum += acc
            train_n += 1

        train_loss_avg = train_loss_sum / train_n
        train_avg_acc = train_acc_sum / train_n

        print(f'Rank {rank} is evaluating...')
        val_loss_sum = 0.0
        val_n = 0
        val_correct = 0
        val_total = 0
        algorithm.eval()
        with torch.no_grad():
            for images, target in val_loader:
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = algorithm(images)
                loss = algorithm.criterion(output, target)
                val_loss_sum += loss.item()
                val_n += 1

                _, pred_label = torch.max(output.data, 1)
                val_correct += (pred_label == target).sum().item()
                val_total += target.size(0)

        val_loss_avg = val_loss_sum / val_n
        val_avg_acc = 100 * val_correct / val_total

        print(f'Epoch {epoch}: train_loss={train_loss_avg}, val_loss={val_loss_avg}, train_acc={train_avg_acc}, val_acc={val_avg_acc}')

        if rank == 0:
            wandb.log({
                'epoch': epoch,
                'global_step': algorithm.global_step,
                'train_loss': train_loss_avg,
                'train_acc': train_avg_acc,
                'val_loss': val_loss_avg,
                'val_acc': val_avg_acc,
            })

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        scheduler.step()

if __name__ == "__main__":
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    cluster = os.environ["CLUSTER_NAME"]

    config = load_config()
    config.rank = rank
    config.world_size = world_size
    config.cluster = cluster

    if config.num_nodes == 1:
        config.num_gpus = config.num_gpus_per_node * config.num_nodes
        config.num_gpus_per_node = config.num_gpus

    config.bs_local = config.bs_per_gpu * config.grad_accu_step
    config.bs_global = config.bs_local * config.num_gpus_per_node * config.num_nodes
    config.bs_effective = config.bs_global * config.sync_interval
    
    main(config)
    cleanup()
