import torch
import torch.distributed as dist
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.optim.lr_scheduler import StepLR
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
    set_seed,
    get_lr_warmup_cosine_scheduler,
)

def cleanup():
    dist.destroy_process_group()

def main(config):
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    rank = int(os.environ["RANK"])
    local_rank = rank % torch.cuda.device_count()
    world_size = int(os.environ["WORLD_SIZE"])

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

    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')

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
        train_dataset = datasets.FakeData(
            1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(
            50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(config.root_path, 'train')
        valdir = os.path.join(config.root_path, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

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
        pin_memory=True,
        shuffle=False,
    )

    scheduler = get_lr_warmup_cosine_scheduler(algorithm.optimizer,
                                               warmup_epochs=config.warmup_epoch,
                                               epochs_budget=config.epoch,
                                               eta_min=0)

    start_epoch = algorithm.get_start_epoch()

    print(f'Rank {rank} is starting training loop...')
    for epoch in range(start_epoch, config.epoch):
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

        algorithm.reset_individual_update()

        print(f'Rank {rank} is evaluating...')
        val_loss = 0
        val_n = 0
        val_correct = 0
        val_total = 0
        algorithm.eval()
        with torch.no_grad():
            for (images, target) in val_loader:
                images = images.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                output = algorithm(images)
                loss = algorithm.criterion(output, target)
                val_loss += loss.item()
                val_n += 1

                _, pred_label = torch.max(output.data, 1)
                val_correct += (pred_label == target).sum().item()
                val_total += target.size(0)

        val_loss /= val_n
        val_avg_acc = 100 * val_correct / val_total

        print(
            f'Epoch {epoch}: train_loss={train_loss_avg}, val_loss={val_loss}, train_acc={train_avg_acc}, val_acc={val_avg_acc}')

        if rank == 0:
            if (config.algorithm == 'postadaptive_localsgd' or config.algorithm == 'post_localsgd') and config.is_decoupled:
                current_outer_lr = algorithm.outer_optimizer.param_groups[0]['lr']
            else:
                current_outer_lr = 0
            current_lr = algorithm.optimizer.param_groups[0]['lr']

            wandb.log({
                'epoch': epoch,
                'global_step': algorithm.global_step,
                'train_loss': train_loss_avg,
                'train_acc': train_avg_acc,
                'val_loss': val_loss,
                'val_acc': val_avg_acc,
                'lr': current_lr,
                'outer_lr': current_outer_lr
            })

        if rank == 0 and config.save_checkpoint and epoch == config.checkpoint_epoch:
            algorithm.save_checkpoint(epoch)

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        scheduler.step()

        if config.algorithm == 'postadaptive_localsgd' or config.algorithm == 'post_localsgd':
            if config.local_sgd_start_iter <= algorithm.global_step:
                algorithm.outer_scheduler.step()

if __name__ == "__main__":
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    config = load_config()
    config.rank = rank
    config.world_size = world_size

    if config.num_nodes == 1:
        config.num_gpus = config.num_gpus_per_node * config.num_nodes
        config.num_gpus_per_node = config.num_gpus

    config.bs_local = config.bs_per_gpu * config.grad_accu_step
    config.bs_global = config.bs_local * config.num_gpus_per_node * config.num_nodes
    config.bs_effective = config.bs_global * config.sync_interval

    steps_per_epoch = int(1281167/config.bs_global)
    config.local_sgd_start_epoch = int(config.local_sgd_start_iter/steps_per_epoch)

    main(config)
    cleanup()
