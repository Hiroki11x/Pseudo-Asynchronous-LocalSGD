
import pynvml
import os

import numpy as np
import PIL
import sys

import yaml
import random

import torch
import torch.nn as nn


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


# Library Versions
def print_library_versions():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


# Hyperparameters
def print_hparams(hparams):
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))


# Distributed Training Information
def print_dist_info(config):

    print('Distributed Training Configuration:')
    print(f'\tWorld size: {config.world_size}')
    print(f'\tRank: {config.rank}')
    print(f'\tNumber of GPUs: {config.num_gpus}')
    print('Hyperparam Related to Batch Size:')
    print(f'\tGradient Accumulation Step: {config.grad_accu_step}')
    print(f'\tSynchronization Interval (LocalSGD): {config.sync_interval}')
    print('Batch Size:')
    print(f'\tBS per GPU: {config.bs_per_gpu}')
    print(
        f'\tBS_local: {config.bs_local} = {config.bs_per_gpu} x {config.grad_accu_step}')
    print(
        f'\tBS_global: {config.bs_global} = {config.bs_local} x {config.num_gpus}')
    print(
        f'\tBS_effective (LocalSGD): {config.bs_effective} = {config.bs_global} x {config.sync_interval}')
    print()


# GPU Utilization
def print_gpu_utilization():
    pynvml.nvmlInit()

    device_count = pynvml.nvmlDeviceGetCount()
    for device_id in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)

        print(f"Device {device_id}:")
        print(f"\tTotal memory: {info.total / 1024**2} MB")
        print(f"\tFree memory: {info.free / 1024**2} MB")
        print(f"\tUsed memory: {info.used / 1024**2} MB")
        print()

        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for proc in processes:
            pid = proc.pid
            used_memory = proc.usedGpuMemory / 1024**2  # Convert bytes to MB

            # Fetch process name
            try:
                process_name = pynvml.nvmlSystemGetProcessName(pid)
            except pynvml.NVMLError as error:
                # If there's an error, set a default process name
                process_name = "Unknown"

            print(f"  PID {pid} ({process_name}): {used_memory:.2f} MB")

    print()
    pynvml.nvmlShutdown()


# Model Parameter Size
def print_model_paramsize(model):
    print('Model Detail:')
    print(f"\tModel: {model.__class__.__name__}")
    print(
        f"\tNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(
        f"\tNumber of parameters (Million): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6:.2f}")
    print(
        f"\tSize of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / 1024**3:.2f} GB")
    print()


# Get Optimizer
def get_optimizer(params, optimizer_name, lr):
    """
    Get the optimizer object based on the optimizer name.
    """

    opt = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'sgd_momentum': torch.optim.SGD,
        'nesterov': torch.optim.SGD,
    }

    # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/resnet50v1.5/README.md
    if optimizer_name == 'adamw' or optimizer_name == 'adam':
        opt_kwargs = {'lr': lr, 'betas': (
            0.9, 0.999), 'weight_decay':  3.0517578125e-05}
    elif optimizer_name == 'sgd':
        opt_kwargs = {'lr': lr, 'weight_decay':  3.0517578125e-05}
    elif optimizer_name == 'sgd_momentum':
        opt_kwargs = {'lr': lr, 'momentum': 0.875,
                      'weight_decay':  3.0517578125e-05}
    elif optimizer_name == 'nesterov':
        opt_kwargs = {'lr': lr, 'momentum': 0.875,
                      'nesterov': True, 'weight_decay':  3.0517578125e-05}

    return opt[optimizer_name](params, **opt_kwargs)


# Set Seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# https://github.com/pytorch/examples/blob/26de41904319c7094afc53a3ee809de47112d387/imagenet/main.py#L496C1-L510C19
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# step scheduler wirh warmup
def get_lr_warmup_step_scheduler(optimizer, outer_lr_decay_factor,
                                 outer_lr_step_interval_epoch=20,
                                 outer_lr_warmup_epochs=20):
    """
    Get the learning rate scheduler object based on the scheduler name.
    """

    from torch.optim.lr_scheduler import LambdaLR, SequentialLR, StepLR

    def warmup_schedule(epoch):
        if epoch < outer_lr_warmup_epochs:
            return float(epoch) / float(max(1, outer_lr_warmup_epochs))
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    step_scheduler = StepLR(optimizer,
                            step_size=outer_lr_step_interval_epoch,
                            gamma=outer_lr_decay_factor)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, step_scheduler],
                             milestones=[outer_lr_warmup_epochs])

    return scheduler

# cosine scheduler with warmup
def get_lr_warmup_cosine_scheduler(optimizer,
                                   warmup_epochs=3,
                                   epochs_budget=90,
                                   eta_min=0):
    """
    Get the learning rate scheduler object based on the scheduler name.
    """

    from torch.optim.lr_scheduler import LambdaLR, SequentialLR, CosineAnnealingLR

    def warmup_schedule(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        return 1.0

    warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_schedule)
    cosine_scheduler = CosineAnnealingLR(optimizer,
                                         T_max=epochs_budget - warmup_epochs,
                                         eta_min=eta_min)
    scheduler = SequentialLR(optimizer,
                             schedulers=[warmup_scheduler, cosine_scheduler],
                             milestones=[warmup_epochs])
    return scheduler
