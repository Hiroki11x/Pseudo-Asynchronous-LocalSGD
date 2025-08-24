
import pynvml
import os

import torch
import numpy as np
import PIL
import sys

import yaml
import random

def print_library_versions():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

def print_hparams(hparams):
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

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
    print(f'\tBS_local: {config.bs_local} = {config.bs_per_gpu} x {config.grad_accu_step}')
    print(f'\tBS_global: {config.bs_local} x {config.num_gpus}')
    print(f'\tBS_effective (LocalSGD): {config.bs_effective} = {config.bs_global} x {config.sync_interval}')
    print()

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
            used_memory = proc.usedGpuMemory / 1024**2

            try:
                process_name = pynvml.nvmlSystemGetProcessName(pid)
            except pynvml.NVMLError as error:
                process_name = "Unknown"
            
            print(f"  PID {pid} ({process_name}): {used_memory:.2f} MB")
    
    print()
    pynvml.nvmlShutdown()

def print_model_paramsize(model):
    print('Model Detail:')
    print(f"\tModel: {model.__class__.__name__}")
    print(f"\tNumber of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"\tNumber of parameters (Million): {sum(p.numel() for p in model.parameters() if p.requires_grad) / 10**6:.2f}")
    print(f"\tSize of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / 1024**3:.2f} GB")
    print()

def get_optimizer(params, optimizer_name, lr, weight_decay=0.0):
    opt = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'sgd_momentum': torch.optim.SGD,
        'nesterov': torch.optim.SGD,
    }

    print(f"weight_decay: {weight_decay}")

    if optimizer_name in ['adamw', 'adam']:
        opt_kwargs = {'lr': lr, 'betas': (0.9, 0.999), 'weight_decay': weight_decay}
    elif optimizer_name == 'sgd':
        opt_kwargs = {'lr': lr, 'weight_decay': weight_decay}
    elif optimizer_name == 'sgd_momentum':
        opt_kwargs = {'lr': lr, 'momentum': 0.9, 'weight_decay': weight_decay}
    elif optimizer_name == 'nesterov':
        opt_kwargs = {'lr': lr, 'momentum': 0.9, 'nesterov': True, 'weight_decay': weight_decay}
    
    return opt[optimizer_name](params, **opt_kwargs)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def accuracy(output, target, topk=(1,)):
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