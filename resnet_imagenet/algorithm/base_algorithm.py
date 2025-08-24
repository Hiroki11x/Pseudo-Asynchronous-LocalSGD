import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Dict
import torchvision.models as models
import os
from util import LabelSmoothing

class BaseAlgorithm(nn.Module):
    def __init__(self, config, device):
        super(BaseAlgorithm, self).__init__()

        self.device = device
        self.config = config
        self.arch = config.arch

        self.global_step = 0
        self.lr = self.config.lr
        self.model = self.initialize_model()
        if self.config.loss_name == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.config.loss_name == "label_smoothing":
            self.criterion = LabelSmoothing(smoothing=config.smoothing).to(self.device)
        else:
            raise ValueError(f"Loss function {config.loss_name} not supported.")

        if config.resume_from_checkpoint:
            self.start_epoch = config.checkpoint_epoch
        else:
            self.start_epoch = 0

    def reset_individual_update(self):
        self.optimizer.zero_grad()
        self.average_model_parameters()

    def initialize_model(self):
        self.model = models.__dict__[self.config.arch]().to(self.device)
        self.average_model_parameters()
        return self.model
    
    def get_start_epoch(self):
        return self.start_epoch

    def load_model_checkpoint(self):
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{self.start_epoch}.pth')
        if os.path.isfile(checkpoint_path):
            print(f"Loading model checkpoint from {checkpoint_path}...")
            checkpoint = torch.load(checkpoint_path)

            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            self.model.load_state_dict(new_state_dict)
            self.global_step = checkpoint.get('global_step', 0)
            print(f"Model checkpoint loaded, resuming from epoch {checkpoint['epoch']}")
            self.start_epoch = checkpoint['epoch']
            return checkpoint['epoch']
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def load_optimizer_checkpoint(self):
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{self.start_epoch}.pth')
        if os.path.isfile(checkpoint_path):
            print(f"Loading optimizer checkpoint from {checkpoint_path}...")
            
            checkpoint = torch.load(checkpoint_path)

            optimizer_state_dict = checkpoint['optimizer_state_dict']
            new_optimizer_state_dict = {}
            for k, v in optimizer_state_dict.items():
                if k.startswith('module.'):
                    new_optimizer_state_dict[k[7:]] = v 
                else:
                    new_optimizer_state_dict[k] = v

            self.optimizer.load_state_dict(new_optimizer_state_dict)

            if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                print("Loading scheduler checkpoint.")
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Optimizer checkpoint loaded.")
            self.start_epoch = checkpoint['epoch']
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def save_checkpoint(self, epoch):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'global_step': self.global_step
        }
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved at {checkpoint_path}')

    def forward(self, images):
        output = self.model(images)
        return output

    def average_model_parameters(self):
        dist.barrier()

        params = [param.data.view(-1) for param in self.model.parameters() if param.requires_grad]
        params_vector = torch.cat(params)

        params_vector = params_vector.clone()

        dist.all_reduce(params_vector, op=dist.ReduceOp.SUM)
        params_vector /= dist.get_world_size()

        pointer = 0
        for param in self.model.parameters():
            if param.requires_grad:
                num_params = param.numel()
                param.data.copy_(params_vector[pointer:pointer + num_params].view_as(param.data))
                pointer += num_params

    def average_optimizer_state(self):
        dist.barrier()
        state_tensors = []
        shapes = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                state = self.optimizer.state[p]
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        if v.device != self.device:
                            v = v.to(self.device)
                        state_tensors.append(v.view(-1))
                        shapes.append(v.shape)

        if state_tensors:
            state_vector = torch.cat(state_tensors)

            dist.all_reduce(state_vector, op=dist.ReduceOp.SUM)
            state_vector /= dist.get_world_size()

            pointer = 0
            idx = 0
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    state = self.optimizer.state[p]
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            num_elements = v.numel()
                            v.copy_(state_vector[pointer:pointer + num_elements].view(shapes[idx]))
                            pointer += num_elements
                            idx += 1

    def average_gradients(self):
        dist.barrier()
        grads = [param.grad.view(-1) for param in self.model.parameters() if param.grad is not None]
        grad_vector = torch.cat(grads)

        dist.all_reduce(grad_vector, op=dist.ReduceOp.SUM)
        grad_vector /= dist.get_world_size()

        pointer = 0
        for param in self.model.parameters():
            if param.grad is not None:
                num_params = param.grad.numel()
                param.grad.data.copy_(grad_vector[pointer:pointer + num_params].view_as(param.grad))
                pointer += num_params

    def train_step(self, image, target, batch_idx):
        raise NotImplementedError("Subclasses should implement this method to train_step.")
