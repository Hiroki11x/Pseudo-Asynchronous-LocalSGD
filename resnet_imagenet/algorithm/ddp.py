import torch
import torch.nn as nn
import torch.distributed as dist

from util import get_optimizer
from algorithm.base_algorithm import BaseAlgorithm

import contextlib

@contextlib.contextmanager
def sync_context():
    """A no-operation context manager."""
    yield

class DDP(BaseAlgorithm):
    """
    Distributed Data Parallel implementation extending the BaseAlgorithm.
    Ref: https://pytorch.org/tutorials/intermediate/ddp_tutorial.html
    """

    def __init__(self, config, device):
        super().__init__(config, device)

        # Load checkpoint if resuming from a saved state
        if self.config.resume_from_checkpoint:
            # raise NotImplementedError("Checkpoint loading not implemented for DDP yet")
            self.load_model_checkpoint()

        # For DDP
        self.model = nn.parallel.DistributedDataParallel(self.model, 
                                                         device_ids=[self.device],
                                                         output_device=self.device)
        self.average_model_parameters()
        self.optimizer = get_optimizer(self.model.parameters(), 
                                       self.config.optimizer_name, 
                                       self.config.lr)
        
        # Load checkpoint if resuming from a saved state
        if self.config.resume_from_checkpoint:
            self.load_optimizer_checkpoint()

        self.num_accum_grads = 0
        self.grad_accu_step = self.config.grad_accu_step


    def reset_individual_update(self):
        """
        Reset the individual update for each node.
        """
        super().reset_individual_update()
        self.num_accum_grads=0


    def forward(self, images):
        output = self.model(images)
        return output


    def train_step(self, image, target, batch_idx):

        self.model.train()

        # Forward pass without synchronization of gradients if not yet accumulated
        with self.model.no_sync() if (self.num_accum_grads + 1) % self.grad_accu_step != 0 else sync_context():
            output = self.model(image)
            loss = self.criterion(output, target)
            loss = loss / self.grad_accu_step # Scaling the loss
            loss.backward()

            _, pred_label = torch.max(output.data, 1)
            train_correct = (pred_label == target).sum().item()
            train_total = target.size(0)

        # Gradient accumulation
        if (self.num_accum_grads + 1) % self.grad_accu_step == 0:
            # [All-Reduce] Synchronize gradients across all nodes by DistributedDataParallel

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.num_accum_grads = 0
            self.global_step+=1
        else:
            self.num_accum_grads += 1

        acc = 100 * train_correct / train_total
        loss = loss.item() * self.grad_accu_step # Rescaling the loss

        return loss, acc 
