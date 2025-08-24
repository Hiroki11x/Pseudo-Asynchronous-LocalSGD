import torch
import torch.nn as nn
import torch.distributed as dist

from util import get_optimizer
from algorithm.base_algorithm import BaseAlgorithm
from torch.nn.utils import clip_grad_norm_

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

    def __init__(self, config, model_config, device):
        super().__init__(config, model_config, device)

        # For DDP
        self.model = nn.parallel.DistributedDataParallel(self.model, 
                                                         device_ids=[self.device],
                                                         output_device=self.device)
        self.average_model_parameters()
        self.optimizer = get_optimizer(self.model.parameters(), 
                                       self.config.optimizer_name, 
                                       self.config.lr)
        
        self.num_accum_grads = 0
        self.grad_accu_step = self.config.grad_accu_step

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output

    def train_step(self, batch, batch_idx):

        self.model.train()

        # Forward pass without synchronization of gradients if not yet accumulated
        with self.model.no_sync() if (self.num_accum_grads + 1) % self.grad_accu_step != 0 else sync_context():
            output = self.model(**batch)  
            loss = output.loss / self.grad_accu_step # Scaling the loss
            loss.backward()

        # Gradient accumulation
        if (self.num_accum_grads + 1) % self.grad_accu_step == 0:
            # [All-Reduce] Synchronize gradients across all nodes by DistributedDataParallel
            if self.config.clip_gradnorm:
                clip_grad_norm_(self.model.parameters(), max_norm=1)
                
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.num_accum_grads = 0
            self.global_step+=1
        else:
            self.num_accum_grads += 1

        return loss.item() * self.grad_accu_step # Rescaling the loss
