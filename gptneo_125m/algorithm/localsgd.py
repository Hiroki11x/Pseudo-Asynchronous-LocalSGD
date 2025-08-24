"""
All rights reserved to the original author
Hiroki Naganuma
July 1st, 2024
"""

import torch
import torch.nn as nn
import torch.distributed as dist

from util import get_optimizer
from algorithm.base_algorithm import BaseAlgorithm
from torch.nn.utils import clip_grad_norm_

class LocalSGD(BaseAlgorithm):
    """
    Local SGD algorithm.
    """

    def __init__(self, config, model_config, device):
        super().__init__(config, model_config, device)
        self.optimizer = get_optimizer(self.model.parameters(), 
                                       self.config.optimizer_name, 
                                       self.config.lr
        )
        self.sync_interval = self.config.sync_interval
        self.accumulated_gradients = 0
        self.grad_accu_step = self.config.grad_accu_step

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output

    def train_step(self, batch, batch_idx):
        
        # Indivisual Steps of Local SGD
        output = self.model(**batch)
        loss = output.loss / self.grad_accu_step
        loss.backward()
        
        if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

            if self.config.clip_gradnorm:
                clip_grad_norm_(self.model.parameters(), max_norm=1)
            
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulated_gradients = 0
            self.global_step+=1

            # Perform local SGD averaging
            if (self.global_step) % self.sync_interval == 0:

                # Average model parameters on all ranks
                self.average_model_parameters()
                torch.cuda.synchronize() # Wait for all ranks to finish averaging

        else:
            self.accumulated_gradients += 1

        return loss * self.grad_accu_step
