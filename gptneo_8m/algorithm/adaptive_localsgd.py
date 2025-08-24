"""
All rights reserved to the original author
Hiroki Naganuma
July 1st, 2024
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from util import get_optimizer
from algorithm.base_algorithm import BaseAlgorithm
from torch.nn.utils import clip_grad_norm_

class AdaptiveLocalSGD(BaseAlgorithm):
    """
    Adaptive Local SGD algorithm.
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

        self.prob_no_grad = self.config.prob_no_grad
        self.eta = self.config.eta
        self.cent_model =self.model
        self.update_cent_model()

        self.is_update_wo_grad = False
        self.return_loss = 0

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output

    def train_step(self, batch, batch_idx):

        current_lrs = [pg['lr'] for pg in self.optimizer.param_groups]

        # update without gradient
        if self.is_update_wo_grad:
            if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:
                with torch.no_grad():
                    for param, cent_param, lr in zip(
                        self.model.parameters(), 
                        self.cent_model.parameters(),
                        current_lrs):
                        param -= lr / self.prob_no_grad * self.eta * (param - cent_param)

                self.accumulated_gradients = 0
                self.is_update_wo_grad = self.get_random_bool()
                self.global_step+=1
            else:
                self.accumulated_gradients += 1
    
        # Indivisual Steps of Local SGD
        else:
            output = self.model(**batch)
            loss = output.loss / self.grad_accu_step
            loss.backward()
            self.return_loss = loss * self.grad_accu_step
            
            if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

                self.adjust_learning_rate(self.optimizer, 1 / (1 - self.prob_no_grad))

                if self.config.clip_gradnorm:
                    clip_grad_norm_(self.model.parameters(), max_norm=1)
                    
                self.optimizer.step()
                self.reset_learning_rate(self.optimizer, current_lrs)

                self.optimizer.zero_grad()
                self.global_step+=1
                self.is_update_wo_grad = self.get_random_bool()
                self.accumulated_gradients = 0
            else:
                self.accumulated_gradients += 1

        # Perform local SGD averaging
        if (self.global_step + 1) % self.sync_interval == 0 and (self.accumulated_gradients == 0):
            self.average_model_parameters()
            self.update_cent_model()

            if self.config.sync_optim_state:
                self.average_optimizer_state()
            torch.cuda.synchronize() # Wait for all ranks to finish averaging

        return self.return_loss

    def get_random_bool(self):
        return np.random.uniform(0, 1) < self.prob_no_grad
    
    def update_cent_model(self):
        self.cent_model.load_state_dict(self.model.state_dict())

    def adjust_learning_rate(self, optimizer, factor):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor

    def reset_learning_rate(self, optimizer, original_lrs):
        for lr, param_group in zip(original_lrs, optimizer.param_groups):
            param_group['lr'] = lr