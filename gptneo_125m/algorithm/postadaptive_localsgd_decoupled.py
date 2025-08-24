"""
All rights reserved to the original author
Hiroki Naganuma
Aug 13th, 2024
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from util import get_optimizer
from algorithm.base_algorithm import BaseAlgorithm
from torch.nn.utils import clip_grad_norm_

import copy

class PostAdaptiveLocalSGDDecoupled(BaseAlgorithm):
    """
    Post Adaptive Local SGD Decoupled algorithm.
    """

    def __init__(self, config, model_config, device):
        super().__init__(config, model_config, device)
        self.optimizer = get_optimizer(self.model.parameters(), 
                                       self.config.optimizer_name, 
                                       self.config.lr
        )
        # Local SGD
        self.sync_interval = self.config.sync_interval
        self.accumulated_gradients = 0
        self.grad_accu_step = self.config.grad_accu_step

        # Post Local SGD
        self.local_sgd_start_iter = self.config.local_sgd_start_iter
        self.use_ddp = True

        # Adaptive Local SGD
        self.prob_no_grad = self.config.prob_no_grad
        self.eta = self.config.eta

        self.is_update_wo_grad = False
        self.return_loss = 0

        # Decoupled Algorithm
        self.outer_model = copy.deepcopy(self.model)
        self.outer_model_sync()
        self.outer_optimizer = get_optimizer(self.outer_model.parameters(), 
                                             self.config.outer_optimizer_name, 
                                             self.config.outer_lr
        )

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output


    def compute_averaged_outer_gradient_with_allreduce(self, outer_model, inner_model):
        """
        Compute the averaged outer gradient using all-reduce.
        """
        diff = []
        for outer_param, inner_param in zip(outer_model.parameters(), inner_model.parameters()):
            param_diff = outer_param.data - inner_param.data
            dist.all_reduce(param_diff, op=dist.ReduceOp.SUM)
            param_diff /= dist.get_world_size()
            diff.append(param_diff)
        return diff
    

    def train_step(self, batch, batch_idx):

        if self.global_step  == self.local_sgd_start_iter:
            print(f"Switch from DDP to Local SGD at global step {self.global_step}")
            self.use_ddp = False
            self.outer_model_sync()

        # DDP
        if self.use_ddp:
            output = self(**batch)
            loss = output.loss / self.grad_accu_step # Scaling the loss
            loss.backward()
            self.return_loss = loss * self.grad_accu_step

            if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

                self.average_gradients()
                self.average_optimizer_state()

                if self.config.clip_gradnorm:
                    clip_grad_norm_(self.model.parameters(), max_norm=1)

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accumulated_gradients = 0
                self.global_step += 1
            else:
                self.accumulated_gradients += 1

        # Adaptive Local SGD
        else:
            current_lrs = [pg['lr'] for pg in self.optimizer.param_groups]

            # Indivisual Steps of Adaptive Local SGD (update without gradient)
            if self.is_update_wo_grad:

                if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

                    with torch.no_grad():
                        for param, cent_param, lr in zip(
                            self.model.parameters(), 
                            self.outer_model.parameters(),
                            current_lrs):

                            param.data -= lr / self.prob_no_grad * self.eta * (param.data - cent_param.data)

                    self.accumulated_gradients = 0
                    self.is_update_wo_grad = self.get_random_bool()
                    self.global_step+=1
                else:
                    self.accumulated_gradients += 1
        
            # # Indivisual Steps of Adaptive Local SGD (update with gradient)
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

            if self.should_average_model():
                '''
                [Post Adaptive Local SGD Decoupled Algorithm]

                1. calc outer_gradient
                2. all reduce outer_gradient
                3. set outer_gradient to gradient of outer_model
                4. update outer_model by using outer_optimizer
                5. set outer_model to model
                
                '''

                # 1. calc outer_gradient
                # 2. all reduce outer_gradient
                averaged_outer_gradient = self.compute_averaged_outer_gradient_with_allreduce(self.outer_model, self.model)

                # 3. set outer_gradient to gradient of outer_model
                for param, averaged_grad in zip(self.outer_model.parameters(), averaged_outer_gradient):
                    param.grad = averaged_grad.data.clone()

                # 4. update outer_model by using outer_optimizer
                self.outer_optimizer.step()

                if self.config.is_grad_clear:
                    self.outer_optimizer.zero_grad()

                if self.config.is_multi_outer_step:
                    for i in range(self.grad_accu_step-1):
                        self.outer_optimizer.step()

                # 5. set outer_model to model
                self.model.load_state_dict(self.outer_model.state_dict())

                if self.config.sync_optim_state:
                    self.average_optimizer_state()
                
                print(f"Rank {self.config.rank} is updating outer model parameters at global step: {self.global_step}")  
                torch.cuda.synchronize() # Wait for all ranks to finish averaging

        return self.return_loss

    def outer_model_sync(self):

        # allreduce client model
        self.average_model_parameters()

        # set the client model to the central model
        self.outer_model.load_state_dict(self.model.state_dict())

    def should_average_model(self):
        if self.config.is_intended_impl:
            return (self.global_step + 1) % self.sync_interval == 0 and (self.accumulated_gradients == 0)
        else:
            return (self.global_step + 1) % self.sync_interval == 0 

    def get_random_bool(self):
        return np.random.uniform(0, 1) < self.prob_no_grad

    def adjust_learning_rate(self, optimizer, factor):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor

    def reset_learning_rate(self, optimizer, original_lrs):
        for lr, param_group in zip(original_lrs, optimizer.param_groups):
            param_group['lr'] = lr