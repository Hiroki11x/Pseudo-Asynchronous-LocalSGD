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

class PostAdaptiveLocalSGD(BaseAlgorithm):
    """
    Post Adaptive Local SGD algorithm.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self.optimizer = get_optimizer(self.model.parameters(), 
                                       self.config.optimizer_name, 
                                       self.config.lr,
                                       self.config.wd)
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
        self.cent_model =self.model
        self.update_cent_model()
        self.is_update_wo_grad = False
        self.return_loss = 0
        self.return_acc = 0

    def forward(self, images):
        output = self.model(images)
        return output

    def train_step(self, image, target, batch_idx):

        if self.global_step  == self.local_sgd_start_iter:
            print(f"Switch from DDP to Local SGD at global step {self.global_step}")
            self.use_ddp = False

        # DDP
        if self.use_ddp:

            output = self.model(image)
            loss = self.criterion(output, target)
            loss = loss / self.grad_accu_step # Scaling the loss
            loss.backward()
            self.return_loss = loss * self.grad_accu_step

            # Accuracy
            _, pred_label = torch.max(output.data, 1)
            train_correct = (pred_label == target).sum().item()
            train_total = target.size(0)
            self.return_acc = 100 * train_correct / train_total

            if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

                self.average_gradients()
                self.average_optimizer_state()

                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accumulated_gradients = 0
                self.global_step += 1
            else:
                self.accumulated_gradients += 1

        # Adaptive Local SGD
        else:
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

                output = self.model(image)
                loss = self.criterion(output, target)
                loss = loss / self.grad_accu_step # Scaling the loss
                loss.backward()
                self.return_loss = loss * self.grad_accu_step
                
                # Accuracy
                _, pred_label = torch.max(output.data, 1)
                train_correct = (pred_label == target).sum().item()
                train_total = target.size(0)
                self.return_acc = 100 * train_correct / train_total

                if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

                    self.adjust_learning_rate(self.optimizer, 1 / (1 - self.prob_no_grad))

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
                torch.cuda.synchronize() # Wait for all ranks to finish averaging

        return self.return_loss, self.return_acc

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