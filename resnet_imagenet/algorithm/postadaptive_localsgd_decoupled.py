"""
All rights reserved to the original author
Hiroki Naganuma
Aug 13th, 2024
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np

from util import get_optimizer, get_lr_warmup_step_scheduler, get_lr_warmup_cosine_scheduler
from algorithm.base_algorithm import BaseAlgorithm
from torch.optim.lr_scheduler import StepLR
import copy

class PostAdaptiveLocalSGDDecoupled(BaseAlgorithm):
    """
    Post Adaptive Local SGD Decoupled algorithm.
    """

    def __init__(self, config, device):
        super().__init__(config, device)

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
        self.return_acc = 0

        if self.config.resume_from_checkpoint:
            # raise NotImplementedError("Checkpoint loading not implemented for DDP yet")
            self.load_model_checkpoint()
            if self.global_step > self.local_sgd_start_iter:
                print(f"Switch from DDP to Local SGD at global step {self.global_step} is larger than local_sgd_start_iter {self.local_sgd_start_iter}")
                self.use_ddp = False

        self.optimizer = get_optimizer(self.model.parameters(), 
                                       self.config.optimizer_name, 
                                       self.config.lr
        )
        # Load checkpoint if resuming from a saved state
        if self.config.resume_from_checkpoint:
            self.load_optimizer_checkpoint()

        # Decoupled Algorithm
        self.outer_model = copy.deepcopy(self.model)
        if self.config.resume_from_checkpoint:
            if self.global_step > self.local_sgd_start_iter:
                self.outer_model.load_state_dict(self.model.state_dict())
                
        self.outer_optimizer = get_optimizer(self.outer_model.parameters(), 
                                             self.config.outer_optimizer_name, 
                                             self.config.outer_lr
        )
        # self.outer_scheduler =get_lr_warmup_cosine_scheduler(self.outer_optimizer,
        #                                        warmup_epochs=config.outer_lr_warmup_epochs,
        #                                        epochs_budget=config.epoch-config.local_sgd_start_epoch,
        #                                        eta_min=0)

        self.outer_scheduler =get_lr_warmup_cosine_scheduler(self.outer_optimizer,
                                               warmup_epochs=config.outer_lr_warmup_epochs,
                                               epochs_budget=config.epoch-config.local_sgd_start_epoch,
                                               eta_min=0.1)

        self.steps_in_epoch = 0


    def reset_individual_update(self):
        """
        Reset the individual update for each node.
        """
        super().reset_individual_update()
        self.num_accum_grads=0
        self.outer_optimizer.zero_grad()
        self.outer_model.load_state_dict(self.model.state_dict())
        self.steps_in_epoch = 0


    def forward(self, images):
        output = self.model(images)
        return output


    def compute_averaged_outer_gradient_with_allreduce(self, outer_model, inner_model):
        """
        Compute the averaged outer gradient using all-reduce.
        """
        outer_params_vector = torch.nn.utils.parameters_to_vector(outer_model.parameters())
        inner_params_vector = torch.nn.utils.parameters_to_vector(inner_model.parameters())

        param_diff = outer_params_vector - inner_params_vector

        dist.all_reduce(param_diff, op=dist.ReduceOp.SUM)
        param_diff /= dist.get_world_size()

        pointer = 0
        for param in outer_model.parameters():
            num_params = param.numel()
            param.grad = param_diff[pointer:pointer + num_params].view_as(param).clone()
            pointer += num_params

    def train_step(self, image, target, batch_idx):

        if self.global_step  == self.local_sgd_start_iter:
            print(f"Switch from DDP to Local SGD at global step {self.global_step}")
            self.use_ddp = False
            self.outer_model.load_state_dict(self.model.state_dict())

        # DDP
        if self.use_ddp:
            
            output = self.model(image)
            loss = self.criterion(output, target)
            loss = loss / self.grad_accu_step # Scaling the loss
            loss.backward()
            self.return_loss = (loss * self.grad_accu_step).item()  # Ensure scalar

            # Accuracy
            _, pred_label = torch.max(output.data, 1)
            train_correct = (pred_label == target).sum().item()
            train_total = target.size(0)
            self.return_acc = 100 * train_correct / train_total

            if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

                self.average_gradients()
                self.average_optimizer_state()
                torch.cuda.synchronize() 
                
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.accumulated_gradients = 0
                self.global_step += 1
                self.steps_in_epoch += 1
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
                    self.steps_in_epoch += 1
                else:
                    # TODO improve / waste of time / skip train this batch
                    self.accumulated_gradients += 1
        
            # # Indivisual Steps of Adaptive Local SGD (update with gradient)
            else:

                output = self.model(image)
                loss = self.criterion(output, target)
                loss = loss / self.grad_accu_step # Scaling the loss
                loss.backward()
                self.return_loss = (loss * self.grad_accu_step).item()  # Ensure scalar
                
                # Accuracy
                _, pred_label = torch.max(output.data, 1)
                train_correct = (pred_label == target).sum().item()
                train_total = target.size(0)
                self.return_acc = 100 * train_correct / train_total

                if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

                    self.adjust_learning_rate(self.optimizer, 1 / (1 - self.prob_no_grad))

                    self.optimizer.step()
                    self.adjust_learning_rate(self.optimizer, (1 - self.prob_no_grad))
                    # self.reset_learning_rate(self.optimizer, current_lrs)

                    self.optimizer.zero_grad()
                    self.global_step+=1
                    self.steps_in_epoch += 1
                    self.is_update_wo_grad = self.get_random_bool()
                    self.accumulated_gradients = 0
                else:
                    self.accumulated_gradients += 1

            # Perform local SGD averaging
            
            if (self.steps_in_epoch + 1) % self.sync_interval == 0 and self.accumulated_gradients == 0:

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
                # 3. set outer_gradient to gradient of outer_model
                self.compute_averaged_outer_gradient_with_allreduce(self.outer_model, self.model)

                # 4. update outer_model by using outer_optimizer
                self.outer_optimizer.step()
                self.outer_optimizer.zero_grad()

                # 5. set outer_model to model
                self.model.load_state_dict(self.outer_model.state_dict())

                torch.cuda.synchronize() # Wait for all ranks to finish averaging
                torch.cuda.empty_cache()

        return self.return_loss, self.return_acc


    # [Hiroki Added / Aug 13th, 2024] 
    def outer_model_sync(self):

        # allreduce client model
        self.average_model_parameters()

        # set the client model to the central model
        self.outer_model.load_state_dict(self.model.state_dict())

    def get_random_bool(self):
        return np.random.uniform(0, 1) < self.prob_no_grad

    def adjust_learning_rate(self, optimizer, factor):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= factor

    def reset_learning_rate(self, optimizer, original_lrs):
        for lr, param_group in zip(original_lrs, optimizer.param_groups):
            param_group['lr'] = lr

    def average_optimizer_state(self):
        super().average_optimizer_state()

    def average_gradients(self):
        super().average_gradients() 