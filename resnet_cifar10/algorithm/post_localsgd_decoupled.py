import torch
import torch.nn as nn
import torch.distributed as dist

from util import get_optimizer
from algorithm.base_algorithm import BaseAlgorithm

import copy

class PostLocalSGDDecoupled(BaseAlgorithm):
    """
    Post Local SGD Decoupled algorithm.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self.optimizer = get_optimizer(self.model.parameters(), 
                                       self.config.optimizer_name, 
                                       self.config.lr,
                                       self.config.wd)
        self.sync_interval = self.config.sync_interval
        self.accumulated_gradients = 0
        self.grad_accu_step = self.config.grad_accu_step
        self.local_sgd_start_iter = self.config.local_sgd_start_iter
        self.use_ddp = True

        # Decoupled Algorithm
        self.outer_model = copy.deepcopy(self.model)
        self.outer_optimizer = get_optimizer(self.outer_model.parameters(), 
                                             self.config.outer_optimizer_name, 
                                             self.config.outer_lr
        )

    def forward(self, images):
        output = self.model(images)
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

    def train_step(self, image, target, batch_idx):

        if self.global_step == self.local_sgd_start_iter:
            print(f"Switch from DDP to Local SGD at global step {self.global_step}")
            self.use_ddp = False
            self.outer_model.load_state_dict(self.model.state_dict())

        output = self.model(image)
        loss = self.criterion(output, target)
        loss = loss / self.grad_accu_step # Scaling the loss
        loss.backward()

        _, pred_label = torch.max(output.data, 1)
        train_correct = (pred_label == target).sum().item()
        train_total = target.size(0)


        # Perform gradient accumulation and synchronization
        if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

            if self.use_ddp:
                self.average_gradients()
                self.average_optimizer_state()
                torch.cuda.synchronize()

            self.optimizer.step()
            self.optimizer.zero_grad()
            self.accumulated_gradients = 0
            self.global_step += 1

            # Local SGD after local_sgd_start_iter steps
            if not self.use_ddp:
                # Perform local SGD averaging
                if (self.global_step) % self.sync_interval == 0:

                    '''
                    [Post Local SGD Decoupled Algorithm]

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

                    # 5. set outer_model to model
                    self.model.load_state_dict(self.outer_model.state_dict())

                    print(f"Rank {self.config.rank} is updating outer model parameters at global step: {self.global_step}")  
                    torch.cuda.synchronize() # Wait for all ranks to finish averaging


        else:
            self.accumulated_gradients += 1


        acc = 100 * train_correct / train_total
        loss = loss.item() * self.grad_accu_step # Rescaling the loss

        return loss, acc 
