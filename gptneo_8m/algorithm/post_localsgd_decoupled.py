import torch
import torch.nn as nn
import torch.distributed as dist

from util import get_optimizer
from algorithm.base_algorithm import BaseAlgorithm
from torch.nn.utils import clip_grad_norm_

import copy

class PostLocalSGDDecoupled(BaseAlgorithm):
    """
    Post Local SGD Decoupled algorithm.
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
        self.local_sgd_start_iter = self.config.local_sgd_start_iter
        self.use_ddp = True

        # Decoupled Algorithm
        self.outer_model = copy.deepcopy(self.model)
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

        if self.global_step == self.local_sgd_start_iter:
            print(f"Switch from DDP to Local SGD at global step {self.global_step}")
            self.use_ddp = False
            self.outer_model.load_state_dict(self.model.state_dict())

        output = self(**batch)
        loss = output.loss / self.grad_accu_step # Scaling the loss
        loss.backward()
        
        # Perform gradient accumulation and synchronization
        if (self.accumulated_gradients + 1) % self.grad_accu_step == 0:

            if self.use_ddp:
                if self.config.sync_gradient:
                    self.average_gradients()
                if self.config.sync_optim_state:
                    self.average_optimizer_state()
                torch.cuda.synchronize()

            if self.config.clip_gradnorm:
                clip_grad_norm_(self.model.parameters(), max_norm=1)

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

                    if self.config.sync_optim_state:
                        self.average_optimizer_state()
                    print(f"Rank {self.config.rank} is updating outer model parameters at global step: {self.global_step}")  
                    torch.cuda.synchronize() # Wait for all ranks to finish averaging


        else:
            self.accumulated_gradients += 1

        return loss * self.grad_accu_step
