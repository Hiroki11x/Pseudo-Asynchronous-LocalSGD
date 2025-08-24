import torch
import torch.nn as nn
import torch.distributed as dist

from util import get_optimizer
from algorithm.base_algorithm import BaseAlgorithm
from torch.nn.utils import clip_grad_norm_

class PostLocalSGD(BaseAlgorithm):
    """
    Post Local SGD algorithm.
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

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output

    def train_step(self, batch, batch_idx):

        if self.global_step == self.local_sgd_start_iter:
            print(f"Switch from DDP to Local SGD at global step {self.global_step}")
            self.use_ddp = False

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
                    # Average model parameters on all ranks
                    self.average_model_parameters()
                    print(f"Rank {self.config.rank} is averaging model parameters at global step: {self.global_step}")       
                    torch.cuda.synchronize() # Wait for all ranks to finish averaging

        else:
            self.accumulated_gradients += 1

        return loss * self.grad_accu_step