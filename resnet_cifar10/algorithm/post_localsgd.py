import torch
import torch.nn as nn
import torch.distributed as dist

from util import get_optimizer
from algorithm.base_algorithm import BaseAlgorithm

class PostLocalSGD(BaseAlgorithm):
    """
    Post Local SGD algorithm.
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

    def forward(self, images):
        output = self.model(images)
        return output

    def train_step(self, image, target, batch_idx):

        self.model.train()

        if self.global_step == self.local_sgd_start_iter:
            print(f"Switch from DDP to Local SGD at global step {self.global_step}")
            self.use_ddp = False

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
                    # Average model parameters on all ranks
                    self.average_model_parameters()
                    print(f"Rank {self.config.rank} is averaging model parameters at global step: {self.global_step}")       
                    torch.cuda.synchronize() # Wait for all ranks to finish averaging

        else:
            self.accumulated_gradients += 1

        acc = 100 * train_correct / train_total
        loss = loss.item() * self.grad_accu_step # Rescaling the loss

        return loss, acc 