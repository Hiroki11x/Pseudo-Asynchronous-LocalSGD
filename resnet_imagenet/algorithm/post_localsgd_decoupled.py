import torch
import torch.nn as nn
import torch.distributed as dist

from util import get_optimizer, get_lr_warmup_step_scheduler, get_lr_warmup_cosine_scheduler
from algorithm.base_algorithm import BaseAlgorithm
from torch.optim.lr_scheduler import StepLR
import copy


class PostLocalSGDDecoupled(BaseAlgorithm):
    """
    Post Local SGD Decoupled algorithm.
    """

    def __init__(self, config, device):
        super().__init__(config, device)
        self.sync_interval = self.config.sync_interval
        self.accumulated_gradients = 0
        self.grad_accu_step = self.config.grad_accu_step
        self.local_sgd_start_iter = self.config.local_sgd_start_iter
        self.use_ddp = True

        # Load checkpoint if resuming from a saved state
        if self.config.resume_from_checkpoint:
            # raise NotImplementedError("Checkpoint loading not implemented for DDP yet")
            self.load_model_checkpoint()
            if self.global_step > self.local_sgd_start_iter:
                print(
                    f"Switch from DDP to Local SGD at global step {self.global_step} is larger than local_sgd_start_iter {self.local_sgd_start_iter}")
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
        self.num_accum_grads = 0
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

        outer_params_vector = torch.nn.utils.parameters_to_vector(
            outer_model.parameters())
        inner_params_vector = torch.nn.utils.parameters_to_vector(
            inner_model.parameters())


        param_diff = outer_params_vector - inner_params_vector
        dist.all_reduce(param_diff, op=dist.ReduceOp.SUM)
        param_diff /= dist.get_world_size()

        pointer = 0
        for param in outer_model.parameters():
            num_params = param.numel()
            param.grad = param_diff[pointer:pointer +
                                    num_params].view_as(param).clone()
            pointer += num_params

    def train_step(self, image, target, batch_idx):

        if self.global_step == self.local_sgd_start_iter:
            print(
                f"Switch from DDP to Local SGD at global step {self.global_step}")
            self.use_ddp = False
            self.outer_model.load_state_dict(self.model.state_dict())

        output = self.model(image)
        loss = self.criterion(output, target)
        loss = loss / self.grad_accu_step  # Scaling the loss
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
            self.steps_in_epoch += 1

            # Local SGD after local_sgd_start_iter steps
            if not self.use_ddp:
                # Perform local SGD averaging
                if (self.steps_in_epoch) % self.sync_interval == 0:

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
                    # 3. set outer_gradient to gradient of outer_model
                    self.compute_averaged_outer_gradient_with_allreduce(
                        self.outer_model, self.model)

                    # 4. update outer_model by using outer_optimizer
                    self.outer_optimizer.step()

                    # 5. set outer_model to model
                    self.model.load_state_dict(self.outer_model.state_dict())
                    # print(f"Rank {self.config.rank} is updating outer model parameters at global step: {self.global_step}")
                    torch.cuda.synchronize()  # Wait for all ranks to finish averaging

        else:
            self.accumulated_gradients += 1

        acc = 100 * train_correct / train_total
        loss = loss.item() * self.grad_accu_step  # Rescaling the loss

        return loss, acc
