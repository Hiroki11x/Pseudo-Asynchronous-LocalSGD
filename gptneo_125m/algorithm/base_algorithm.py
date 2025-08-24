import torch
import torch.nn as nn
import torch.distributed as dist
from transformers import GPTNeoForCausalLM, GPTNeoConfig
from typing import List, Dict

class BaseAlgorithm(nn.Module):
    def __init__(self, config, model_config, device):
        super(BaseAlgorithm, self).__init__()
        self.device = device
        self.config = config
        self.model_config = model_config

        self.global_step = 0
        self.lr = self.config.lr
        self.model = self.initialize_model()

    def initialize_model(self):
        model_config = self.model_config
        gptneo_config = GPTNeoConfig(
            max_position_embeddings=model_config['max_position_embeddings'],
            hidden_size=model_config['hidden_size'],
            num_attention_heads=model_config['num_attention_heads'],
            num_hidden_layers=model_config['num_hidden_layers']
        )
        self.model = GPTNeoForCausalLM(gptneo_config).to(self.device)
        self.average_model_parameters()
        return self.model

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return output

    def average_model_parameters(self):
        dist.barrier()

        model_to_average = self.model.module if hasattr(
            self.model, "module") else self.model

        params = [p for p in model_to_average.parameters() if p.requires_grad]

        with torch.no_grad():
            flat_params = torch.cat([param.data.view(-1)
                                    for param in params], dim=0)

            dist.all_reduce(flat_params, op=dist.ReduceOp.SUM)
            flat_params /= dist.get_world_size()

            offset = 0
            for param in params:
                numel = param.data.numel()
                param.data.copy_(
                    flat_params[offset:offset+numel].view_as(param.data))
                offset += numel

    def average_optimizer_state(self):
        if not hasattr(self, 'optimizer'):
            return

        dist.barrier()

        state_tensors = []
        shapes = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p in self.optimizer.state and p.grad is not None:
                    for k, v in self.optimizer.state[p].items():
                        if isinstance(v, torch.Tensor):
                            v_on_device = v.to(self.device)
                            state_tensors.append(v_on_device.view(-1))
                            shapes.append(
                                (p, k, v.shape, v.device, v.dtype, v.numel()))

        if len(state_tensors) == 0:
            return

        with torch.no_grad():
            flat_state = torch.cat(state_tensors, dim=0)

            dist.all_reduce(flat_state, op=dist.ReduceOp.SUM)
            flat_state /= dist.get_world_size()

            offset = 0
            idx = 0
            for (p, k, orig_shape, orig_device, orig_dtype, numel) in shapes:
                chunk = flat_state[offset:offset+numel].view(orig_shape)
                offset += numel

                v_copy = chunk.to(device=orig_device, dtype=orig_dtype)
                self.optimizer.state[p][k].copy_(v_copy)
                idx += 1

    def average_gradients(self):
        dist.barrier()

        model_to_average = self.model.module if hasattr(
            self.model, "module") else self.model

        grads = []
        shapes = []
        for param in model_to_average.parameters():
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
                shapes.append(param.grad.data.shape)

        if len(grads) == 0:
            return

        with torch.no_grad():
            flat_grads = torch.cat(grads, dim=0)

            dist.all_reduce(flat_grads, op=dist.ReduceOp.SUM)
            flat_grads /= dist.get_world_size()

            offset = 0
            i = 0
            for param in model_to_average.parameters():
                if param.grad is not None:
                    numel = param.grad.data.numel()
                    param.grad.data.copy_(
                        flat_grads[offset:offset+numel].view(shapes[i]))
                    offset += numel
                    i += 1

    def train_step(self, batch, batch_idx):
        raise NotImplementedError("Subclasses should implement this method to train_step.")
