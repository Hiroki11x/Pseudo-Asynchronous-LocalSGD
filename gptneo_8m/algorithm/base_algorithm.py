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
        for param in self.model.parameters():
            if param.requires_grad:
                with torch.no_grad():
                    tensor = param.data.clone()
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    tensor /= dist.get_world_size()
                    param.data.copy_(tensor)

    def average_optimizer_state(self):
        dist.barrier()
       
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    state = self.optimizer.state[p]
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            tensor = v.clone().to(self.device)
                            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                            tensor.div_(dist.get_world_size())
                            with torch.no_grad():
                                v.copy_(tensor)

    def average_gradients(self):
        dist.barrier()
        for param in self.model.parameters():
            if param.grad is not None:
                with torch.no_grad():
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= dist.get_world_size()

    def train_step(self, batch, batch_idx):
        raise NotImplementedError("Subclasses should implement this method to train_step.")
