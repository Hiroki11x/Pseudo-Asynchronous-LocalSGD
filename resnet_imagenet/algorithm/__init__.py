from algorithm.ddp import DDP
from algorithm.post_localsgd_decoupled import PostLocalSGDDecoupled
from algorithm.postadaptive_localsgd_decoupled import PostAdaptiveLocalSGDDecoupled

def get_algorithm(config, device):
    if config.algorithm == 'ddp': 
        return DDP(config, device)
    
    elif config.algorithm == 'post_localsgd':
        if config.is_decoupled:
            return PostLocalSGDDecoupled(config, device)
        else:
            return NotImplementedError('PostLocalSGD is not supported in this version.')

    elif config.algorithm == 'postadaptive_localsgd':
        if config.is_decoupled:
            return PostAdaptiveLocalSGDDecoupled(config, device)
        else:
            return NotImplementedError('PostAdaptiveLocalSGD is not supported in this version.')
    else:
        raise ValueError(f'Algorithm {config.algorithm} not supported.')