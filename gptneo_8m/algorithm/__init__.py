from algorithm.ddp import DDP
from algorithm.localsgd import LocalSGD
from algorithm.post_localsgd import PostLocalSGD
from algorithm.adaptive_localsgd import AdaptiveLocalSGD
from algorithm.postadaptive_localsgd import PostAdaptiveLocalSGD
from algorithm.post_localsgd_decoupled import PostLocalSGDDecoupled
from algorithm.postadaptive_localsgd_decoupled import PostAdaptiveLocalSGDDecoupled

def get_algorithm(config, model_config, device):
    if config.algorithm == 'ddp': 
        return DDP(config, model_config, device)
    
    elif config.algorithm == 'localsgd':
        return LocalSGD(config, model_config, device)
    
    elif config.algorithm == 'post_localsgd':
        if config.is_decoupled:
            return PostLocalSGDDecoupled(config, model_config, device)
        else:
            return PostLocalSGD(config, model_config, device)

    elif config.algorithm == 'adaptive_localsgd':
        return AdaptiveLocalSGD(config, model_config, device)
    
    elif config.algorithm == 'postadaptive_localsgd':
        if config.is_decoupled:
            return PostAdaptiveLocalSGDDecoupled(config, model_config, device)
        else:
            return PostAdaptiveLocalSGD(config, model_config, device)

    else:
        raise ValueError(f'Algorithm {config.algorithm} not supported.')