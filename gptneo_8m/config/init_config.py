import argparse
import os

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def load_config():
    parser = argparse.ArgumentParser()
    ROOT_PATH = os.environ.get("EXP_ROOT", "./")
    
    parser.add_argument("--root_path", type=str, default=ROOT_PATH)
    parser.add_argument("--seed", type=int, default=2023)

    parser.add_argument("--is_debug_mode", type=str2bool, default=False)

    parser.add_argument("--wandb_project", type=str, default="project_name")
    parser.add_argument("--wandb_entity", type=str, default="entity")
    parser.add_argument("--wandb_expname", type=str, default="base")

    parser.add_argument(
        "--dataset_name", type=str, 
        default="roneneldan/TinyStories", 
        choices=["roneneldan/TinyStories", "togethercomputer/RedPajama-Data-1T-Sample", "bigcode/the-stack-smol-xl"]
    )
    parser.add_argument("--max_length", type=int, default=300)
    parser.add_argument("--test_size", type=float, default=0.1)

    parser.add_argument("--model_load_ckpt_pth", help="Default: none")
    parser.add_argument("--model_config_path", type=str, default="./config/models/gptneo_8m.yaml")

    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--bs_per_gpu", type=int, default=128)
    parser.add_argument("--grad_accu_step", type=int, default=16)
    parser.add_argument("--num_dataload_workers", type=int, default=16)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--precision", type=str, default="32")

    parser.add_argument("--clip_gradnorm", type=str2bool, default=False)
    parser.add_argument("--optimizer_name", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.25)
    parser.add_argument("--outer_optimizer_name", type=str, default="adam")
    parser.add_argument("--outer_lr", type=float, default=0.25)

    parser.add_argument(
        "--algorithm", type=str, default="ddp",
        choices=["ddp", "localsgd", "adaptive_localsgd", "post_localsgd", "postadaptive_localsgd"]
    )
    
    parser.add_argument("--is_decoupled", type=str2bool, default=False)
    parser.add_argument("--sync_optim_state", type=str2bool, default=False)
    parser.add_argument("--sync_gradient", type=str2bool, default=False)
    parser.add_argument("--sync_interval", type=int, default=100)
    parser.add_argument("--local_sgd_start_iter", type=int, default=100)

    parser.add_argument("--prob_no_grad", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.5)

    parser.add_argument("--checkpoint_dirpath", type=str, default= ROOT_PATH + "/checkpoints")
    parser.add_argument(
        "--checkpoint_filename", type=str, default="ckpt-{step}"
    )

    parser.add_argument("--is_intended_impl", type=str2bool, default=True)
    parser.add_argument("--is_grad_clear", type=str2bool, default=True)
    parser.add_argument("--is_multi_outer_step", type=str2bool, default=False)

    args = parser.parse_args()
    return args