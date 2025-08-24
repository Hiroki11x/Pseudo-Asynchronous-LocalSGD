import argparse
import os
import torchvision.models as models

def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def load_config():
    parser = argparse.ArgumentParser()
    
    ROOT_PATH = os.environ.get("EXP_ROOT", "./")
    
    parser.add_argument("--is_debug_mode", type=str2bool, default=False, help="use fake data to benchmark")

    parser.add_argument("--root_path", type=str, default=ROOT_PATH)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vgg16',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

    parser.add_argument("--wandb_project", type=str, default="")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_expname", type=str, default="")
    parser.add_argument('--wandb_offline', type=str2bool, default=False)

    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--bs_per_gpu", type=int, default=128)
    parser.add_argument("--grad_accu_step", type=int, default=1)
    parser.add_argument("--num_dataload_workers", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--num_gpus_per_node", type=int, default=4)
    parser.add_argument("--num_nodes", type=int, default=1)

    parser.add_argument("--optimizer_name", type=str, default="adam")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--outer_optimizer_name", type=str, default="adam")
    parser.add_argument("--outer_lr", type=float, default=0.25)
    parser.add_argument("--outer_lr_decay_factor", type=float, default=1)
    parser.add_argument("--outer_lr_step_interval_epoch", type=int, default=20)
    parser.add_argument("--outer_lr_warmup_epochs", type=int, default=20)
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay (default: 1e-4)')

    parser.add_argument(
        "--algorithm", type=str, default="ddp",
        choices=["ddp", "localsgd", "adaptive_localsgd", "post_localsgd", "postadaptive_localsgd"]
    )
    
    parser.add_argument("--is_decoupled", type=str2bool, default=False)
    parser.add_argument("--sync_interval", type=int, default=100)
    parser.add_argument("--local_sgd_start_iter", type=int, default=100)

    parser.add_argument("--prob_no_grad", type=float, default=0.2)
    parser.add_argument("--eta", type=float, default=0.5)

    parser.add_argument("--checkpoint_dirpath", type=str, default= ROOT_PATH + "/checkpoints")
    parser.add_argument(
        "--checkpoint_filename", type=str, default="ckpt-{step}"
    )

    parser.add_argument("--is_grad_clear", type=str2bool, default=True)
    parser.add_argument("--is_intended_impl", type=str2bool, default=True)
    parser.add_argument("--is_multi_outer_step", type=str2bool, default=False)

    args = parser.parse_args()
    return args