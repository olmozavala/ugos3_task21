import argparse
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device
import os

# Only for jvelasco (torch has some problems to compile models)
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# Force unbuffered output so progress is visible in log files and SLURM
# .out/.err in real time, without needing `python -u`.
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def main(config):
    logger = config.get_logger("train")

    # Data loaders
    data_loader       = config.init_obj("data_loader", module_data)
    valid_data_loader = config.init_obj("data_loader", module_data, training=False)

    # Model
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = torch.compile(model)

    # Loss / metrics / optimiser / scheduler
    criterion = module_loss.build_loss(config["loss"])
    if isinstance(criterion, torch.nn.Module):
        criterion = criterion.to(device)

    metrics          = [getattr(module_metric, met) for met in config["metrics"]]
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer        = config.init_obj("optimizer",    torch.optim,              trainable_params)
    lr_scheduler     = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    torch.set_float32_matmul_precision("medium")

    trainer = Trainer(
        model, criterion, metrics, optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()

    # Return the weights directory so callers (shell scripts, notebooks, etc.)
    # can pass it directly to test.py without touching config.yml.
    return 0


if __name__ == "__main__":
    import sys

    # Fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    # All diagnostic prints go to stderr so that stdout carries only the
    # weights directory path, making $() capture safe in shell/SLURM scripts.
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"), file=sys.stderr)
    print("CUDA devices available:", torch.cuda.device_count(), file=sys.stderr)
    for i in range(torch.cuda.device_count()):
        print(f"  Device {i}: {torch.cuda.get_device_name(i)}", file=sys.stderr, flush=True)

    args = argparse.ArgumentParser(description="UNet SSH predictor – training")
    args.add_argument("-c", "--config", default="config.yml", type=str,
                      help="Path to config.yml (default: config.yml)")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="Path to a checkpoint to resume from (default: None)")
    args.add_argument("-d", "--device", default=None, type=str,
                      help="Comma-separated CUDA device indices (default: all visible)")
    args.add_argument("-i", "--uid", default=None, type=str,
                      help=(
                          "Unique run identifier prefix, e.g. $(date +%%m%%d_%%H%%M%%S). "
                          "Config-derived params are appended automatically. "
                          "Pass the same value to test.py -i to locate these weights."
                      ))

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(["--bs", "--batch_size"],    type=int,   target="data_loader;args;batch_size"),
    ]

    config = ConfigParser.from_args(args, options)

    main(config)

    # Print the save directory as the very last line of stdout so it can be
    # captured cleanly by a shell script or SLURM step:
    #
    #   WEIGHTS_DIR=$(python train.py -c config.yml 2>train.log)
    #   python test.py -c config.yml -wd "$WEIGHTS_DIR"
    #
    sys.exit(0)