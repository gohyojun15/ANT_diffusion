# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from mtl_library.uw import UncertaintyWeighting
from util import dist_util
from util.data_util import create_dataloader
from util.util import create_logger
from util.uw_util import initialize_cluster, sample_t_batch
import torch.distributed as dist

torch.backends.cudnn.benchmark = True

from models.create_model import create_model
from torch.nn.parallel import DistributedDataParallel as DDP

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse

import os
from omegaconf import OmegaConf

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
@torch.compile()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    with torch.no_grad():
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def save_checkpoint(model, ema, opt, args, config, checkpoint_dir, train_steps, logger, uw):
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
        "config": config,
        "uw": uw.state_dict(),
    }
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dist_util.setup_dist(args)
    config = OmegaConf.load(args.model_config)
    device = dist_util.device()
    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."

    # Setup an experiment folder:
    if dist.get_rank() == 0:
        os.makedirs(
            args.results_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = config.model.name.replace(
            "/", "-"
        )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    config.model.param["latent_size"] = latent_size
    config.model.param["num_classes"] = args.num_classes
    model = create_model(model_config=config.model)
    # Note that parameter initialization is done within the DiT constructor

    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[dist_util.device()], bucket_cap_mb=300)

    diffusion = create_diffusion(
        timestep_respacing=""
    )  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device)
    scaling_factor = 0.18215
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    uw = DDP(
        UncertaintyWeighting(num_task=args.total_clusters).to(device),
        device_ids=[dist_util.device()],
    )
    opt = torch.optim.AdamW(
        [
            {"params": model.parameters()},
            {"params": uw.parameters(), "lr": 0.025, "weight_decay": 0.0},
        ],
        lr=config.optim.lr,
        weight_decay=config.optim.weight_decay,
    )

    # Setup data:
    loader, sampler = create_dataloader(args, logger)

    # Prepare models for training:
    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Resume checkpoint
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        opt.load_state_dict(ckpt["opt"])
        model.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        uw.module.load_state_dict(ckpt["uw"])

    args.resume = 0
    epoch = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    clusters = initialize_cluster(
        grouping_method=args.grouping_method,
        total_clusters=args.total_clusters,
        num_timesteps=diffusion.num_timesteps,
    )

    @torch.compile()
    def vae_encode(x):
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(scaling_factor)
        return x

    from torch.backends.cuda import enable_flash_sdp

    enable_flash_sdp(True)

    model = torch.compile(model)
    logger.info(f"Training for {args.iterations} iterations...")
    scaler = torch.cuda.amp.GradScaler()
    for train_steps in tqdm(range(args.iterations), dynamic_ncols=True):
        try:
            x, y = next(batch_iterator)
        except:
            sampler.set_epoch(epoch)
            batch_iterator = iter(loader)
            logger.info(f"Beginning epoch {epoch}...")
            epoch += 1
            x, y = next(batch_iterator)

        x = x.to(device)
        y = y.to(device)
        with torch.cuda.amp.autocast():
            x = vae_encode(x)
            t = sample_t_batch(x.shape[0], clusters, device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = 0
            for task_ind in range(args.total_clusters):
                ind_left, ind_right = int(t.shape[0] / args.total_clusters * task_ind), int(
                    t.shape[0] / args.total_clusters * (task_ind + 1)
                )
                l = loss_dict["loss"][ind_left:ind_right].mean() / args.total_clusters
                uw_loss = uw(l, task_ind)
                loss += uw_loss

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        # update_ema(ema, model)
        update_ema(ema, model.module)

        # Log loss values:
        running_loss += loss_dict["loss"].mean().item()
        log_steps += 1
        train_steps += 1
        if train_steps % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)

            logger.info(
                f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
            )
            logger.info(f"uw_weight: {uw.module.get_loss_weight()}")
            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()
        # Save Checkpoint
        if train_steps % args.ckpt_every == 0:  # and train_steps > 0:
            if dist.get_rank() == 0:
                save_checkpoint(
                    model.module,
                    ema,
                    opt,
                    args,
                    config,
                    checkpoint_dir,
                    train_steps,
                    logger,
                    uw.module,
                )
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model_config", type=str, default="config/DiT-L.yaml")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=400000)
    parser.add_argument("--global-batch-size", type=int, default=240)
    parser.add_argument("--global-seed", type=int, default=0)

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--gpu_offset", type=int, default=0)  # It choice starting gpu ids
    parser.add_argument("--total_clusters", type=int, default=5)
    parser.add_argument("--grouping_method", choices=["uniform"], default="uniform")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    main(args)
