# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from mtl_library.nash import NashMTL
from mtl_library.uw import UncertaintyWeighting
from mtl_library.pcgrad import PCgrad
from util.data_util import center_crop_arr
from util.uw_util import initialize_cluster, sample_t_batch

torch.backends.cudnn.benchmark = True

from models.create_model import create_model

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


def save_checkpoint(model, ema, opt, args, config, checkpoint_dir, train_steps, logger):
    checkpoint = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": args,
        "config": config,
    }
    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    # Device setup:
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    torch.cuda.set_device(args.gpu)
    config = OmegaConf.load(args.model_config)
    device = "cuda"

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = config.model.name.replace(
        "/", "-"
    )  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="[\033[34m%(asctime)s\033[0m] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")],
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    config.model.param["latent_size"] = latent_size
    config.model.param["num_classes"] = args.num_classes
    model = create_model(model_config=config.model)
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    model = model.to(device)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps.
    vae = AutoencoderKL.from_pretrained(f"madebyollin/sdxl-vae-fp16-fix").to(device)
    scaling_factor = vae.config.scaling_factor
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer:
    if args.mtl_method == "uw":
        uw = UncertaintyWeighting(num_task=args.total_clusters).to(device)
        # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
        opt = torch.optim.AdamW(
            [
                {"params": model.parameters()},
                {"params": uw.parameters(), "lr": 0.025, "weight_decay": 0.0},
            ],
            lr=config.optim.lr,
            weight_decay=config.optim.weight_decay,
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=config.optim.lr, weight_decay=config.optim.weight_decay
        )

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = ImageFolder(args.data_path, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size),
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    logger.info(f"dataset class: {len(dataset.classes)}")
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Prepare models for training:
    # Resume checkpoint
    args.resume = 0
    epoch = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # Setup task clusters
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

    # Optimizations.
    from torch.backends.cuda import enable_flash_sdp

    enable_flash_sdp(True)

    model = torch.compile(model)
    logger.info(f"Training for {args.iterations} iterations...")
    scaler = torch.cuda.amp.GradScaler()

    if args.mtl_method == "nash":
        nash = NashMTL(num_tasks=args.total_clusters, model_module=model, device=device, opt=opt)
    elif args.mtl_method == "pcgrad":
        pcgrad = PCgrad(args.total_clusters, model)

    for train_steps in tqdm(range(args.iterations), dynamic_ncols=True):
        try:
            x, y = next(batch_iterator)
        except:
            batch_iterator = iter(loader)
            logger.info(f"Beginning epoch {epoch}...")
            epoch += 1
            x, y = next(batch_iterator)
        x = x.to(device)
        y = y.to(device)

        if args.mtl_method == "uw":
            with torch.cuda.amp.autocast():
                # Calculate the loss.
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
            update_ema(ema, model._orig_mod)

        elif args.mtl_method == "nash":
            with torch.cuda.amp.autocast():
                x = vae_encode(x)
                t = sample_t_batch(x.shape[0], clusters, device)

                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

                losses = []
                for task_ind in range(args.total_clusters):
                    ind_left, ind_right = int(t.shape[0] / args.total_clusters * task_ind), int(
                        t.shape[0] / args.total_clusters * (task_ind + 1)
                    )
                    l = loss_dict["loss"][ind_left:ind_right].mean()
                    losses.append(l)
                final_loss = nash(losses, logger)

            opt.zero_grad()
            scaler.scale(final_loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), nash.max_norm)
            scaler.step(opt)
            scaler.update()
            update_ema(ema, model._orig_mod)

        elif args.mtl_method == "pcgrad":
            with torch.cuda.amp.autocast():
                x = vae_encode(x)
                t = sample_t_batch(x.shape[0], clusters, device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)

                task_grad_list = torch.zeros((args.total_clusters, pcgrad.grad_dim), device=device)
                for task_ind in range(args.total_clusters):
                    ind_left, ind_right = int(t.shape[0] / args.total_clusters * task_ind), int(
                        t.shape[0] / args.total_clusters * (task_ind + 1)
                    )
                    l = loss_dict["loss"][ind_left:ind_right].mean() / args.total_clusters
                    opt.zero_grad(set_to_none=True)
                    if task_ind < args.total_clusters - 1:
                        scaler.scale(l).backward(retain_graph=True)
                    else:
                        scaler.scale(l).backward()
                    grads = []
                    for param in model.parameters():
                        if param.grad is not None:
                            grads.append(param.grad.data.view(-1))
                    task_grad_list[task_ind] = torch.cat(grads, dim=0)
            opt.zero_grad()
            new_grad = pcgrad.do_pc_grad(task_grad_list)
            pcgrad._reset_grad(new_grad)
            scaler.step(opt)
            scaler.update()

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

            # Reset monitoring variables:
            running_loss = 0
            log_steps = 0
            start_time = time()
        # Save Checkpoint
        if train_steps % args.ckpt_every == 0:  # and train_steps > 0:
            save_checkpoint(
                model._orig_mod, ema, opt, args, config, checkpoint_dir, train_steps, logger
            )
    model.eval()  # important! This disables randomized embedding dropout
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results", help="path to save results")

    parser.add_argument(
        "--model_config", type=str, default="config/DiT-S.yaml", help="path to model config"
    )

    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=800000)
    parser.add_argument("--global_batch_size", type=int, default=50)
    parser.add_argument("--global_seed", type=int, default=0)

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--gpu", type=int, default=0, help="gpu id run on")
    parser.add_argument("--total_clusters", type=int, default=5)
    parser.add_argument("--grouping_method", choices=["uniform"], default="uniform")
    parser.add_argument("--mtl_method", choices=["uw", "nash", "pcgrad"])
    args = parser.parse_args()
    main(args)
