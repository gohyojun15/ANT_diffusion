"""
This code calculates the affinity score between a denoising tasks.

## Note.
Since my affiliation is changed during the neurips review process of our paper, some parts of my code are lost.
Note that Note that implementation can be different from codes originally used in my original paper.
"""

import os
import argparse
import torch

torch.backends.cudnn.benchmark = True
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import numpy as np

from models.create_model import create_model
from diffusion import create_diffusion
from util.data_util import center_crop_arr
from diffusers.models import AutoencoderKL


def calculate_task_afinity_score(
    weight_folder, model_config, image_size, num_classes, num_images_for_gradient, data_path
):
    assert torch.cuda.is_available(), "calculating TAS currently requires at least one GPU."
    config = OmegaConf.load(model_config)

    latent_size = image_size // 8
    config.model.param["latent_size"] = latent_size
    config.model.param["num_classes"] = num_classes

    model = create_model(model_config=config.model)
    model.to("cuda")
    diffusion = create_diffusion(timestep_respacing="")
    model.train()
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to("cuda")
    scaling_factor = 0.18215

    def create_dataset():
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        dataset = ImageFolder(data_path, transform=transform)
        return dataset

    dataset = create_dataset()
    # shuffle dataset and select num_images_for_gradient
    indices = torch.randperm(len(dataset))[:num_images_for_gradient]
    batch_size = 100
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(indices),
        num_workers=2,
        pin_memory=True,
    )

    def vae_encode(x):
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(scaling_factor)
        return x

    # list up all the weights
    ckpt_path_list = os.listdir(weight_folder)

    tas_all = []
    for ckpt in ckpt_path_list:
        ckpt_path = os.path.join(weight_folder, ckpt)
        model.load_state_dict(torch.load(ckpt_path)["model"], strict=False)

        gradient_list = []
        for t in tqdm(range(diffusion.num_timesteps)):
            for data in tqdm(dataloader):
                x, y = data
                y = y.cuda()
                timestep = torch.zeros_like(y) + t
                x = x.cuda()
                latent = vae_encode(x)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, latent, timestep, model_kwargs)

                loss_dict["loss"].mean().backward()

            gradient_list.append(
                torch.cat(
                    [
                        p.grad.clone().detach().cpu().view(-1)
                        for p in model.parameters()
                        if p.grad is not None
                    ]
                ),
            )
            # zero model gradient
            model.zero_grad()

        tas = np.zeros((diffusion.num_timesteps, diffusion.num_timesteps))
        # calculate cosine similarity between gradients
        for i in range(diffusion.num_timesteps):
            for j in range(diffusion.num_timesteps):
                gradient_i = gradient_list[i]
                gradient_j = gradient_list[j]
                cosine_similarity = torch.nn.functional.cosine_similarity(
                    gradient_i.unsqueeze(0), gradient_j.unsqueeze(0)
                )
                tas[i, j] = cosine_similarity.item()
        tas_all.append(tas)
        np.save(f"tas_{ckpt}.npy", tas)  # for saving
    return tas_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task Affinity Score")
    parser.add_argument(
        "--weight_folder",
        type=str,
        help="Folder containing the weights of the models through iterations",
    )
    parser.add_argument("--model_config", type=str, default="config/DiT-S.yaml")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--data_path", type=str, default="/root/dataset/imagenet/train")
    parser.add_argument("--num_images_for_gradient", type=int, default=1000)
    args = parser.parse_args()
    task_afinity_score = calculate_task_afinity_score(
        args.weight_folder,
        args.model_config,
        args.image_size,
        args.num_classes,
        args.num_images_for_gradient,
        args.data_path,
    )
