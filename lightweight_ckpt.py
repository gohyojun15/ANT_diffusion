import torch

ckpt = torch.load("results/009-DiT-L-2/checkpoints/0100000.pt")

a = 3

new_dict = {
    "ema": ckpt["ema"],
    "model": ckpt["model"],
}

torch.save(new_dict, "ANT_UW_DIT_L(multi-gpu-run-result).pt")
