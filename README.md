# [Neurips 2023] Addressing Negative Transfer in Diffusion Models.

<!-- Arxiv Link, Project Link -->
<a href="https://arxiv.org/abs/2306.00354"><img src="https://img.shields.io/badge/arXiv-2306.00354-b31b1b.svg"></a>
<a href="https://gohyojun15.github.io/ANT_diffusion/"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>

This repository contains the official PyTorch implementation of the following paper: "Addressing Negative Transfer in Diffusion Models" (Neurips 2023). To gain a better understanding of the paper, please visit our [project page](https://gohyojun15.github.io/ANT_diffusion/) and paper on [arXiv](https://arxiv.org/abs/2306.00354).


This code is to train DiT model with ANT on ImageNet dataset.
Our implementation is based on [DiT](https://github.com/facebookresearch/DiT), [LibMTL](https://github.com/median-research-group/LibMTL), [NashMTL](https://github.com/AvivNavon/nash-mtl).



## Updates
- **2023.10.07**: [DTR](https://arxiv.org/abs/2310.07138) uses ANT-UW and shows that ANT-UW outperforms [P2weight](https://arxiv.org/abs/2204.00227) and [MinSNR](https://arxiv.org/abs/2310.07138).
- **2023.11.05**: Initial release.


## Install pre-requisites
```bash
$ pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
$ pip install -r requirements.txt
```


## Training DiT with ANT.
- ### You can train DiT-S model with ANT 
    ```
    python train_ant_single_gpu.py \
    --data_path [Your Data Path] \
    --total_clusters [Your cluster size (Default: 5)] \
    --mtl_method [uw, nash, or pcgrad]
    ```

- ### ANT-UW with multiple-gpus.
    ```
    python train_uw_multi_gpu.py \
    --data_path train \
    --total_clusters 5 
    ```


## Sampling DiT trained with ANT.
```
python sample_ddp.py \
--model_config [Your config path (config/DiT-L.yaml, config/DiT-S.yaml)] \
--ckpt [Your ckpt path]
```

## Interval clustering
The example codes for interval clustering are shown in `interval_clustering.py`
