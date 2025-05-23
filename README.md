# IIA

> **Incremental Information-Aware: Mine Abundant and Accurate Information for Video Captioning**
>
> Ningkai Zhong, Bin Fang, Mengdi Li, and Langping Wang.
>
> 

## Content

- [IIA](#IIA)
  - [Content](#Content)
  - [Environment](#environment)
  - [Running](#running)
    - [Overview](#overview)
    - [Training](#training)
    - [Testing](#testing)
    - [Show Results](#show-results)
  - [Reproducibility](#reproducibility)
    - [Main Experiments](#main-experiments)
    - [Ablation Study](#ablation-study)
    - [Analysis](#analysis)
  - [Citation](#citation)
  - [Acknowledgements](#acknowledgements)

## Environment
Clone and enter the repo:

```shell
git clone https://github.com/Zhongnibug/IIA.git
cd IIA
```

Here we use:
- `Python` 3.11.8
- `torch` 2.2.1
- `cuda` 12.1

If you are using Anaconda, you can use the following command to create and activate the environment:
```shell
conda env create -f iia_env.yml
conda activate IIA
```

# Data Preparation
