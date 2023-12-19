# MineSegSAT

This repository is an implementation of [MineSegSAT](https://arxiv.org/abs/2311.01676), presented at [ECRS 2023](https://ecrs2023.sciforum.net/). MineSegSAT is a deep learning model that identifies environmentally impacted areas of mineral extraction sites using the [SegFormer](https://arxiv.org/abs/2105.15203) model architecture trained on Sentinel-2 data.

## Installation

This project uses poetry as a package manager which can be installed by following the instructions found [here](https://python-poetry.org/docs/#installation).

To install the dependencies in this project run the following command while your virtual environment is activated:

```bash
poetry install
```

Data that accompanies this repository can be downloaded from [here](https://drive.google.com/drive/folders/1FMruAwQeOB0T8BunxzBmjQI5R5uj6wAp?usp=sharing). A sample configuration file that could be used for training a model on the provided dataset can be found in the `configs` directory.
