# CS5834 Final Project

## Authors

Yongzhe Xu ([yongzhe@vt.edu](mailto:yongzhe@vt.edu))
Hanwen Ju ([hanwen@vt.edu](mailto:hanwen@vt.edu))
Yuhang Jia ([yuhangjia@vt.edu](mailto:yuhangjia@vt.edu))

This repository contains the implementation for the CS5834 Final Project, which analyzes monthly foot-traffic patterns using the Dewey Data Monthly Patterns dataset. The project includes data preprocessing, feature extraction, and model execution via `main.py`.

## Dataset Setup

1. Download the dataset from:
   [https://app.deweydata.io/data/advan/monthly-patterns-foot-traffic-container/monthly-patterns-foot-traffic](https://app.deweydata.io/data/advan/monthly-patterns-foot-traffic-container/monthly-patterns-foot-traffic)

2. Divide the dataset by month and save each file in the `input/` directory using the format:

```
input/
├── 2021-01.csv.gz
├── 2021-02.csv.gz
├── ...
```

## Environment Setup

It is recommended to use a virtual environment such as conda.

Example (conda):

```bash
conda create -n cs5834proj python=3.9
conda activate cs5834proj
```

## Installation

From the project root:

```bash
python3 -m pip install -r requirements.txt
```

## Running the Project

```bash
python3 main.py
```

All generated figures and output files will be saved in the `output/` directory.
