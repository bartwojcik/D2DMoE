#!/bin/bash
source user.env
eval "$(conda shell.bash hook)"
conda activate effbench_env
python -m scripts.$1