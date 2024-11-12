#!/bin/bash
eval "$(conda shell.bash hook)"
export CONDA_ALWAYS_YES="true"
if [ -f environment.yml ]; then
  conda env create -f environment.yml
else
  conda create -n effbench_env python=3.12
  conda activate effbench_env
  conda install ninja cuda=12.1 -c nvidia -c conda-forge
  #conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
  mkdir pip-build
  # comment out if installing torch via conda:
  TMPDIR=pip-build pip --no-input --no-cache-dir --log pip-build/pip.log install torch torchvision torchaudio
  TMPDIR=pip-build pip --no-input --no-cache-dir --log pip-build/pip.log install triton accelerate transformers tiktoken datasets
  TMPDIR=pip-build pip --no-input --no-cache-dir --log pip-build/pip.log install scikit-learn k-means-constrained
  TMPDIR=pip-build pip --no-input --no-cache-dir --log pip-build/pip.log install fvcore
  TMPDIR=pip-build pip --no-input --no-cache-dir --log pip-build/pip.log install submitit omegaconf tabulate
  TMPDIR=pip-build pip --no-input --no-cache-dir --log pip-build/pip.log install tensorboard wandb seaborn
  rm -rf pip-build
  conda env export | grep -v "^prefix: " >environment.yml
fi
