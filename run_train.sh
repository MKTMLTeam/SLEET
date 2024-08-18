#!/bin/bash
export HYDRA_FULL_ERROR=1

source ~/anaconda3/etc/profile.d/conda.sh
conda activate SLEET
python3 scripts/train.py