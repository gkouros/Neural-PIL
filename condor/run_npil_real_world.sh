#!/bin/bash
echo 'Neural-PIL job started'

NAME=$1
EXP=$2

source ~/miniconda3/etc/profile.d/conda.sh
conda activate neuralpil

export PATH="/usr/local/cuda-11/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11/lib64:/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
DIR=/users/visics/gkouros/projects/nerf-repos/Neural-PIL
cd $DIR

python3 train_neural_pil.py --datadir /esat/topaz/gkouros/datasets/nerf/nerd/real_world/$NAME --basedir logs/$NAME --expname $EXP --gpu $CUDA_VISIBLE_DEVICES --config configs/neural_pil/real_world.txt --spherify

conda deactivate
echo 'Neural-PIL job terminated'
