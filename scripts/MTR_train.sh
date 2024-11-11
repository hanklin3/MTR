#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1 -o output/waymo/mtr+100_percent_data/model1.log-%j

source /etc/profile

module load anaconda/Python-ML-2024b
source activate mtr3

NGPUS=1
cd tools && bash scripts/dist_train.sh $NGPUS --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 5 --epochs 30 --extra_tag model1