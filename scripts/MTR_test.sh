#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1 -o output/waymo/mtr+100_percent_data/0004_upweight2.0_time40_60_validation.log-%j

source /etc/profile

module load anaconda/Python-ML-2024b
# source activate mtr3
# source activate catk
source activate mtr

# PYTHON=/home/gridsan/thlin/.conda/envs/mtr3/bin/python

NGPUS=1

CKPT=../output/waymo/mtr+100_percent_data/model1/ckpt/best_model.pth
MODEL_NAME=model1

# CKPT=../output/waymo/mtr+100_percent_data/0002_upweight2.0_time40/ckpt/best_model.pth
# MODEL_NAME=0002_upweight2.0_time40

# CKPT=../output/waymo/mtr+100_percent_data/0003_time0-40/ckpt/best_model.pth
# MODEL_NAME=0003_time0-40

CKPT=../output/waymo/mtr+100_percent_data/0004_upweight2.0_time40_60/ckpt/best_model.pth
MODEL_NAME=0004_upweight2.0_time40_60

CKPT=../output/waymo/mtr+100_percent_data/0005_upweight2.0_time40_cls_loss/ckpt/best_model.pth
MODEL_NAME=0005_upweight2.0_time40_cls_loss
cd tools && bash scripts/dist_test.sh $NGPUS --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --ckpt $CKPT --extra_tag $MODEL_NAME --batch_size 80 