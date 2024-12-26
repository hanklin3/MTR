#!/bin/bash
#SBATCH -n 4 --gres=gpu:volta:1 -o output/waymo/mtr+100_percent_data/0005_upweight2.0_time40_cls_loss.log-%j

source /etc/profile

module load anaconda/Python-ML-2024b
source activate mtr

NGPUS=1
MODEL_NAME=model_test
# MODEL_NAME=0002_upweight2.0_time40
# MODEL_NAME=0003_time0-40
# MODEL_NAME=0004_upweight2.0_time40_60
MODEL_NAME=0005_upweight2.0_time40_cls_loss
cd tools && bash scripts/dist_train.sh $NGPUS --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 5 --epochs 30 --extra_tag $MODEL_NAME