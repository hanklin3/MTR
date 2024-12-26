#!/bin/bash
#SBATCH -n 4 -o output/waymo/mtr+100_percent_data/process_data_testing.log-%j

source /etc/profile

module load anaconda/Python-ML-2024b
source activate mtr

python mtr/datasets/waymo/data_preprocess.py ./data/waymo/scenario/  ./data/waymo