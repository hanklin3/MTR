module load anaconda/Python-ML-2024b
source activate mtr
python3 -m pip install easydict
python setup.py develop --install-dir ./

module load anaconda/Python-ML-2024b
conda create --name mtr python=3.10.14 -y
python -m pip install waymo-open-dataset-tf-2-11-0==1.6.1