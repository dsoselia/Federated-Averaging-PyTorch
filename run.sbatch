#!/bin/bash
#SBATCH --qos=high
#SBATCH --partition=dpart
#SBATCH --account=furongh
#SBATCH --gres=gpu:3
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dsoselia@umd.edu

module load cuda/11.3.1   
source ~/miniconda3/bin/activate
conda activate swin 
cd /nfshomes/dsoselia/Federated-Averaging-PyTorch
python main.py