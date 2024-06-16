#!/bin/sh
#SBATCH --qos=opp
#SBATCH --account=nova
#SBATCH --time=8:00:00
#SBATCH --job-name=bigo_model_mask2former
#SBATCH --partition=wc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=92G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maldonadoagudelorafael@gmail.com
#SBATCH --output=bigo_model_nerdprong_%x.o%A
#SBATCH --no-requeue

export HOME=/wclustre/nova/users/rafaelma2/
export https_proxy=http://squid.fnal.gov:3128
export http_proxy=http://squid.fnal.gov:3128

hostname

nvidia-smi --list-gpus

module load cuda/11.8.0

module load anaconda/2023.07-2

conda activate /work1/nova/achriste/conda/mlinstseg

which python


cd /wclustre/nova/users/rafaelma2/segmentation/src/modelado/nerdprong

python bigo.py




