#!/bin/sh
#SBATCH --qos=opp
#SBATCH --account=nova
#SBATCH --time=8:00:00
#SBATCH --job-name=bigo_model_mask2former
#SBATCH --partition=wc_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:2
#SBATCH --cpus-per-gpu=20
#SBATCH --mem-per-gpu=92G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=maldonadoagudelorafael@gmail.com
#SBATCH --output=train_model_m2f_%x.o%A
#SBATCH --no-requeue



hostname

nvidia-smi --list-gpus

module load cuda/11.8.0

source /wclustre/nova/users/rafaelma2/venv385/bin/activate

which python

export HOME=/wclustre/nova/users/rafaelma2/
export HUGGING_FACE_HUB_TOKEN=/wclustre/nova/users/rafaelma2/.cache/huggingface/token
export XDG_DATA_HOME=/wclustre/nova/users/rafaelma2/.local
export XDG_CACHE_HOME=/wclustre/nova/users/rafaelma2/.cache

export https_proxy=http://squid.fnal.gov:3128
export http_proxy=http://squid.fnal.gov:3128

cd /wclustre/nova/users/rafaelma2/segmentation/src/modelado/mask2former

python bigo.py


