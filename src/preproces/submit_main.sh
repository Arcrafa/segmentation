#!/bin/sh
#SBATCH --qos=opp
#SBATCH --account=nova
#SBATCH --job-name=filter_dataset
#SBATCH --gres=gpu:1
#SBATCH --partition gpu_gce
#SBATCH --nodelist=wcgpu03
#SBATCH --time=8:00:00




source /wclustre/nova/users/rafaelma/python385/bin/activate

which python

export HOME=/wclustre/nova/users/rafaelma/
export HUGGING_FACE_HUB_TOKEN=/wclustre/nova/users/rafaelma/.cache/huggingface/token
export XDG_DATA_HOME=/wclustre/nova/users/rafaelma/.local
export XDG_CACHE_HOME=/wclustre/nova/users/rafaelma/.cache

cd /wclustre/nova/users/rafaelma/preproces

python filter_ds.py
