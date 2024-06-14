#!/bin/sh
#SBATCH --job-name=mrcnn
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --nodelist=gpu4

# What to call this trial
tag=$1

# My output area
basedir=/data/psihas/SSlicer/
script=/home/psihas/SSlicerTest/GitSSDir/nova_basic_${tag}.py
singularity=/data/grohmc/singularity-ML-tf1.12.simg

# Check if we are starting a new training or continuing an old one
if [ -d "${basedir}/logs/${tag}" ]
then
    # Find the latest model
    dir=$(ls -d ${basedir}/logs/${tag}/nova201* | head -1)
    model=$(ls $dir/mask_rcnn_* | tail -1)
    echo "Continuing training on ${tag} with ${model}" | tee -a ${basedir}/logs/${tag}/log.log
    cmd="python3 ${script} train --dataset=${basedir}/data/ --logdir=${basedir}/logs/${tag}/ --model=${model}"
else
    echo "Beginning training on ${tag}" | tee -a ${basedir}/logs/${tag}/log.log
    mkdir ${basedir}/logs/${tag}
    cmd="python3 ${script} train --dataset=${basedir}/data/ --logdir=${basedir}/logs/${tag}/"
fi

# Tell us what gpu we are on
nvidia-smi -L | tee -a ${basedir}/logs/${tag}/log.log
singularity exec --nv -B /usr/bin:/opt $singularity $cmd | tee -a ${basedir}/logs/${tag}/log.log
