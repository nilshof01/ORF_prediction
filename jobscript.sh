#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J 1000_1000_pingpong_LR10-6
### -- ask for number of cores (default: 1) --
#BSUB -n 2
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 5:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --


#BSUB -u jobs.nilshof@outlook.de
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o output_scripts/1000_1000_pingpong_LR10-6.out
#BSUB -e error_scripts/1000_1000_pingpong_LR10-6.err
# -- end of LSF options --

python3 /zhome/20/8/175218/orf_prediction/training.py
