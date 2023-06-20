#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J 3000frags_5000orgs_40bs_
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 14:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=35GB]"
### -- set the email address --


#BSUB -u jobs.nilshof@outlook.de
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o output_scripts/3000frags_5000orgs_40bs_.out
#BSUB -e error_scripts/3000frags_5000orgs_40bs_.err
# -- end of LSF options --

#module load python3/3.6.2
#module load numpy/1.13.1-python-3.6.2-openblas-0.2.20 
#module load pandas/0.20.3-python-3.6.2
#module load scipy/0.19.1-python-3.6.2

module load python3/3.8.1
module swap openblas/0.3.7
module swap cuda/11.0   
module swap cudnn/v8.0.2.39-prod-cuda-11.0  
module swap ffmpeg/4.2.2   
module load numpy/1.18.1-python-3.8.1-openblas-0.3.7 
module load pandas/1.0.3-python-3.8.1


python3 /zhome/20/8/175218/orf_prediction/training.py
