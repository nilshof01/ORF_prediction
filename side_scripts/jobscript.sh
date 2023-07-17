#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J test_cuda
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 1:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=1GB]"
### -- set the email address --


#BSUB -u jobs.nilshof@outlook.de
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o output_scripts/%J.out
#BSUB -e error_scripts/%J.err
# -- end of LSF options --

module load python3/3.6.2
module load numpy/1.13.1-python-3.6.2-openblas-0.2.20 
module load pandas/0.20.3-python-3.6.2
#module load scipy/0.19.1-python-3.6.2

python3 /zhome/20/8/175218/orf_prediction/training.py
