#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpua100
### -- set the job Name --
#BSUB -J nodam_30nt
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 20:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=55GB]"
### -- set the email address --


#BSUB -u jobs.nilshof@outlook.de
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o job_output/nodam_30nt.out
#BSUB -e job_output/nodam_30nt.err
# -- end of LSF options --

#module load python3/3.6.2
#module load numpy/1.13.1-python-3.6.2-openblas-0.2.20 
#module load pandas/0.20.3-python-3.6.2
#module load scipy/0.19.1-python-3.6.2

Loaded module: binutils/2.34
Loaded module: gcc/8.4.0
Loaded module: python3/3.8.2
Loaded module: openblas/0.3.9
Loaded module: cuda/11.8
Loaded module: cudnn/v8.6.0.163-prod-cuda-11.X
Loaded module: ffmpeg/4.2.2
Loaded module: numpy/1.18.2-python-3.8.2-openblas-0.3.9
Loaded module: pandas/1.0.3-python-3.8.2
Loaded module: scipy/1.4.1-python-3.8.2

python3 /work3/s220672/ORF_prediction/training_model/training.py
