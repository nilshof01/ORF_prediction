#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J chunker6000_1000o_35
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=45GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 45GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 7:00 
### -- set the email address --


#BSUB -u jobs.nilshof@outlook.de
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o job_output/chunker6000_1000o_35.out 
#BSUB -e job_output/chunker6000_5000o_35.err 




module load python3/3.8.1
module swap openblas/0.3.7
module swap ffmpeg/4.2.2   
module load numpy/1.18.1-python-3.8.1-openblas-0.3.7 
module load pandas/1.0.3-python-3.8.1

# here follow the commands you want to execute with input.in as the input file
python3 /work3/s220672/ORF_prediction/chunck_script.py
