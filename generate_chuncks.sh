#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J 30nt_8000frag_1600orgs_
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=40GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 50GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 9:00 
### -- set the email address --


#BSUB -u jobs.nilshof@outlook.de
### -- send notification at start -- 
#BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o job_output/30nt_8000frag_1600orgs__.out 
#BSUB -e job_output/30nt_8000frag_1600orgs__.err 




module load python3/3.8.1
module swap openblas/0.3.7
module swap ffmpeg/4.2.2   
module load numpy/1.18.1-python-3.8.1-openblas-0.3.7 
module load pandas/1.0.3-python-3.8.1


train_data="/work3/s220672/ORF_prediction/processed/8000frag_1600orgs/one_hot_blocks_all_8000frag_1600orgs_30nt.npy.gz"
results_train="/work3/s220672/ORF_prediction/processed/8000frag_1600orgs/results_all_8000frag_1600orgs_30nt.npy.gz"
val_data="/work3/s220672/ORF_prediction/processed/8000frag_1600orgs/one_hot_blocks_all_val_8000frag_1600orgs_30nt.npy.gz"
val_results="/work3/s220672/ORF_prediction/processed/8000frag_1600orgs/results_all_val_8000frag_1600orgs_30nt.npy.gz"
num_subset=20


# here follow the commands you want to execute with input.in as the input file
python3 -m src.data_preparation.chunck_processing $train_data $results_train $val_data $val_results $num_subset
