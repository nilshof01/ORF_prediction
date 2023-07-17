#!/bin/bash


script="data_preparation/encode_numbers.py"

file_name="8000frag_2000orgs_33nt"
data_folder="/home/people/s220672/ReadsMatchProtein/results/6000frags_63nt"            #r"/home/projects/metagnm_asm/nils/dresults/middam"
base_dir_save=r"/home/people/s220672/ReadsMatchProtein/one_hot_encoded_tables"
train_seq_no=8000
val_seq_no=1000
test_seq_no=1000
limit_train_orgs=2000
limit_val_orgs=1000
limit_test_orgs=500
sequence_max_length=33

python $script $file_name $data_folder $base_dir_save $train_seq_no $val_seq_no $test_seq_no $limit_train_orgs $limit_val_orgs $limit_test_orgs $sequence_max_length

