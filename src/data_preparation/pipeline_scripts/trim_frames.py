#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from Bio import SeqIO
import numpy as np
import sys
import os.path

fastq_file = str(sys.argv[1])
lower = int(sys.argv[2])
upper = int(sys.argv[3])
id_ = str(sys.argv[4])
save_path = str(sys.argv[5])

sequences = pd.DataFrame(columns=["sequence", "result", "id"])
n = 0
number_id = 0
lower = int(lower)
upper = int(upper)
with open(fastq_file, "r") as handle:
    # Use SeqIO.parse() to read the file as FASTQ records
    for record in SeqIO.parse(handle, "fastq"):
        # Access the sequence and quality scores of each record
        seq = str(record.seq)
        bin_ = record.id[-1]

        data = {"sequence": seq, "result": bin_, "id": str(number_id)}
        data = pd.DataFrame([data])
        sequences = pd.concat([sequences, data])
        # Process the data as needed

        n = n + 1

        if n == 6:
            n = 0
            number_id = number_id + 1


# define function to slice sequences
def trim_sequences(group):
    min_len = min(len(seq) for seq in group['sequence'])
    group['Trimmed_Sequence'] = group['sequence'].apply(lambda x: x[:min_len])
    return group


def get_truncated_normal_random_value(lower, upper, mean, std_dev):
    value = int(np.random.normal(loc=mean, scale=std_dev))
    return int(value)


def trim_sequences_randomly(group, lower, upper, mean, std_dev):
    random_value = get_truncated_normal_random_value(lower, upper, mean, std_dev)
    min_len = min(random_value, min(len(seq) for seq in group['sequence']))
    trimmed_seqs = group['sequence'].apply(lambda x: x[:min_len])
    return trimmed_seqs


mean = abs((upper + lower) / 2)
std_dev = abs((upper - lower) / 4)
trimmed_seqs_all = pd.DataFrame([])
for group_id, group in sequences.groupby('id'):
    group = trim_sequences(group)
    trimmed_seqs = trim_sequences_randomly(group, lower, upper, mean, std_dev)
    trimmed_seqs_all = pd.concat([trimmed_seqs_all, trimmed_seqs])

sequences["Trimmed_Sequence"] = trimmed_seqs_all

save_full = os.path.join(save_path, id_ + "_trimmed.csv")
print(save_full)
sequences.to_csv(save_full, header=None)


