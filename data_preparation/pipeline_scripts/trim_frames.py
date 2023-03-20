#!/usr/bin/env python
# coding: utf-8


import pandas as pd
from Bio import SeqIO
from scipy.stats import truncnorm
import sys
import os.path

fastq_file = str(sys.argv[0])
lower = int(sys.argv[1])
upper = int(sys.argv[2])
id = str(sys.argv[3])

sequences = pd.DataFrame(columns = ["sequence", "result", "id"])
n = 0
number_id =0
with open(fastq_file, "r") as handle:
    # Use SeqIO.parse() to read the file as FASTQ records
    for record in SeqIO.parse(handle, "fastq"):
        # Access the sequence and quality scores of each record
        seq = str(record.seq)
        bin_ = record.id[-1]

        data = {"sequence": seq, "result": bin_, "id":str(number_id)}
        data = pd.DataFrame([data])
        sequences = pd.concat([sequences, data])
        # Process the data as needed

        n =n + 1

        if n == 6:
            n = 0
            number_id = number_id + 1

# define function to slice sequences
def trim_sequences(group):
    min_len = min(len(seq) for seq in group['sequence'])
    group['Trimmed_Sequence'] = group['sequence'].apply(lambda x: x[:min_len])
    return group

result = sequences.groupby('id').apply(trim_sequences)

def get_truncated_normal_random_value(lower, upper, mean, std_dev):
    distribution = truncnorm((lower - mean) / std_dev, (upper - mean) / std_dev, loc=mean, scale=std_dev)
    return int(distribution.rvs(1))

def trim_sequences_randomly(group, lower, upper, mean, std_dev):
    random_value = get_truncated_normal_random_value(lower, upper, mean, std_dev)
    min_len = min(random_value, min(len(seq) for seq in group['sequence']))
    group['Trimmed_Sequence'] = group['sequence'].apply(lambda x: x[:min_len])
    return group

mean = (upper + lower) / 2
std_dev = (upper - lower) / 4

result = result.groupby('id').apply(lambda x: trim_sequences_randomly(x, lower, upper, mean, std_dev))
result = result[["result", "Trimmed_Sequence", "id"]]
result.to_csv(id + "_trimmed.csv", header = None)