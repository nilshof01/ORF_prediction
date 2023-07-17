#!/usr/bin/env python
# coding: utf-8

'''
Hello!
This script takes two files as input:
1. The csv ("output_df.csv") which contains the correct reading-frame-assignments of a set of reads (It is the output of my "Reads Match Protein" script)
2. The reads (.fq) which are filtered the coding reagions (given the bam file in the "Reads Match Protein" script).
Given the correct reading frame the nucleotide sequences are adapted accordingly.
The resulting fastq file contains the reads in the correct reading frame.
-> This means, that the correct reading frame is always ORF 1 in downstream analysis
'''

import sys
import os
import Bio
from Bio import SeqIO
from Bio import Seq
import re
import pandas as pd
import time
import numpy as np
import copy


orfs = sys.argv[1] 
reads =  sys.argv[2]
out_dir =  sys.argv[3]
uniqid = sys.argv[4]

out_name = out_dir + "/" + "reads_corr_orf_" + str(uniqid) + ".fq"

orfs_df = pd.read_csv(orfs, sep="\t", header=None)
#orfs_df.head()

orfs_uniq_df = orfs_df.drop_duplicates(keep=False)

header_orf_ls = orfs_uniq_df[3].values.tolist()
header_ls = orfs_uniq_df[0].values.tolist()

# Read in fastq file as dictionary for fast access 
record_dict = SeqIO.index(reads, "fastq")

'''
This will adapt the reads to be in the correct reading frame. Length as original simulated which is 100bp in this case.
Do not run again if "reads_corr_orf_uniq.fq" already exists.
'''

with open(out_name, "a") as external_file:
    #record_dict.letter_annotations = {}
    for item in range(len(header_orf_ls)):
        
        # header of the read
        header = header_ls[item]
        
        # orf id number
        orf = header_orf_ls[item][-1]

        orf_int = int(orf) - 1
        og_record = record_dict[header]
        copy_fw = copy.deepcopy(og_record)
        copy_rev = copy.deepcopy(og_record)
        copy_rev.seq = copy_rev.seq.reverse_complement()
        
        for i in range(6):
            if i < 3:
                record = copy_fw[i:]
                if i == orf_int:
                    record.id = og_record.id + ":" + str(i) + ":1"
                    record.description = og_record.description + ":" + str(i) + ":1"
                else:
                    record.id = og_record.id + ":" + str(i) + ":0"
                    record.description = og_record.description + ":" + str(i) + ":0"
                if "unknown" in record.id:
                    assert False
                print(record.format("fastq"), file = external_file)
        
            else:
                #orf_rev = orf_int - 3
                record = record_dict[header]
                record_trim = copy_rev[(i-3):]
                
                if i == orf_int:
                    record_trim.id = og_record.id + ":" + str(i) + ":1"
                    record_trim.description = og_record.description + ":" + str(i) + ":1"
                else:
                    record_trim.id = og_record.id + ":" + str(i) + ":0"   
                    record_trim.description = og_record.description + ":" + str(i) + ":0"
                if "unknown" in record_trim.id:
                    assert False

                #with open("reads_corr_orf.fq", "a") as external_file:
                print(record_trim.format("fastq"), file = external_file)

external_file.close()


# remove all empty lines
with open(out_name, 'r+') as file:
    # Read the contents of the file into a list of lines
    lines = file.readlines()

    # Filter out the empty lines
    lines = filter(lambda x: x.strip(), lines)

    # Move the file pointer to the beginning of the file
    file.seek(0)

    # Overwrite the file with the non-empty lines
    file.writelines(lines)

    # Truncate the remaining content in the file (if any)
    file.truncate()
