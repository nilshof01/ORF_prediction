#!/usr/bin/env python
# coding: utf-8

import sys
import os
import Bio
from Bio import SeqIO
from pybedtools import BedTool
import pandas as pd
import pyranges as pr


# requires pyranges, pybedtools, biopython, pandas


# Files
reads = str(sys.argv[1])
gff =str(sys.argv[2])

#reads = "/net/node07/home/projects/metagnm_asm/1_gnm_man/snake_asm/trimmed/10e7/f40/f40_nodam_2000samples.fq"
#gff = "/net/node07/home/projects/metagnm_asm/1_gnm_man/snake_sim/RMP/data/GCF_000155995.1_ASM15599v1_genomic.gff"

print("Fastq file to bed",file=sys.stdout)

# Creating dictionary with Read information from fastq headers
fq_dic = {}
columns = ["Chromosome", "Start", "End", "ID", "Score", "Strand"]

with open(reads) as handle:
    for record in SeqIO.parse(handle, "fastq"):
        header = record.id
        header_ls = header.split(":")
        if header not in fq_dic:
        	fq_dic[header] = [header_ls[0], header_ls[2], header_ls[3], header, "NaN", header_ls[1]]
 

# Creating a DataFrame from fastq dictionary
reads_df = pd.DataFrame.from_dict(fq_dic, orient = "index", columns = columns)
reads_df = reads_df.reset_index(drop = True)
reads_df.to_csv('fastq.bed', sep='\t', header=None, index=False)
#reads_check = pd.read_csv('reads.bed', sep='\t')
#reads_check

print("Done",file=sys.stdout)
print("Converting gff to bed",file=sys.stdout)

# Converting gff reference into bed file
# Reading gff as DataFrame using package pyranges
ref_gff = pr.read_gff3(gff)

# Getting rid of pyranges object and turn it into real pandas DataFrame 
ref_gff.to_csv("ref_gff.csv", sep="\t", header=True)
ref_csv = pd.read_csv("ref_gff.csv", sep="\t")

# Filtering reference annotation for CDS only
ref_bed = ref_csv.query("Feature == 'CDS'")

# Formatting DataFrame and renaming columns
ref_bed = ref_bed[["Chromosome", "Start", "End", "ID", "Score", "Strand"]]
ref_bed.rename(columns = {"chrom" : "Chromosome", "chromStart": "Start", "chromEnd": "End", "name": "ID", "strand": "Strand", "Score": "score"}, inplace=True)

# Saving as bed file
ref_bed.to_csv('ref_gff.bed', sep='\t', header=None, index=False)
#ref_check = pd.read_csv('ref.bed', sep='\t')
#ref_check

print("Done",file=sys.stdout)
print("Finding intersection of intervals using pybedtools",file=sys.stdout)

# Applying pybedtools for intersection (Only reads which are fully covered by CDS interval)
ref = BedTool("ref_gff.bed")
fastq = BedTool("fastq.bed")
ReadsMatch = fastq.intersect(ref, f=1.0)
#print(ReadsMatch)


# Safe output to file
with open("fq_interval_matches.csv", "w") as external_file:
    print(ReadsMatch, file=external_file, sep="\t")
    external_file.close()


# Get fastq reads which matched the intervals
# DataFrame with results from bedtools
colnames = ["chrom", "start", "end", "id", "score", "strand"]
matches_df = pd.read_csv("fq_interval_matches.csv", sep="\t", names = colnames)
matches_ls = matches_df["id"].to_list()
#print(len(matches_ls))

print("Generating fastq output",file=sys.stdout)

# fastq to dict for instant access
record_dict = {}

with open(reads) as handle:
    for record in SeqIO.parse(handle, "fastq"):
        if record.id not in record_dict:
            record_dict[record.id] = record

            
with open("out.fq", "a") as external_file:
    for match in matches_ls:
        print(record_dict[match].format("fastq"), file = external_file)
external_file.close()
        
