import random
from Bio import SeqIO
import sys

# Provide the input and output file paths
input_file = str(sys.argv[1])
output_file = str(sys.argv[2])

def cut_sequences(sequences):
    cut_sequences = {}
    for header, sequence in sequences.items():
        
        thresh = random.random()
        if thresh <= 0.03:  # 3% chance of trimming
            cut_sequence = sequence[:39]
        elif thresh > 0.03 and thresh <= 0.06:
                cut_sequence = sequence[:40]
        elif thresh > 0.06 and thresh <= 0.09:
                cut_sequence = sequence[:41]
        elif thresh > 0.09 and thresh <= 0.12:
                cut_sequence = sequence[:42]
        elif thresh > 0.12 and thresh <= 0.15:
                cut_sequence = sequence[:43]
        else:
            cut_sequence = sequence
        cut_sequences[header] = cut_sequence
    return cut_sequences


# Read the input FASTA file
sequences = SeqIO.to_dict(SeqIO.parse(input_file, 'fasta'))

# Cut the sequences with a chance of trimming
cut_sequences = cut_sequences(sequences)

# Write the cut sequences to the output FASTA file
SeqIO.write(cut_sequences.values(), output_file, 'fasta')

