from Bio import SeqIO
import csv
from Bio.Seq import Seq
import sys

fasta_file = str(sys.argv[1])
csv_file = str(sys.argv[2])

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)

    for record in SeqIO.parse(fasta_file, "fasta"):
        sequence = str(record.seq)
        header = record.description

        # Append original sequence
        writer.writerow([sequence, header])

        # Append shifted sequences
        for i in range(1, 3):
            shifted_sequence = sequence[i:]
            writer.writerow([shifted_sequence, header])

        # Append reverse complement sequence
        reverse_complement = str(Seq(sequence).reverse_complement())
        writer.writerow([reverse_complement, header])

        # Append shifted reverse complement sequences
        for i in range(1, 3):
            shifted_reverse_complement = reverse_complement[i:]
            writer.writerow([shifted_reverse_complement, header])

