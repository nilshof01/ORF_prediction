import os.path
import torch
from models.small_cnn import TheOneAndOnly
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess
import sys

def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}  # Mapping of nucleotides to integers
    seq_encoded = [mapping[nucleotide] for nucleotide in seq]  # Convert sequence to integers

    seq_encoded = torch.tensor(seq_encoded)  # Convert list to tensor
    seq_encoded = torch.nn.functional.one_hot(seq_encoded, num_classes=4)  # One hot encoding

    return seq_encoded

def find_orfs(sequence, chosen_length):
    orfs = []
    orfs_seq = []

    # Forward frames
    for frame in range(3):
        int_frame = sequence[frame:]
        int_frame = int_frame[:chosen_length]
        orfs_seq.append(int_frame)
        orfs.append(one_hot_encode(int_frame))

    # Reverse frames
    reverse_sequence = sequence[::-1]  # Reverse the sequence
    for frame in range(3):
        int_frame = reverse_sequence[frame:]
        int_frame = int_frame[:chosen_length]
        orfs_seq.append(int_frame)
        orfs.append(one_hot_encode(int_frame))

    orfs_tensor = torch.stack(orfs, dim=0)  # Stack the ORFs along a new dimension
    orfs_tensor = orfs_tensor.unsqueeze(0)  # Add an extra dimension at the beginning for batch size
    orfs_tensor = orfs_tensor.permute(0, 3, 1, 2)  # Permute the dimensions to get to the required size

    return orfs_seq, orfs_tensor

def prediction(fastq_file_path,precision_thresh, damage, threshold_seq_length, save_dir):
    damage_list = ["nodam", "middam", "highdam"]
    possible_seq_length = [32, 35, 38, 41]
    assert os.path.isfile(fastq_file_path), "The give file path does not exist"
    assert isinstance(precision_thresh, float), "Please enter your precision threshold as float"
    assert damage in damage_list, "Please enter for the degree of damage nodam, middam or highdam"
    assert threshold_seq_length in possible_seq_length, "Please enter a valid threshold for the sequences. So far these are: "  + str(possible_seq_length
                                                                                                                                      )
    assert os.path.isdir(save_dir), "Please enter a valid directory to save the output files"
    trained_seq_length = threshold_seq_length - 2
    weights_path = os.path.join("~", "src", "model_weights", f"{damage}_{trained_seq_length}nt.pth")
    model = TheOneAndOnly(channels=4,
                          test=False)
    pretrained_weights = torch.load(weights_path,
                                    map_location=torch.device('cpu'))
    pretrained_weights = {k.replace("module.", ""): v for k, v in pretrained_weights.items()}
    predicted_sequences = []
    discarded_sequences = []
    with open(fastq_file_path, "r") as handle:
        # Iterate over each sequence record in the file
        for record in SeqIO.parse(handle, "fastq"):
            # Access the sequence and quality scores
            if len(record.seq) >= threshold_seq_length:
                sequence = record.seq
                sequence = sequence[:threshold_seq_length]
                id_seq = record.id
                orfs_seq, orfs_tensor = find_orfs(sequence, 30)
                model_output = model(orfs_tensor)
                index_correct = model_output.argmax().item()
                max_item = model_output.max()
                if max_item > precision_thresh:
                    pred_seq = orfs_seq[index_correct]
                    predicted_record = SeqRecord(Seq(pred_seq),
                                                 id=id_seq,  # use the original sequence's ID
                                                 )
                    predicted_sequences.append(predicted_record)
                else:
                    discarded_seq = SeqRecord(sequence,
                                              id=id_seq)
                    discarded_sequences.append(discarded_seq)
            else:
                discarded_seq = SeqRecord(sequence,
                                          id=id_seq)
                discarded_sequences.append(discarded_seq)
    base_fastq = os.path.basename(fastq_file_path)
    name, ext = os.path.splitext(base_fastq)  # Split the extension
    if not save_dir:
        corr_orfs_path = name + "_corrORFS"
        disc_orfs_path = name + "_discORFS"
    else:
        corr_orfs_path = os.path.join(save_dir, name+"_corrORFS")
        disc_orfs_path = os.path.join(save_dir, name + "_discORFS")

    with open(corr_orfs_path, "w") as output_handle:
        SeqIO.write(predicted_sequences, output_handle, "fasta")

    with open(disc_orfs_path, "w") as output_handle:
        SeqIO.write(discarded_sequences, output_handle, "fasta")
    subprocess.run(['gzip', "f", corr_orfs_path])
    subprocess.run(['gzip', "f", disc_orfs_path])

fastq_file_path=sys.argv[1]
precision_thresh=float(sys.argv[2])
damage=sys.argv[3]
threshold_seq_length=int(sys.argv[4])
save_dir=sys.argv[5]
prediction(fastq_file_path, precision_thresh, damage, threshold_seq_length, save_dir)