import os.path
import torch
from models.small_cnn import TheOneAndOnly
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import subprocess
import sys
import argparse
import gzip
import zipfile

def get_filename_without_ext(full_path):
    file_name_with_extension = os.path.basename(full_path)
    file_name, file_extension = os.path.splitext(file_name_with_extension)
    is_zipped = False
    # If the file was a .zip file, remove the .zip extension
    if file_extension in [".gz", ".zip"]:
        is_zipped = True
        file_name, file_extension_2 = os.path.splitext(file_name)

    return file_name, is_zipped, file_extension

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
    orfs_tensor = orfs_tensor.permute(0, 3, 2, 1)  # Permute the dimensions to get to the required size
    orfs_tensor = orfs_tensor.float()

    return orfs_seq, orfs_tensor


def loop_fastq(handle, threshold_seq_length,precision_thresh, model):
    predicted_sequences = []
    discarded_sequences = []
    input_length_model = threshold_seq_length - 2
    for record in SeqIO.parse(handle, "fastq"):
        # Access the sequence and quality scores
        sequence = record.seq
        id_seq = record.id
        if len(record.seq) > threshold_seq_length:
            sequence = sequence[:threshold_seq_length]
            orfs_seq, orfs_tensor = find_orfs(sequence, input_length_model)
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
    return predicted_sequences, discarded_sequences


def prediction(fastq_file_path,precision_thresh, damage, threshold_seq_length, save_dir):
    name, is_zipped, file_extension = get_filename_without_ext(fastq_file_path)
    assert file_extension in [".gz", ".fastq", ".zip"], "Please ensure that your input file has one of the following endings: .fastq, .zip, .gz "
    damage_list = ["nodam", "middam", "highdam"]
    possible_seq_length = [32, 35, 38, 41]
    dir_path = os.path.dirname(os.path.abspath(__file__))
    assert os.path.isfile(fastq_file_path), "The give file path does not exist"
    assert isinstance(precision_thresh, float), "Please enter your precision threshold as float"
    assert damage in damage_list, "Please enter for the degree of damage nodam, middam or highdam"
    assert threshold_seq_length in possible_seq_length, "Please enter a valid threshold for the sequences. So far these are: "  + str(possible_seq_length                                                                                                                          )
    assert os.path.isdir(save_dir), "Please enter a valid directory to save the output files"
    trained_seq_length = threshold_seq_length - 2
    weights_path = os.path.join(dir_path, "model_weights", f"{damage}_{trained_seq_length}nt.pth")
    model = TheOneAndOnly(channels=4,
                          test=False)
    pretrained_weights = torch.load(weights_path,
                                    map_location=torch.device('cpu'))
    pretrained_weights = {k.replace("module.", ""): v for k, v in pretrained_weights.items()}
    model.eval()

    if file_extension == ".gz":
        with gzip.open(fastq_file_path, "rt") as handle:
            predicted_sequences, discarded_sequences = loop_fastq(handle, threshold_seq_length,precision_thresh, model)

    elif file_extension == ".zip":
        with zipfile.ZipFile(fastq_file_path, 'r') as zip_ref:
            zip_ref.extractall()

            # Assuming there's only one file in the .zip, and we know it's a .fastq
            fastq_file_name = zip_ref.namelist()[0]  # Gets the first (and only) filename in the .zip

            with open(fastq_file_name, "r") as handle:
                predicted_sequences, discarded_sequences = loop_fastq(handle, threshold_seq_length,precision_thresh, model)

    else:
        with open(fastq_file_path, "r") as handle:
            predicted_sequences, discarded_sequences = loop_fastq(handle, threshold_seq_length,precision_thres, model)

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
  #  subprocess.run(['gzip', "f", corr_orfs_path])
   # subprocess.run(['gzip', "f", disc_orfs_path])





parser = argparse.ArgumentParser(description='Predictor')

# Add arguments to the parser
parser.add_argument('fastq_file_path',type = str, help='Path to the fastq file')
parser.add_argument('--precision_thresh', type=float, default=0.01, help="The threshold for the minimum certainty of the model to accept an open reading frame. Input type is a float in range 0 < x < 1 Default is 0.01.")
parser.add_argument('--damage', default='nodam', help='Degree of damage')
parser.add_argument('--threshold_seq_length', type=int, default=32, help=' The minimum sequence length as input for the model. All reads with a lower length will filtered and saved in _discORFs.')
parser.add_argument('--save_dir', default=os.getcwd(), help='Directory to save output files')
args = parser.parse_args()

fastq_file_path = args.fastq_file_path
precision_thresh = args.precision_thresh
damage = args.damage
threshold_seq_length = args.threshold_seq_length
save_dir = args.save_dir

prediction(fastq_file_path, precision_thresh, damage, threshold_seq_length, save_dir)