import numpy as np



def decode_sequence(block, channel):

    nucleotide_map = {
            'A': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'T': [0, 0, 0, 1]
        }

    reverse_map = {tuple(v): k for k, v in nucleotide_map.items()}
    decoded_seqs = []
    image_data = block.numpy()
  #  image_data = np.swapaxes(image_data,0, 2)
    for seq_idx in range(image_data.shape[0]):
        seq = ''
        for base_idx in range(image_data.shape[1]):
            pixel = tuple(image_data[seq_idx, base_idx])
            base = reverse_map[pixel]
            seq += base
        decoded_seqs.append(seq)
    return decoded_seqs
