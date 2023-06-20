import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import glob
import pandas as pd
import random
#lower_seq_length = 18
#upper_seq_length =
import os
import glob
import re
#import scipy.sparse as sp
import subprocess
import gzip
import io
import shutil


def unzip_memory(file):
    with gzip.open(file, 'rb') as f:
        uncompressed_data = f.read()
    # Step 2: Load the uncompressed data into memory using numpy
    array_data = np.load(io.BytesIO(uncompressed_data))
    return array_data

def gunzip_file(file_path):
    subprocess.run(['gunzip', file_path])
    
def zip_file(file_path):
    subprocess.run(['gzip', file_path])
    


def safe_subsets_temp(subsets_X, subsets_Y, save_dir, mode):
    for i, (subset_X, subset_Y) in enumerate(zip(subsets_X, subsets_Y)):
        if mode == "train":
            subset_X_file = os.path.join(save_dir, f"subset_{i+1}_X.pt")
            subset_Y_file = os.path.join(save_dir, f"subset_{i+1}_Y.pt")
        else:
            subset_X_file = os.path.join(save_dir, f"subset_{i+1}_X_val.pt")
            subset_Y_file = os.path.join(save_dir, f"subset_{i+1}_Y_val.pt")

        torch.save(subset_X, subset_X_file)
        torch.save(subset_Y, subset_Y_file)
        
        zip_file(subset_X_file)
        zip_file(subset_Y_file)


def create_chuncks(train_data, results_train, val_data, val_results, channels, num_subsets = 10, sparse_matrix = False, is_zipped = True):
    par_dir = os.path.dirname(train_data)
    assert os.path.isfile(train_data), "The training data file does not exist."
    assert os.path.isfile(results_train), "The training results file does not exist."
    assert os.path.isfile(val_data), "The validation data file does not exist."
    assert os.path.isfile(val_results), "The validation results file does not exist."
    if sparse_matrix and train_data.endswith(".npy"):
        print("The training data file is not a sparse matrix.")
        return
    if sparse_matrix and val_data.endswith(".npy"):
        print("The validation data file is not a sparse matrix.")
        return
    if is_zipped and not train_data.endswith(".gz"):
        print("The training data file is not zipped.")
        return
    if is_zipped and not val_data.endswith(".gz"):
        print("The validation data file is not zipped.")
        return

    #output = os.popen("nvcc --version").read()
    #match = re.search(r"release (\d+\.\d+)", output)
    #cuda_version = match.group(1)
    #if cuda_version != "11.0":
        #   assert cuda_version != "11.0", "You need to perform module swap cuda/11.0 or find a way to change to cuda 11.0 to train on Tesla A100 PCIE 80 GB. Currently you have " + cuda_version
    if sparse_matrix != True:
        if is_zipped:
            one_hot_blocks_all = unzip_memory(train_data)
            results_all = unzip_memory(results_train)
            one_hot_blocks_all_val = unzip_memory(val_data)
            results_all_val = unzip_memory(val_results)
        else:
            one_hot_blocks_all = np.load(train_data)
            #one_hot_blocks_all = one_hot_blocks_all.transpose((0, 1, 3, 2))
            
            results_all = np.load(results_train)

            one_hot_blocks_all_val = np.load(val_data)
        # one_hot_blocks_all_val = one_hot_blocks_all_val.transpose((0, 1, 3, 2))
            
            results_all_val = np.load(val_results)
        one_hot_blocks_size = one_hot_blocks_all.nbytes / (1024 * 1024)
        print("size of training array: " + str(one_hot_blocks_size) + " Mb")
        print(one_hot_blocks_all.shape)
        print(one_hot_blocks_all_val.shape)
        print("upload successful")
        print("Dimension of training block: " + str(one_hot_blocks_all.shape))
    num_sequences = one_hot_blocks_all.shape[0]
    if one_hot_blocks_all.shape[1] != channels:
        raise ValueError("Your channels are in the wrong Tensor dimension")
    if one_hot_blocks_all.shape[2] > one_hot_blocks_all.shape[3]:
        print("Your input tensor is wrong. Your third dimension has to be the height of the tensor and your fourth dimension your length.")
        one_hot_blocks_all = one_hot_blocks_all.swapaxes(2, 3)
        one_hot_blocks_all_val = one_hot_blocks_all_val.swapaxes(2, 3)
    else:
        pass
    # a = np.expand_dims(train_image_standard_hot[0], axis = 0)
    X_train = torch.from_numpy(one_hot_blocks_all)
    Y_train = torch.from_numpy(results_all)


    subsets_X = torch.chunk(X_train, num_subsets, dim=0)
    subsets_Y = torch.chunk(Y_train, num_subsets, dim=0)


    safe_subsets_temp(subsets_X,subsets_Y, par_dir, "train")

    print("size of trainloader: " + str(Y_train.size()) +str(X_train.size()))
    X_val = torch.from_numpy(one_hot_blocks_all_val)
    Y_val = torch.from_numpy(results_all_val)
    subsets_X = torch.chunk(X_val, num_subsets, dim=0)
    subsets_Y = torch.chunk(Y_val, num_subsets, dim=0)

    safe_subsets_temp(subsets_X,subsets_Y, par_dir, "val")
    print("size of val loader "+ str(X_val.size())+str(Y_val.size()))
    # Just checking you have as many labels as inputs
    
    
    
#create_chuncks(r"/work3/s220672/processed/6000frags_5000orgs/one_hot_blocks_all_6000frags_5000o.npy.gz", r"/work3/s220672/processed/6000frags_5000orgs/results_all_6000frags_5000o.npy.gz", r"/work3/s220672/processed/6000frags_5000orgs/one_hot_blocks_all_val_6000frags_5000o.npy.gz",r"/work3/s220672/processed/6000frags_5000orgs/results_all_val_6000frags_5000o.npy.gz", channels = 4)
    
    
def create_dataset(X, Y, batch_size, mode):     
    assert X.shape[0] == Y.shape[0]
    if mode == "train":
        
        dset_train = torch.utils.data.TensorDataset(X, Y)  # merge both together in a dataaset
        set_loader = torch.utils.data.DataLoader(dset_train,
                                            batch_size=batch_size,  # choose your batch size
                                            shuffle=True)  # generally a good idea
    if mode == "val":
        dset_val = torch.utils.data.TensorDataset(X, Y)

        # remember: batch_size of 100 means that the trainloader contains 100 parts of equal size

        set_loader = torch.utils.data.DataLoader(dset_val,
                                                batch_size=batch_size,  # choose your batch size
                                                shuffle=True)  # generally a good idea
    return set_loader


def create_dataset_from_sparse(one_hot_blocks_all, results_all, one_hot_blocks_all_val, results_all_val, batch_size, channels):
    num_sequences = one_hot_blocks_all.shape[0]
    if one_hot_blocks_all.shape[1] != channels:
        raise ValueError("Your channels are in the wrong Tensor dimension")
    if one_hot_blocks_all.shape[2] > one_hot_blocks_all.shape[3]:
        print("Your input tensor is wrong. Your third dimension has to be the height of the tensor and your fourth dimension your length.")
        one_hot_blocks_all = one_hot_blocks_all.swapaxes(2, 3)
        one_hot_blocks_all_val = one_hot_blocks_all_val.swapaxes(2, 3)
    else:
        pass
    # a = np.expand_dims(train_image_standard_hot[0], axis = 0)
    X_train = one_hot_blocks_all.toarray().reshape(-1, 4, 30, 6)
    Y_train = torch.from_numpy(results_all)
    print("size of trainloader: " + str(Y_train.size()) +str(X_train.size()))
    X_val = torch.from_numpy(one_hot_blocks_all_val)
    Y_val = torch.from_numpy(results_all_val)
    print("size of val loader "+ str(X_val.size())+str(Y_val.size()))
    # Just checking you have as many labels as inputs
    assert X_train.shape[0] == Y_train.shape[0]
    dset_train = torch.utils.data.TensorDataset(X_train, Y_train)  # merge both together in a dataaset

    dset_val = torch.utils.data.TensorDataset(X_val, Y_val)
    trainloader = torch.utils.data.DataLoader(dset_train,
                                              batch_size=batch_size,  # choose your batch size
                                              shuffle=True)  # generally a good idea
    # remember: batch_size of 100 means that the trainloader contains 100 parts of equal size

    validloader = torch.utils.data.DataLoader(dset_val,
                                              batch_size=batch_size,  # choose your batch size
                                              shuffle=True)  # generally a good idea
    return trainloader, validloader

