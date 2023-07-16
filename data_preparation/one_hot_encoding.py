import numpy as np
import torch
from torch.utils.data import Dataset, random_split
import pandas as pd





    
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

