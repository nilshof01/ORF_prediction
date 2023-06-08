#!/usr/bin/env python
# coding: utf-8

from calculate_file_sizes import calculate_size
import numpy as np
#import torch
#from torch.utils.data import Dataset, random_split
import glob
import pandas as pd
import random
import os

def encode_sequence(seq):
    nucleotide_map = {
        'A': [1, 0, 0, 0],#[1, 0, 0, 0]
        'C': [0, 1, 0, 0],#[0, 1, 0, 0]
        'G': [0, 0, 1, 0],#[0, 0, 1, 0]
        'T': [0, 0, 0, 1]#[0, 0, 0, 1]
    }
    return np.array([nucleotide_map[nucleotide] for nucleotide in seq]).flatten()

def encoding(file_name, data_folder,train_seq_no = None,val_seq_no = None,test_seq_no = None, org_limit = None, limit_val_orgs = None, limit_test_orgs = None):
    calculate_size(org_limit, train_seq_no, val_seq_no, test_seq_no)
    nt_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num_nucleotides = len(nt_dict)
    one_hot_blocks_all = None
    one_hot_blocks_all_val = None
    one_hot_blocks_all_test = None
    data_tables = glob.glob(data_folder + "/*.csv")
    if org_limit != None:
        data_tables = data_tables[:org_limit]
    num_files = len(data_tables)
    random.shuffle(data_tables)
    if not limit_test_orgs:
        test_number = int(num_files * 0.1)
    else:
        test_number = limit_test_orgs
    test_files = random.sample(data_tables, test_number)
    data_tables = [x for x in data_tables if x not in test_files]
    if not limit_val_orgs:
        num_val_files = int(num_files * 0.2)
    else:
        num_val_files = limit_val_orgs
    train_files = num_files - num_val_files
    val_files = random.sample(data_tables, num_val_files)
    data_tables = [x for x in data_tables if x not in val_files]
    train_files = data_tables
    for i in train_files:
        try:
            table = pd.read_csv(i, header=None)
            sequences = table.iloc[:, 1]
            sequences = sequences.str[:30]
            if train_seq_no != None:
                lim_sequences = train_seq_no * 6
                sequences = sequences[:lim_sequences]
            num_sequences = sequences.shape[0]
            max_sequence_length = 30
            sequences2 = pd.DataFrame(sequences)
            sequences2.columns = ["Sequences"]
            df_encoded = pd.DataFrame(list(sequences2['Sequences'].apply(encode_sequence)))
            one_hot_matrix = np.array(df_encoded)


            ## now you have to put the six sequences in blocks
            one_hot_blocks = one_hot_matrix.reshape(-1,num_nucleotides, 6, max_sequence_length)
            one_hot_blocks = one_hot_blocks.astype(np.uint8)
            result = np.array(table.iloc[:, 2])
            result = result.reshape(-1, 6)
            if train_seq_no != None:
                result = result[:train_seq_no]
            result = result.astype(np.uint8)
            del one_hot_matrix
            if one_hot_blocks_all is None:
                one_hot_blocks_all = one_hot_blocks
                results_all = result
            else:
                one_hot_blocks_all = np.concatenate((one_hot_blocks_all, one_hot_blocks), axis=0)
                results_all = np.concatenate((results_all, result), axis=0)
        except:
            pass
    for i in val_files:
        try:
            table = pd.read_csv(i, header=None)
            sequences = table.iloc[:, 1]
            sequences = sequences.str[:30]
            if val_seq_no != None:
                lim_sequences = val_seq_no * 6
                sequences = sequences[:lim_sequences]
            num_sequences = sequences.shape[0]
            max_sequence_length = 30
            sequences2 = pd.DataFrame(sequences)
            sequences2.columns = ["Sequences"]
            df_encoded = pd.DataFrame(list(sequences2['Sequences'].apply(encode_sequence)))
            one_hot_matrix = np.array(df_encoded)



            ## now you have to put the six sequences in blocks
            one_hot_blocks = one_hot_matrix.reshape(-1,num_nucleotides, 6, max_sequence_length)
            one_hot_blocks = one_hot_blocks.astype(np.uint8)
            result = np.array(table.iloc[:, 2])
            result = result.reshape(-1, 6)
            if val_seq_no != None:
                
                result = result[:val_seq_no]
            result = result.astype(np.uint8)
            del one_hot_matrix
            if one_hot_blocks_all_val is None:
                one_hot_blocks_all_val = one_hot_blocks
                results_all_val = result
            else:
                one_hot_blocks_all_val = np.concatenate((one_hot_blocks_all_val, one_hot_blocks), axis=0)
                results_all_val = np.concatenate((results_all_val, result), axis=0)
        except:
            pass
    for i in test_files:
        try:
            table = pd.read_csv(i, header=None)
            sequences = table.iloc[:, 1]
            sequences = sequences.str[:30]
            if test_seq_no != None:
                lim_sequences = test_seq_no * 6
                sequences = sequences[:lim_sequences]
            num_sequences = sequences.shape[0]
            max_sequence_length = 30
            sequences2 = pd.DataFrame(sequences)
            sequences2.columns = ["Sequences"]
            df_encoded = pd.DataFrame(list(sequences2['Sequences'].apply(encode_sequence)))
            one_hot_matrix = np.array(df_encoded)
            
            ## now you have to put the six sequences in blocks
            one_hot_blocks = one_hot_matrix.reshape(-1,num_nucleotides,  6, max_sequence_length)
            one_hot_blocks = one_hot_blocks.astype(np.uint8)
            result = np.array(table.iloc[:, 2])
            result = result.reshape(-1, 6)
            if test_seq_no != None:
                result = result[:test_seq_no]
            result = result.astype(np.uint8)
            del one_hot_matrix
            if one_hot_blocks_all_test is None:
                one_hot_blocks_all_test = one_hot_blocks
                results_all_test = result
            else:
                one_hot_blocks_all_test = np.concatenate((one_hot_blocks_all_test, one_hot_blocks), axis=0)
                results_all_test = np.concatenate((results_all_test, result), axis=0)
        except:
            pass


 #   one_hot_blocks_all = np.transpose(one_hot_blocks_all, (0, 3, 2, 1))
  #  one_hot_blocks_all_val = np.transpose(one_hot_blocks_all_val, (0, 3, 2, 1))
   # one_hot_blocks_all_test = np.transpose(one_hot_blocks_all_test, (0, 3, 2, 1))
    np.save(r"/home/people/s220672/ReadsMatchProtein/one_hot_encoded_tables/one_hot_blocks_all_" + file_name , one_hot_blocks_all)
    np.save(r"/home/people/s220672/ReadsMatchProtein/one_hot_encoded_tables/one_hot_blocks_all_val_" +file_name, one_hot_blocks_all_val)
    np.save(r"/home/people/s220672/ReadsMatchProtein/one_hot_encoded_tables/one_hot_blocks_all_test_" +file_name, one_hot_blocks_all_test)
    np.save(r"/home/people/s220672/ReadsMatchProtein/one_hot_encoded_tables/results_all_test_" +file_name, results_all_test)
    np.save(r"/home/people/s220672/ReadsMatchProtein/one_hot_encoded_tables/results_all_val_" + file_name, results_all_val)
    np.save(r"/home/people/s220672/ReadsMatchProtein/one_hot_encoded_tables/results_all_" + file_name, results_all)


encoding(file_name = "4000frags_5000o_", data_folder = r"/home/people/s220672/ReadsMatchProtein/results/6000frags",train_seq_no = 4000, val_seq_no = 500, test_seq_no = 500, org_limit = 5000, limit_val_orgs = 1000, limit_test_orgs = 300)
