from training_funcs import training
import torch
import numpy as np
import torch
from torch import nn
import torch.optim as optim
#from cnn_model_diff_act_func import TheOneAnd
from save_training_results import log_training_info
#from models.cnn_model import TheOneAndOnly
from models.small_cnn import TheOneAndOnly
import torch.nn.parallel
import torch.distributed as dist
import psutil
from one_hot_encoding import create_chuncks
import os


test_model = False
base_dir = "/work3/s220672/ORF_prediction"
batch_size = 120
channels = 4 # a network with channel 1 showed far less good results: maybe because the numbers do not equalize the nucleotides which is problematic in kernel operations
training_name = "1000frag_10000orgs_70bs"
limit = 6*700*5000
LEARNING_RATE = 0.0001 # before 0.00001
wDecay = 0.00005 # 0.005 could lead to too high regularization bc i could see that the model didnt not learn or was not flexible enough. the validation accuracies were about 10 % lower than training but a further factor to consider is that i didnt use dropout and my network was quiet big
epochs = 17 
train_optim = "ADAM"
momentum = 0.95
is_sparse = False
sequence_length = 30

train_dir = "/work3/s220672/ORF_prediction/processed/1000frag_10000orgs"
assert os.path.isdir(train_dir), "The training directory does not exist."

assert torch.cuda.is_available(), ("The system could not connect with cuda. You will continue with cpu.")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
Net = TheOneAndOnly(channels = channels,
                    test = test_model, 
                    sequence_length = sequence_length)



out = Net(torch.randn(batch_size, 4, 6, sequence_length, device="cpu"))


print("test_successful")
Net = TheOneAndOnly(channels = channels,
                    test = test_model,
                     sequence_length = sequence_length)

Net = Net.to(device)
# Check if multiple GPUs are available
#if torch.cuda.device_count() > 1:

#dist.init_process_group(backend='nccl')
print("Using", torch.cuda.device_count(), "GPUs for parallel processing.")

Net = nn.DataParallel(Net) # use multiple gpus if available

min_valid_loss = np.inf
if train_optim == "SGD":
    optimizer = optim.SGD(Net.parameters(),
                        lr = LEARNING_RATE,
                        momentum = momentum,
                        weight_decay=wDecay)
if train_optim=="ADAM":
    optimizer = optim.Adam(Net.parameters(),
                        lr=LEARNING_RATE,
                         weight_decay=wDecay,)


ram_usage = psutil.Process().memory_info().rss / 1024 ** 2  # RAM usage in MB
print(f"RAM usage before training: {ram_usage:.2f} MB")


if not test_model:
    train_loss_mean, valid_loss_mean,train_precision, valid_precision = training(Net,
                                                                                  optimizer,
                                                                                  epochs,
                                                                                  base_dir,
                                                                                  batch_size,
                                                                                  train_dir,
                                                                                  training_name)
    log_training_info(filename = "training_results",
                        training_loss = train_loss_mean,
                        training_accuracy = train_precision,
                        validation_loss = valid_loss_mean,
                        validation_accuracy = valid_precision,
                        epoch_number = epochs,
                        learning_rate_value=LEARNING_RATE,
                        weight_decay_value=wDecay,
                        batch_size_value=batch_size,
                        data_input = training_name)
    print("training finished")