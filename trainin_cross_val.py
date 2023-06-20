from training_funcs import load_data_training, training
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


test_model = False
training_name = "6000frags_10000org_sparse_50bs_again"
base_dir = "/zhome/20/8/175218/orf_prediction"
is_sparse = True
    
if test_model:
    train_data = r"/zhome/20/8/175218/orf_prediction/processed/test/one_hot_blocks_all_1000frags_1000o_.npy"
    results_train = r"/zhome/20/8/175218/orf_prediction/processed/test/results_all_1000frags_1000o_.npy"
    val_data = r"/zhome/20/8/175218/orf_prediction/processed/test/one_hot_blocks_all_val_1000frags_1000o_.npy"
    val_results = "/zhome/20/8/175218/orf_prediction/processed/test/results_all_val_1000frags_1000o_.npy"
else:
    train_data = r"/zhome/20/8/175218/orf_prediction/processed/one_hot_blocks_all_1000frag_5000orgs.npy.gz"
    results_train = r"/zhome/20/8/175218/orf_prediction/processed/results_all_1000frag_5000orgs.npy.gz"
    val_data = r"/zhome/20/8/17521/zhome/20/88/orf_prediction/processed/one_hot_blocks_all_val_1000frag_5000orgs.npy.gz"
    val_results = r"/zhome/20/8/175218/orf_prediction/processed/results_all_val_1000frag_5000orgs.npy.gz"



assert torch.cuda.is_available(), ("The system could not connect with cuda. You will continue with cpu.")
device = 'cuda' if torch.cuda.is_available() else 'cpu'


dif_batches = [30, 50, 70, 100, 120]

for i in range(len(dif_batches)):
    batch_size = dif_batches[i]
    channels = 4 # a network with channel 1 showed far less good results: maybe because the numbers do not equalize the nucleotides which is problematic in kernel operations
    if not test_model:
        trainloader, validloader = load_data_training(test_model,
                                                    batch_size,
                                                    train_data,
                                                    results_train,
                                                    val_data,
                                                    val_results,
                                                    channels = channels,
                                                    sparse_matrix=is_sparse,
                                                    is_zipped = True)
    limit = 6*700*5000
    LEARNING_RATE = 0.000001 # before 0.00001
    wDecay = 0.00005 # 0.005 could lead to too high regularization bc i could see that the model didnt not learn or was not flexible enough. the validation accuracies were about 10 % lower than training but a further factor to consider is that i didnt use dropout and my network was quiet big
    epochs = 12 
    train_optim = "ADAM"
    momentum = 0.95
    training_name = training_name + str(batch_size)
    Net = TheOneAndOnly(channels = channels,
                        test = test_model,
                        sequence_length = 30)
    out = Net(torch.randn(batch_size, 4, 6, 30, device="cpu"))
    print("test_successful")
    Net = TheOneAndOnly(channels = channels,
                        test = test_model, 
                        sequence_length = 30)
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

    if not test_model:
        train_loss_mean, valid_loss_mean,train_precision, valid_precision = training(Net,
                                                                                    optimizer,
                                                                                    epochs,
                                                                                    trainloader,
                                                                                    validloader,
                                                                                    base_dir,
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