import sys
import numpy as np
import torch
from torch import nn
import torch.optim as optim
#from cnn_model_diff_act_func import TheOneAndOnly
from models.cnn_model import TheOneAndOnly
from one_hot_encoding import create_dataset
import sys
from decode import decode_sequence
from models.simple_model import Simple
#from models.interchange_model import Interchange
from save_training_results import log_training_info
import os
import re
#import scipy.sparse as sp
import subprocess
import gzip
import io
import psutil
import torch.autograd.profiler as profiler
from torch.cuda.amp import autocast, GradScaler
import glob



def unzip_memory(file):
    with gzip.open(file, 'rb') as f:
        uncompressed_data = f.read()
    array_data = torch.load(io.BytesIO(uncompressed_data))
    return array_data

def zip_file(file_path):
    subprocess.run(['gzip', file_path])
    
def check_max_ram(max_ram_usage, position):
    ram_usage = psutil.Process().memory_info().rss / 1024 ** 2  # RAM usage in MB
    if ram_usage > max_ram_usage:
        max_ram_usage = ram_usage
        position = position
    return max_ram_usage, position
    
def training(Net, optimizer, epochs, base_dir,batch_size, train_dir, training_name):
    max_ram_usage = 0
    criterion = nn.CrossEntropyLoss() # not binary cross entropy loss. You do not have binary cross entropy because you have a multi class problem. multi class = 6 different open reading frames. Thats why you cannot classify with binary cross entropy.
    step = 0
    valid_precision = []
    train_precision = []
    mean_val_loss_all = []
    mean_train_loss_all = []
    train_filenames =[file for file in glob.glob(os.path.join(train_dir, "*")) if not "val" in file]
    print(train_filenames)
    print(len(train_filenames))
    assert len(train_filenames) > 0, "No training files found."
    validation_filenames = glob.glob(os.path.join(train_dir,  "*val.pt.gz"))
    print(len(validation_filenames))
    assert len(validation_filenames) > 0, "No validation files found."
    max_ram_usage, position = check_max_ram(max_ram_usage, 1)
    for e in range(epochs):
        train_loss = []
        valid_loss = []
        n_totalT = 0
        train_precision_batches = []
        valid_precision_batches = []
        Net.train()  # Optional when not using Model Specific layer
        if os.environ.get("CUDA_PROFILE") == "1":
            print(prof.key_averages().table(sort_by="cuda_time_total"))
        max_ram_usage, position = check_max_ram(max_ram_usage, 2)
       # with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        for train_file in range(int(len(train_filenames)/2)): #chunck puller
            X = [string for string in train_filenames if "subset_" + str(train_file + 1) + "_X" in string]
            Y = [string for string in train_filenames if "subset_" + str(train_file + 1) + "_Y" in string]
      #      if not os.path.isfile(X[0]) or not os.path.isfile(Y[0]):
       #         print("The training data file does not exist.")

            if os.path.isfile(X[0]) and os.path.isfile(Y[0]):
                subset_X = unzip_memory(X[0])
                subset_Y = unzip_memory(Y[0])
                
                trainloader = create_dataset(subset_X, subset_Y, batch_size, mode = "train")	
                print("trainloader created")

                del subset_X
                del subset_Y
                for data, target in trainloader: # mini batch gradient descent
                    #ram_usage = psutil.Process().memory_info().rss / 1024 ** 2  # RAM usage in MB
                    #print(f"RAM usage epoch {e:.1f} beginning: {ram_usage:.2f} MB")
                    
                    if torch.cuda.is_available():
                        data = data.to(torch.device("cuda"))
                        target = target.to(torch.device("cuda"))
                        data = data.float()
                        target = target.float()
                        # data, target = data.cuda(), target.cuda()
                    else:
                        print("device not assigned")
                    optimizer.zero_grad() # releases ram 

                    with torch.autograd.profiler.profile(enabled=(os.environ.get("CUDA_PROFILE") == "1")) as prof:
                        prediction = Net(data)
                    # backward pass
                    # predicted values should be percentage values because loss function needs predicted probabilities
                    loss = criterion(prediction, target)
                    
                    predicted_labels = torch.argmax(prediction, dim=1)
                    target_labels = torch.argmax(target, dim=1)
                    correct_predictions = (predicted_labels == target_labels).sum().item()
                    total_predictions = target_labels.shape[0]
                    precision = correct_predictions/total_predictions

                    #   loss.requires_grad = True
                    # calculate gradients

                    loss.backward()

                    optimizer.step()
                    train_loss.append(loss)
                    step = + 1
                    n_totalT += 1
                    predictions = prediction.max(1)[1]
                    train_precision_batches.append(precision)
                    if e==(epochs-1):
                #      print(prediction)
                    #    print(target)
                        pass
                    del data
                    del target
                    max_ram_usage, position = check_max_ram(max_ram_usage, 3)
                    
            #    print(prof.key_averages().table(sort_by="cuda_time_total"))
                del trainloader
        max_ram_usage, position = check_max_ram(max_ram_usage,4)
        optimizer.zero_grad() # releases ram 
        train_precision.append(np.mean(np.array(train_precision_batches)))
        train_loss_np = [loss.item() for loss in train_loss]
        train_loss_mean = np.mean(np.array(train_loss_np))
        mean_train_loss_all.append(train_loss_mean)
        max_ram_usage, position = check_max_ram(max_ram_usage, 5)
        with torch.no_grad():
            Net.eval()
            max_ram_usage, position = check_max_ram(max_ram_usage, 6)
            for valid_file in range(int(len(validation_filenames)/2)):
                X = [string for string in validation_filenames if "subset_" + str(valid_file + 1) + "_X" in string]
                print(X[0])
                Y = [string for string in validation_filenames if "subset_" + str(valid_file + 1) + "_Y" in string]
                subset_X = unzip_memory(X[0])
                subset_Y = unzip_memory(Y[0])
                validloader = create_dataset(subset_X, subset_Y, batch_size, mode = "val")	
                del subset_X
                del subset_Y
                max_ram_usage, position = check_max_ram(max_ram_usage, 7)
                for inputs, targets in validloader:
                    
                    inputs = inputs.to(torch.device("cuda"))
                    targets = targets.to(torch.device("cuda"))
                    inputs = inputs.float()
                    targets = targets.float()
                    output = Net(inputs)
                    loss = criterion(output, targets)
                    predicted_labels = torch.argmax(output, dim=1)
                    target_labels = torch.argmax(targets, dim=1)
                    correct_predictions_val = (predicted_labels == target_labels).sum().item()
                    total_predictions = target_labels.shape[0]
                    valid_loss.append(loss)
                    precision = correct_predictions_val/total_predictions
                    valid_precision_batches.append(precision)
                    max_ram_usage, position = check_max_ram(max_ram_usage, 8)
                    
                del validloader
            valid_precision.append(np.mean(np.array(valid_precision_batches)))
            valid_loss_np = [loss.item() for loss in valid_loss]
            valid_loss_mean = np.mean(np.array(valid_loss_np))
            mean_val_loss_all.append(valid_loss_mean)
            seqs_batch = inputs[:10]
            max_ram_usage, position = check_max_ram(max_ram_usage, 9)
                
                #for i in seqs_batch:
                    #list_sequences = decode_sequence(i, channel = channels)
                    #for j in list_sequences:
                    #   print(j)
            #     print(output[n])
                #    print(target_labels[n])
        #         n = n+1

                
            print(
                f'Epoch {e + 1} \t\t Training Loss: {train_loss_mean} \t\t Validation Loss: {valid_loss_mean}')
            print(
                f'Epoch {e + 1} \t\t Train Precision: {np.mean(np.array(train_precision_batches))} \t\t Validation Precision: {np.mean(np.array(valid_precision_batches))}')
        #  if min_valid_loss > valid_loss:
        #     print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            #    min_valid_loss = valid_loss
            #   # Saving State Dict
            #  torch.save(Net.state_dict(), 'saved_model.pth')

        #model = torch.load("/zhome/20/8/175218/saved_model.pth")
            ram_usage = psutil.Process().memory_info().rss / 1024 ** 2  # RAM usage in MB
            print(f"RAM usage epoch {e:.1f} end: {ram_usage:.1f} MB")
        print("Max ram usage per loop: " + str(max_ram_usage))
        print("position max ram:"  + str(position)) 
    with open(base_dir + "/" + training_name + ".txt", 'w') as f:
        for item in train_precision:
            f.write("%s\n" % item)
    with open(base_dir + "/" + training_name + ".txt", 'w') as f:
        for item in valid_precision:
            f.write("%s\n" % item)
    train_loss_mean = train_loss_mean.tolist()
    valid_loss_mean = valid_loss_mean.tolist()

    return mean_train_loss_all, mean_val_loss_all, train_precision, valid_precision



