import numpy as np
import torch
from torch import nn
import torch.optim as optim
#from cnn_model_diff_act_func import TheOneAndOnly
from data_preparation.one_hot_encoding import create_dataset
import sys
from models.simple_model import Simple
from models.interchange_model import Interchange


base_dir = "/zhome/20/8/175218/orf_prediction"
batch_size = 120
channels = 4 # a network with channel 1 showed far less good results: maybe because the numbers do not equalize the nucleotides which is problematic in kernel operations
training_name = "1000_1000_pingpong_LR10-6"
limit = 6*700*5000
print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
test_model = False
if test_model != True:
    one_hot_blocks_all = np.load(r"/zhome/20/8/175218/orf_prediction/processed/test/one_hot_blocks_all_1000frags_1000o_.npy")
    #one_hot_blocks_all = one_hot_blocks_all.transpose((0, 1, 3, 2))
    print(one_hot_blocks_all.shape)
    results_all = np.load(r"/zhome/20/8/175218/orf_prediction/processed/test/results_all_1000frags_1000o_.npy")

    one_hot_blocks_all_val = np.load(r"/zhome/20/8/175218/orf_prediction/processed/test/one_hot_blocks_all_val_1000frags_1000o_.npy")
   # one_hot_blocks_all_val = one_hot_blocks_all_val.transpose((0, 1, 3, 2))
    print(one_hot_blocks_all_val.shape)
    results_all_val = np.load(r"/zhome/20/8/175218/orf_prediction/processed/test/results_all_val_1000frags_1000o_.npy")

    print("upload successful")
    print("Dimension of training block: " + str(one_hot_blocks_all.shape))
    one_hot_blocks_size = one_hot_blocks_all.nbytes / (1024 * 1024)
    print("size of training array: " + str(one_hot_blocks_size) + " Mb")


    trainloader, validloader = create_dataset(one_hot_blocks_all, #train one hot encoded data
                                            results_all, # train true labels
                                          one_hot_blocks_all_val, # validation one hot encoded data
                                            results_all_val, # validation true labels
                                             batch_size,
                                             channels = channels)
    size_trainloader = sys.getsizeof(trainloader)

    print("Size of trainloader:" + str(size_trainloader) + " bytes")
    del one_hot_blocks_all_val, one_hot_blocks_all, results_all, results_all_val

    LEARNING_RATE = 0.000001 # before 0.00001
    wDecay = 0.00005 # 0.005 could lead to too high regularization bc i could see that the model didnt not learn or was not flexible enough. the validation accuracies were about 10 % lower than training but a further factor to consider is that i didnt use dropout and my network was quiet big
    epochs = 20
    train_optim = "ADAM"
    momentum = 0.95
Net = Simple(channels = channels,
                    test = test_model,
                     sequence_length = 30)
out = Net(torch.randn(batch_size, 4, 6, 30, device="cpu"))
print("test_successful")
Net = Interchange(channels = channels,
                    test = test_model, 
                    sequence_length = 30)
Net = Net.to(device)
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
criterion = nn.CrossEntropyLoss() # not binary cross entropy loss. You do not have binary cross entropy because you have a multi class problem. multi class = 6 different open reading frames. Thats why you cannot classify with binary cross entropy.


step = 0
valid_precision = []
train_precision = []
for e in range(epochs):

    train_loss = []
    valid_loss = []
    n_totalT = 0
    train_precision_batches = []
    valid_precision_batches = []
    Net.train()  # Optional when not using Model Specific layer
    for num, local_batch in enumerate(trainloader, 0):

        data, target = local_batch
        if torch.cuda.is_available():
            data = data.to(torch.device("cuda"))
            target = target.to(torch.device("cuda"))
            # data, target = data.cuda(), target.cuda()
        else:
            print("device not assigned")
        optimizer.zero_grad()
        data = data.float()
        target = target.float()

        # forward pass
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
    train_precision.append(np.mean(np.array(train_precision_batches)))
    train_loss_np = [loss.item() for loss in train_loss]
    train_loss_mean = np.mean(np.array(train_loss_np))

    with torch.no_grad():
        Net.eval()
 
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
        valid_precision.append(np.mean(np.array(valid_precision_batches)))
        Net.train()
        valid_loss_np = [loss.item() for loss in valid_loss]
        valid_loss_mean = np.mean(np.array(valid_loss_np))
        inputs = inputs.to(torch.device("cpu"))
        seqs_batch = inputs[:10]
        n = 0
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


with open(base_dir + "/" + training_name + ".txt", 'w') as f:
    for item in train_precision:
        f.write("%s\n" % item)
with open(base_dir + "/" + training_name + ".txt", 'w') as f:
    for item in valid_precision:
        f.write("%s\n" % item)
