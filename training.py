import numpy as np
import torch
from torch import nn
import torch.optim as optim

print(torch.cuda.is_available())
device = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.00005
wDecay = 0.0004
epochs = 50
Net = model
Net = Net.to(device)
min_valid_loss = np.inf
optimizer = optim.Adam(Net.parameters(),
                       lr=LEARNING_RATE,
                       weight_decay=wDecay)
criterion = nn.BCELoss() # binary cross entropy loss
batch_size = 22
# criterion = nn.BCELoss()
step = 0
for e in range(epochs):

    train_loss = 0.0
    n_totalT = 0
    train_accuracies_batches = []
    valid_accuracies_batches = []
    Net.train()  # Optional when not using Model Specific layer
    for num, local_batch in enumerate(trainloader, 0):

        data, target = local_batch
        if torch.cuda.is_available():
            data = data.to(torch.device("cuda"))
            target = target.to(torch.device("cuda"))
            # data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        data = data.float()
        target = target.float()

        # forward pass
        prediction = Net(data)
        # prediction = prediction[:, 1:8]

        #  target = target[:, 1:8]

        # backward pass
        loss = criterion(prediction, target)

        #   loss.requires_grad = True
        # calculate gradients
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        step = + 1
        n_totalT += 1
        predictions = prediction.max(1)[1]
        train_accuracies_batches.append(accuracy(target, predictions))
    valid_loss = 0.0
    n_totalV = 0
    Net.eval()  # Optional when not using Model Specific layer
    for num_val, proof_batch in enumerate(validloader, 0):
        data, target = proof_batch
        if torch.cuda.is_available():
            data = data.to(torch.device("cuda"))
            target = target.to(torch.device("cuda"))
        # data, target = data.cuda(), target.cuda()

        data = data.float()
        target = target.float()
        prediction = Net(data)
        # prediction = prediction[:, 1:8]
        # target = target[:, 1:8]
        # print(target[1:8].size())
        # print(target.size())
        # print(prediction[1:8].size())
        # print(prediction.size())

        loss = criterion(prediction, target)
        # print(loss)
        valid_loss += loss.item() * data.size(0)
        predictions = prediction.max(1)[1]
        n_totalV += 1
        valid_accuracies_batches.append(accuracy(target, predictions))
    print(
        f'Epoch {e + 1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / n_totalV}')
    print(
        f'Epoch {e + 1} \t\t Train Accuracy: {sum(train_accuracies_batches) / n_totalT} \t\t Validation Accuracy: {sum(valid_accuracies_batches) / n_totalV}')
    if min_valid_loss > valid_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        min_valid_loss = valid_loss
        # Saving State Dict
        torch.save(Net.state_dict(), 'saved_model.pth')

model = torch.load("/zhome/20/8/175218/saved_model.pth")