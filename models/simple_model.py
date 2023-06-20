import torch
import torch.nn as nn
import torch.nn.functional as F
from line_eraser import EraseLines


class Simple(nn.Module):
    def __init__(self,channels,test,   sequence_length = 30, first_channels_conv = [64, 128,256, 180], last_channels_conv = [64, 128,256,120, 240]): # horizontal kernel with horizontal stride
        super(Simple, self).__init__()
        self.test_model = test
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels = channels, # will be 20 because you will one hot encode the aminoacid so that you have 20 values on the z axis
                      out_channels = last_channels_conv[0],
                      kernel_size = (1, 3),
                      stride = 1,
                      padding = (0, 1),
                      padding_mode="zeros"
                      ),
      
      #      EraseLines(p = 0.2, L = 2),
            nn.Dropout2d(0.25),
            nn.BatchNorm2d(last_channels_conv[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1, 3),
                         stride = (1, 3),
                         padding = (0, 1))
            
        ) # before that i have to normalize on length somehow



        input_dense = last_channels_conv[-1] * int(sequence_length/3) * 6
        
        self.dense_layer = nn.Sequential(
            nn.Linear(3840, 300), # training showed that number of neurons about 5000 are too low. About 30000 was better.
            nn.BatchNorm1d(300),
            nn.ReLU(),
            
        )
        ## maybe add normalization
        self.dense_layer2 = nn.Sequential(
            nn.Linear(300,30),
            nn.BatchNorm1d(30),
            nn.ReLU()
        )
        self.dense_layer3 = nn.Sequential(
            nn.Linear(30, 6)
        )


    def forward(self, x):
 #       x = self.for_back_conv(x)
        x = self.first_layer(x)
        if self.test_model == True:
            print(x.size())


        num_features = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, num_features)

        x = self.dense_layer(x)
        
        x = self.dense_layer2(x)

        x = self.dense_layer3(x)

       # x = nn.functional.sigmoid(x)  ## 
        x = F.softmax(x, dim=1) # you have a multi class classificatrion problem because you have 6 different open reading frame which have to be classified. One of the classes is 1 while the rest is 0.
        return x


