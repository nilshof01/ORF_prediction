import torch
import torch.nn as nn
import torch.nn.functional as F
from line_eraser import EraseLines


class Interchange(nn.Module):
    def __init__(self,channels,test,   sequence_length = 30, first_channels_conv = [32, 64,256, 180], last_channels_conv = [32, 64,256,120, 240]): # horizontal kernel with horizontal stride
        super(Interchange, self).__init__()
        self.test_model = test

        self.first_layer_hor = nn.Sequential(
            nn.Conv2d(in_channels = channels, # will be 20 because you will one hot encode the aminoacid so that you have 20 values on the z axis
                      out_channels = last_channels_conv[0],
                      kernel_size = (1, 3),
                      stride = 1,
                      padding = (0, 1),
                      padding_mode="zeros"
                      ),
      
      #      EraseLines(p = 0.2, L = 2),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(last_channels_conv[0]),
            nn.ReLU(),
           # nn.
            nn.MaxAvg2d(kernel_size = (1, 3),
             #            stride = 1,
              #           padding = (0, 1))
            
        ) # before that i have to normalize on length somehow

        self.first_layer_ver = nn.Sequential(
            nn.Conv2d(in_channels = first_channels_conv[0], # will be 20 because you will one hot encode the aminoacid so that you have 20 values on the z axis
                      out_channels = last_channels_conv[1],
                      kernel_size = (3, 3),
                      stride = 1,
                      padding = (1, 1),
                      padding_mode="zeros"
                      ),
      
      #      EraseLines(p = 0.2, L = 2),
            nn.Dropout2d(0.4),
            nn.BatchNorm2d(last_channels_conv[1]),
            nn.ReLU(),
           # nn.MaxPool2d(kernel_size = (3, 3),
            #             stride = 1,
             #            padding = (0, 1))
            
        )
        self.second_conv_hor = nn.Sequential(
            nn.Conv2d(in_channels = first_channels_conv[1],
                      out_channels = last_channels_conv[1],
                      kernel_size = (1, 5),
                     stride = 1,
                      padding = (0, 2),
                      ),
      #      EraseLines(p = 0.2, L = 2),
            nn.Dropout2d(0.2),
            nn.BatchNorm2d(last_channels_conv[1]),
            nn.ReLU(),
          #  nn.MaxPool2d(kernel_size = (1, 3),
           #             stride = 1,
            #            padding = (0, 1)),
        ) # the idea between the previous and the following layer is that we can capture the duplet codon triplet bias for certain organisms. I used these two to prevent even kernel sizes which have no center point. But maybe the idea is bullshit
            ##vertical pooling to pick out the row with highest probability didnt work as initially thought
        self.second_conv_ver = nn.Sequential(
            nn.Conv2d(in_channels = first_channels_conv[1],
                      out_channels = last_channels_conv[2],
                    kernel_size = (3, 5),
                    stride = 1,
                    padding = (1, 2)),
            nn.Dropout2d(0.4),
            nn.BatchNorm2d(last_channels_conv[2]),
            nn.ReLU(),
            
        #    nn.MaxPool2d(kernel_size = (1, 3),
         #                stride = (1, 3),
          #               padding = (0, 1))
        )



     #   self.third_conv = nn.Sequential(
      #      nn.Conv2d(
       #             in_channels = first_channels_conv[1],
        #            out_channels = last_channels_conv[2],
         #           kernel_size = (1, 7),
          #      	stride = 1,
           #         padding = (0, 3)),
           # nn.Dropout2d(0.3),
           # nn.BatchNorm2d(last_channels_conv[2]),
           # nn.ReLU(),
            #nn.MaxPool2d(kernel_size = (1, 3),
             #            stride = (1, 3),
              #           padding = (0, 1))
       # )
    #    self.versus_pooling2 = nn.Sequential(
     #       nn.MaxPool2d(kernel_size=(3, 1),
      #                   stride = (3, 1)),
       # )

 


      #  padding = 0
       # stride = 1
       # filter_height = 6
       # filter_width = 20
       # output_height = (H - filter_height + 2 * padding) / stride + 1
        #output_width = (W - filter_width + 2 * padding) / stride + 1
        #num_filters = 64
        #num_units = num_filters * output_height * output_width
        input_dense = last_channels_conv[-1] * int(sequence_length/3) * 6
        
        self.dense_layer = nn.Sequential(
            nn.Linear(15360, 2500), # training showed that number of neurons about 5000 are too low. About 30000 was better.
            nn.BatchNorm1d(2500),
            nn.ReLU(),
            
        )
        ## maybe add normalization
        self.dense_layer2 = nn.Sequential(
            nn.Linear(2500,300),
            nn.BatchNorm1d(300),
            nn.ReLU()
        )

        self.dense_layer3 = nn.Sequential(
            nn.Linear(300,30),
            nn.BatchNorm1d(30),
            nn.ReLU()
        )
        self.dense_layer4 = nn.Sequential(
            nn.Linear(30, 6),

        )
    def forward(self, x):
 #       x = self.for_back_conv(x)
        x = self.first_layer_hor(x)
        if self.test_model == True:
            print(x.size())
        x = self.first_layer_ver(x)
        if self.test_model == True:
            print(x.size())
        x = self.second_conv_hor(x)
        if self.test_model==True:
            print(x.size())
        x = self.second_conv_ver(x)
        if self.test_model==True:
            print(x.size())
    #    x = self.versus_pooling(x)
      #  print(x.size())
     #   x = self.versus_pooling2(x)
      #  x = self.fourth_conv(x)

       # x = self.fifth_conv(x)

        num_features = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, num_features)

        x = self.dense_layer(x)
        
        x = self.dense_layer2(x)

        x = self.dense_layer3(x)

        x = self.dense_layer4(x)

        x = F.softmax(x, dim=1) # relative the logits to each other
        return x


