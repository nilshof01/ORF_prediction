import torch
import torch.nn as nn
import torch.nn.functional as F
from experiment.line_eraser import EraseLines


class TheOneAndOnly(nn.Module):
    def __init__(self,channels,   sequence_length = 30, first_channels_conv = [128, 256, 512, 180], last_channels_conv = [128, 256, 512,120, 240]): # horizontal kernel with horizontal stride
        super(TheOneAndOnly, self).__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels = channels, # will be 20 because you will one hot encode the aminoacid so that you have 20 values on the z axis
                      out_channels = last_channels_conv[0],
                      kernel_size = (1, 3),
                      stride = 1,
                      padding = (0, 1),
                      padding_mode="zeros"
                      ),
            EraseLines(p = 0.2, L = 2),
            nn.Dropout2d(0.3),
            nn.BatchNorm2d(last_channels_conv[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1, 3),
                        stride = 1,
                         padding = (0, 1))
            
        ) # before that i have to normalize on length somehow
        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels = first_channels_conv[0],
                      out_channels = last_channels_conv[1],
                      kernel_size = (1, 5),
                      stride = 1,
                      padding = (0, 2),
                      ),
       #    EraseLines(p = 0.2, L = 2),
            nn.Dropout2d(0.3),
            nn.BatchNorm2d(last_channels_conv[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = (1, 5),
                         stride = 1,
                         padding = (0, 2))

        ) # the idea between the previous and the following layer is that we can capture the duplet codon triplet bias for certain organisms. I used these two to prevent even kernel sizes which have no center point. But maybe the idea is bullshit
    #    self.third_conv = nn.Sequential(
     #       nn.Conv2d(in_channels=first_channels_conv[1],
      #                out_channels = last_channels_conv[2],
       #               kernel_size = (1, 3),
        #              stride = 1,
         #             padding = (0,)
          #            ),
           # EraseLines(p = 0.2, L = 2),
           # nn.BatchNorm2d(last_channels_conv[2]),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(1,3), stride=3)
        #)

        self.multi_start_layer = nn.Sequential(
                                nn.Conv2d(first_channels_conv[1], 
                                            last_channels_conv[2],
                                            kernel_size=(1, 3),
                                            stride=1),
                                nn.MaxPool2d(kernel_size = (1,3),
                                            stride = 3),
                                nn.BatchNorm2d(last_channels_conv[2]),
                                nn.ReLU()
                                            )

        input_dense = last_channels_conv[-1] * int(sequence_length/3) * 6
        self.dense_layer = nn.Sequential(
            nn.Linear(10240, 2500),
            nn.BatchNorm1d(2500),
            nn.ReLU(),
            
        )
        ## maybe add normalization
        self.dense_layer2 = nn.Sequential(
            nn.Linear(2500, 500),
            nn.BatchNorm1d(500),
            nn.ReLU()
        )

        self.dense_layer3 = nn.Sequential(
            nn.Linear(500, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )
        self.dense_layer4 = nn.Sequential(
            nn.Linear(100, 6),

        )
    def forward(self, x):
 #       x = self.for_back_conv(x)
        x = self.first_layer(x)
        print(x.size())
        x = self.second_conv(x)
        print(x.size())

     #   x = self.third_conv(x)
        outputs = []
        for padding in range(1,3):
            padded_x = nn.functional.pad(x, (padding, 0, )) # tuple for padding: (left, right, top, bottom)
            output = self.multi_start_layer(padded_x)
            outputs.append(output)
        x = torch.cat(outputs, dim=1)
        print(x.size())

      #  x = self.fourth_conv(x)

       # x = self.fifth_conv(x)

        num_features = x.size(1) * x.size(2) * x.size(3)
        x = x.view(x.size(0), -1)
        print(x.size())
        x = self.dense_layer(x)

        x = self.dense_layer2(x)

        x = self.dense_layer3(x)

        x = self.dense_layer4(x)

      #  x = nn.functional.sigmoid(x)
        x = F.softmax(x, dim=1) # relative the logits to each other
        return x


