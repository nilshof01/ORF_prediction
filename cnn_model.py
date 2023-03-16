import torch
import torch.nn as nn
import torch.nn.functional as F
first_channels_conv = [20, 40, 80, 80, 120]
last_channels_conv = [40, 80, 80, 120]
class TheOneAndOnly(nn.Module):
    def __init__(self,  sequence_length = 60): # horizontal kernel with horizontal stride
        super(TheOneAndOnly, self).__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels = 20, # will be 20 because you will one hot encode the aminoacid so that you have 20 values on the z axis
                      out_channels = 40,
                      kernel_size = (1, 3),
                      stride = (1, 1),
                      padding = (0, 1),
                      padding_mode="zeros"
                      ),
            nn.BatchNorm2d(40),
            nn.ReLU()
        ) # before that i have to normalize on length somehow
        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels = 40,
                      out_channels = 80,
                      kernel_size = (1, 5),
                      stride = (1, 1),
                      padding = (0, 2),
                      ),
            nn.BatchNorm2d(80),
            nn.ReLU()
        ) # the idea between the previous and the following layer is that we can capture the duplet codon triplet bias for certain organisms. I used these two to prevent even kernel sizes which have no center point. But maybe the idea is bullshit
        self.third_conv = nn.Sequential(
            nn.Conv2d(in_channels=80,
                      out_channels = 80,
                      kernel_size = (1, 7),
                      stride = (1, 1),
                      padding = (0, 3)
                      ),
            nn.BatchNorm2d(80),
            nn.ReLU()
        )
        self.fourth_conv = nn.Sequential(
            nn.Conv2d(in_channels= 80,
                        out_channels= 120,
                        kernel_size = (1, 3),
                        stride = (1, 3)), # the kernel moves three steps so you consider triplets now. This should identify stop codons for instance
                        nn.BatchNorm2d(120),
                        nn.ReLU(),
                                )# if sequence length = 60: 40 x


      #  padding = 0
       # stride = 1
       # filter_height = 6
       # filter_width = 20
       # output_height = (H - filter_height + 2 * padding) / stride + 1
        #output_width = (W - filter_width + 2 * padding) / stride + 1
        #num_filters = 64
        #num_units = num_filters * output_height * output_width
        input_dense = last_channels_conv[-1] * sequence_length/3 * 6
        self.dense_layer = nn.Linear(input_dense, 300)
        ## maybe add normalization
        self.dense_layer2 = nn.Linear(300, 6)
    def forward(self, x):
        x = self.first_layer(x)

        x = self.second_conv(x)

        x = self.third_conv(x)

        x = self.fourth_conv(x)

        num_features = x.size(1) * x.size(2) * x.size(3)
        x = x.view(-1, num_features)

        x = self.dense_layer(x)

        x = self.dense_layer2(x)

        x = nn.functional.sigmoid(x)
        x = F.softmax(x, dim=1) # relative the logits to each other
        return x


model = TheOneAndOnly()
device = torch.device('cpu')
out = model(torch.randn(20, 20, 6, 60, device=device))

