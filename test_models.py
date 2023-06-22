import torch
from models.cnn_model import TheOneAndOnly
from models.simple_model import Simple
from models.small_cnn import TheOneAndOnly
from save_training_results import log_training_info
#from models.interchange_model import Interchange
channels = 4
test_model = True
batch_size = 120
Net = TheOneAndOnly(channels = channels,
                    test = test_model,
                    sequence_length = 33,)
out = Net(torch.randn(batch_size, 4, 6, 33, device="cpu"))
