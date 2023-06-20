import torch
from models.cnn_model import TheOneAndOnly
from models.simple_model import Simple
from save_training_results import log_training_info
#from models.interchange_model import Interchange
channels = 4
test_model = True
batch_size = 120
Net = TheOneAndOnly(channels = channels,
                    test = test_model,
                     sequence_length = 30)
out = Net(torch.randn(batch_size, 4, 6, 30, device="cpu"))
log_training_info("test.csv", 1, 2, 3, 4, 5, 6, 7, 8, 9)
print("test_successful")
