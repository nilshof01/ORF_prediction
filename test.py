import torch

def test_model(model, input_tensor_3d):
    device = torch.device('cpu')
    batch_size = input_tensor_3d[0]
    num_channels = input_tensor_3d[1]
    num_sequences = input_tensor_3d[2] #6
    sequence_length = input_tensor_3d[3] # variable since

    out = model(torch.randn(batch_size,num_channels, num_sequences, sequence_length, device=device))
    print("Output shape:", out.size())
    print(f"Output logits:\n{out.detach().cpu().numpy()}")
    print(f"Output probabilities:\n{out}") # doesnt have softmax yet
