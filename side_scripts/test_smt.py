import glob
import os
import torch
import gzip
import io

train_dir = r"/work3/s220672/ORF_prediction/processed/1000frag_5000orgs"
train_filenames =[file for file in glob.glob(os.path.join(train_dir, "*")) if not "val" in file]
print(train_filenames)
print(len(train_filenames))
print([string for string in train_filenames if "_X" in string])
for train_file in range(int(len(train_filenames)/2)): #chunck puller
    X = [string for string in train_filenames if "subset_" + str(train_file + 1) + "_X" in string]
    print(X[0])
    print(train_file)
validation_filenames = glob.glob(os.path.join(train_dir,  "*val.pt.gz"))
assert len(validation_filenames) > 0, "No validation files found."
batch_size = 120
def unzip_memory(file):
    with gzip.open(file, 'rb') as f:
        uncompressed_data = f.read()
    array_data = torch.load(io.BytesIO(uncompressed_data))
    return array_data

#X = unzip_memory(r"/work3/s220672/ORF_prediction/processed/1000frag_10000orgs/subset_1_X.pt.gz")
#Y = unzip_memory(r"/work3/s220672/ORF_prediction/processed/1000frag_10000orgs/subset_1_Y.pt.gz")
#print(X.shape)
#print(Y.shape)
#dset_train = torch.utils.data.TensorDataset(X, Y)  # merge both together in a dataaset
#set_loader = torch.utils.data.DataLoader(dset_train,
 #                                   batch_size=batch_size,  # choose your batch size
  #                                  shuffle=True)  # generally a good idea
#print(len(set_loader))