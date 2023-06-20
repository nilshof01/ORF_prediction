import glob
import os
train_dir = r"/work3/s220672/ORF_prediction/processed/1000frag_10000orgs"
train_filenames =[file for file in glob.glob(os.path.join(train_dir, "*")) if not file.endswith("val.pt.gz")]
print(train_filenames)
validation_filenames = glob.glob(os.path.join(train_dir,  "*val.pt.gz"))
assert len(validation_filenames) > 0, "No validation files found."