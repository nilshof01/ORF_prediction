from one_hot_encoding import create_chuncks

train_data = r"/work3/s220672/ORF_prediction/processed/6000frags_5000orgs/one_hot_blocks_all_6000frags_5000o_.npy.gz"
results_train = r"/work3/s220672/ORF_prediction/processed/6000frags_5000orgs/results_all_6000frags_5000o_.npy.gz"
val_data = r"/work3/s220672/ORF_prediction/processed/6000frags_5000orgs/one_hot_blocks_all_val_6000frags_5000o_.npy.gz"
val_results = r"/work3/s220672/ORF_prediction/processed/6000frags_5000orgs/results_all_val_6000frags_5000o_.npy.gz"
    
create_chuncks(train_data, results_train, val_data, val_results, channels = 4)
    