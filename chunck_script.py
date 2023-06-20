from one_hot_encoding import create_chuncks

train_data = r"/work3/s220672/ORF_prediction/processed/6000frag_1000orgs_35nt/one_hot_blocks_all_6000frag_1000orgs_35nt.npy.gz"
results_train = r"/work3/s220672/ORF_prediction/processed/6000frag_1000orgs_35nt/results_all_6000frag_1000orgs_35nt.npy.gz"
val_data = r"/work3/s220672/ORF_prediction/processed/6000frag_1000orgs_35nt/one_hot_blocks_all_val_6000frag_1000orgs_35nt.npy.gz"
val_results = r"/work3/s220672/ORF_prediction/processed/6000frag_1000orgs_35nt/results_all_val_6000frag_1000orgs_35nt.npy.gz"
num_subset = 10
create_chuncks(train_data, results_train, val_data, val_results, 4, num_subset)
    