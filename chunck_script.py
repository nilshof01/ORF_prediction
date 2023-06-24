from one_hot_encoding import create_chuncks

train_data = r"/work3/s220672/ORF_prediction/processed/6000frags_2000orgs_30nt/one_hot_blocks_all_6000frag_2000orgs_30nt.npy.gz"
results_train = r"/work3/s220672/ORF_prediction/processed/6000frags_2000orgs_30nt/results_all_6000frag_2000orgs_30nt.npy.gz"
val_data = r"/work3/s220672/ORF_prediction/processed/6000frags_2000orgs_30nt/one_hot_blocks_all_val_6000frag_2000orgs_30nt.npy.gz"
val_results = r"/work3/s220672/ORF_prediction/processed/6000frags_2000orgs_30nt/results_all_val_6000frag_2000orgs_30nt.npy.gz"
num_subset = 20
create_chuncks(train_data, results_train, val_data, val_results, 4, num_subset)
    