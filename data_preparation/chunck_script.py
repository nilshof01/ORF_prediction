from data_preparation.one_hot_encoding import create_chuncks

train_data = r"/work3/s220672/ORF_prediction/processed/8000frag_2000orgs_30nt_middam/one_hot_blocks_all_8000frag_2000orgs_30nt_middam.npy.gz"
results_train = r"/work3/s220672/ORF_prediction/processed/8000frag_2000orgs_30nt_middam/results_all_8000frag_2000orgs_30nt_middam.npy.gz"
val_data = r"/work3/s220672/ORF_prediction/processed/8000frag_2000orgs_30nt_middam/one_hot_blocks_all_val_8000frag_2000orgs_30nt_middam.npy.gz"
val_results = r"/work3/s220672/ORF_prediction/processed/8000frag_2000orgs_30nt_middam/results_all_val_8000frag_2000orgs_30nt_middam.npy.gz"
num_subset = 20
create_chuncks(train_data, results_train, val_data, val_results, 4, num_subset)
    