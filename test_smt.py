from one_hot_encoding import create_chuncks


create_chuncks("/zhome/20/8/175218/orf_prediction/processed/1000frags_10000orgs/one_hot_blocks_all_1000frag_10000orgs.npy.gz", "/zhome/20/8/175218/orf_prediction/processed/1000frags_10000orgs/results_all_1000frag_10000orgs.npy.gz", "/zhome/20/8/175218/orf_prediction/processed/1000frags_10000orgs/one_hot_blocks_all_val_1000frag_10000orgs.npy.gz", "/zhome/20/8/175218/orf_prediction/processed/1000frags_10000orgs/results_all_val_1000frag_10000orgs.npy.gz", channels = 4, num_subsets = 10, sparse_matrix = False, is_zipped = True)
