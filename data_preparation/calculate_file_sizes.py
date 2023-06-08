
def calculate_size(org_limit_train, org_limit_val, org_limit_test, train_seq_no, val_seq_no, test_seq_no):
    org_m = 1.921243
    frag_m = 0.16021111
    size_mb_train = org_limit_train * org_m + train_seq_no * frag_m
    print("Your training data will have around: " + str(size_mb_train) + " Mb")
    size_mb_val = org_limit_val * org_m + val_seq_no * frag_m
    print("Your validation data will have around: " + str(size_mb_val) + " Mb")
    size_mb_test = org_limit_test * org_m + test_seq_no * frag_m
    print("Your test data will have around: " + str(size_mb_test) + " Mb")
    size_all = size_mb_train + size_mb_val + size_mb_test
    print("Overall your dataset will have around: " + str(size_all) + " Mb")
    
calculate_size(10000*0.7, 1000, 300, 6000, 1000, 300)