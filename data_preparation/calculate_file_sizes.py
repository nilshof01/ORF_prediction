
def calculate_size(org_limit, train_seq_no, val_seq_no, test_seq_no):
    org_m = 0.48064444
    frag_m = 0.048063333
    size_mb_train = org_limit * org_m + train_seq_no * frag_m
    print("Your training data will have around: " + str(size_mb_train) + " Mb")
    size_mb_val = org_limit * org_m + val_seq_no * frag_m
    print("Your validation data will have around: " + str(size_mb_val) + " Mb")
    size_mb_test = org_limit * org_m + test_seq_no * frag_m
    print("Your test data will have around: " + str(size_mb_test) + " Mb")
    size_all = size_mb_train + size_mb_val + size_mb_test
    print("Overall your dataset will have around: " + str(size_mb_test) + " Mb")