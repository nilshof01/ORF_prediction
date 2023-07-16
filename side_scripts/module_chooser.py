import subprocess
import re
import os


def choose_cuda(train):
    output = os.popen("nvcc --version").read()
    test_version = "8.0"
    # Extract CUDA version using regular expressions
    match = re.search(r"release (\d+\.\d+)", output)
    
    serious_train_version = "11.0"
    cuda_version = match.group(1)
    print(type(cuda_version))
    print(cuda_version)
    if train:
        if cuda_version == test_version:
            pass
        else:
            os.popen("module swap cuda/8.0")
    else:
        if cuda_version == serious_train_version:
            pass
        else:
            os.popen("module swap cuda/11.0")
    output = os.popen("nvcc --version").read()
    match = re.search(r"release (\d+\.\d+)", output)
    print(match)
subprocess.run("module swap cuda/9.0", shell=True)
subprocess.call("module swap cuda/10.0", shell = True)
output = os.popen("nvcc --version").read()
match = re.search(r"release (\d+\.\d+)", output)
print(match)