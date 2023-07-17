## General Purpose of this repository
This project contains all script to prepare the data and train on this, including the model and training script to predict the Open reading frame in ancient DNA samples from Exons and without additional damage.

### Data Preparation
The data for the training of the models is simulated using tools such as Gargammel, seqkit, leeHom and art_illumina to create reads from next generation sequencing. The script generate_reads.sh in data_preparation can be used to generate these where the read length and the number of fragments can be defined. Further the directory containing the genomes of the organisms (from ncbi) should be given. Additionally, the number of organisms from which the fragments should be generated can be given by defining max_dirs. The script generate_reads outputs for each genome one csv which can be used in furter processing steps. 
</b>
To be able to use the csv files for training the model, you have to first one-hot encode them so that you have four dimensions for the four nucleotides. The script one_hot.sh outputs six files which are:
</b>
- `--one_hot_encoded_all`: Training data for the model which contains the trimmed reads one hot encoded and arranged to subsets of six where five reads represent the true negatives and the remaining is the true positive
- `--results_all`: Training results for the model which contains the labels in subsets of six where five values are 0 and one is 1.
- `--one_hot_encoded_all_val`: Validation data for the model which contains the trimmed reads one hot encoded and arranged to subsets of six where five reads represent the true negatives and the remaining is the true positive
- `--results_all_val`: Validation results for the model which contains the labels in subsets of six where five values are 0 and one is 1.
- `--one_hot_encoded_all_test`: Test data for the model which contains the trimmed reads one hot encoded and arranged to subsets of six where five reads represent the true negatives and the remaining is the true positive
- `--results_all_test`: Test results for the model which contains the labels in subsets of six where five values are 0 and one is 1.
</b>
- '--file_name':
The inputs for one_hot.sh are
file_name="8000frag_2000orgs_33nt"
data_folder="/home/people/s220672/ReadsMatchProtein/results/6000frags_63nt"            #r"/home/projects/metagnm_asm/nils/dresults/middam"
base_dir_save=r"/home/people/s220672/ReadsMatchProtein/one_hot_encoded_tables"
train_seq_no=8000
val_seq_no=1000
test_seq_no=1000
limit_train_orgs=2000
limit_val_orgs=1000
limit_test_orgs=500
sequence_max_length=33
The data should be zipped after its generation and then encode_numbers.py can be used to one hot encode and prepare the data suitable for the neural network. After that the files should be zipped and chuncks should be created to avoid an exploding RAM during the training. Therefore chunck_script should be used for the training and validation data to split these in a given amount of subsets. Chunck script creates the chuncks in the same directory like the original datasets. So only the path to this directory is needed to indicate the dataset on which the model should be trained on.

### Training
The model is fairly big and if it is trained on more than 1000 organisms with more than 1000 fragments per organism it should be trained on high performance graphic nodes.  Tesla A100 PCIE 80 GB on an LSF cluster was used to train the model and for that the modules are defined in requirements.txt. Probably, these will differ for different architectures so they should be adjusted if other graphic cards are used. jobscript_big.sh was used to submit the jobs which is based on the script training.py. There one can define all relevant parameters for the training such as optimizer, batch size, data directory etc. 