## General Purpose of this repository
This project contains all script to prepare the data and train on this, including the model and training script to predict the Open reading frame in ancient DNA samples from Exons and without additional damage.

### Data Preparation
The data for the training of the models is simulated using tools such as Gargammel, seqkit, leeHom and art_illumina to create reads from next generation sequencing. The script generate_reads.sh in data_preparation can be used to generate these where the read length and the number of fragments can be defined. Further the directory containing the genomes of the organisms (from ncbi) should be given. Additionally, the number of organisms from which the fragments should be generated can be given by defining max_dirs. The script generate_reads outputs for each genome one csv which can be used in furter processing steps. 
</b>
To be able to use the csv files for training the model, you have to first one-hot encode them so that you have four dimensions for the four nucleotides. The script one_hot.sh outputs six files which are:
</b>

- `--one_hot_encoded_all`: Training data for the model. Contains the trimmed reads, one-hot encoded and arranged in subsets of six, where five reads represent the true negatives and the remaining is the true positive.
- `--results_all`: Training results for the model. Contains the labels in subsets of six, where five values are 0 and one is 1.
- `--one_hot_encoded_all_val`: Validation data for the model. Contains the trimmed reads, one-hot encoded and arranged in subsets of six, where five reads represent the true negatives and the remaining is the true positive.
- `--results_all_val`: Validation results for the model. Contains the labels in subsets of six, where five values are 0 and one is 1.
- `--one_hot_encoded_all_test`: Test data for the model. Contains the trimmed reads, one-hot encoded and arranged in subsets of six, where five reads represent the true negatives and the remaining is the true positive.
- `--results_all_test`: Test results for the model. Contains the labels in subsets of six, where five values are 0 and one is 1.

#### one_hot.sh script inputs

- `--file_name`: The identifier name of the output_files - the script will attach to `one_hot_encoded_all` the given name.
- `--data_folder`: The folder where you store the csv files generated by `generate_reads.sh`.
- `--base_dir_save`: The folder where you want to store the output files.
- `--train_seq_no`: The number of sequences you want to take from each csv file for the training.
- `--val_seq_no`: The number of sequences you want to take from each csv file for validation.
- `--test_seq_no`: The number of sequences you want to take from each csv file for testing.
- `--limit_train_orgs`: The number of organisms you want to include in your train dataset (equals the number of csv files).
- `--limit_val_orgs`: The number of organisms you want to include in your validation dataset (equals the number of csv files).
- `--limit_test_orgs`: The number of organisms you want to include in your test dataset (equals the number of csv files).
- `--sequences_max_length`: The sequences length you want to have in your dataset (all sequences will have this length).

**Note**: After the one hot encoding you should zip the data because the files will be relatively large. Lastly, you need to create chunks from your training and validation dataset because the training algorithm was designed to only accept corresponding files to reduce the demand in RAM. For that, run `generate_chunks.sh` in which you have to define the following inputs:

#### generate_chunks.sh script inputs

- `--train_data`: Filepath to `one_hot_encoded_all` (training data).
- `--results_train`: Filepath to `results_all` (training labels).
- `--val_data`: Filepath to `one_hot_encoded_val` (validation data).
- `--val_results`: Filepath to `results_all_val` (validation labels).
- `--num_subsets`: Number of chunks (subsets) generated by the script.
The script will output the number of subsets for each file in the folder where the data was stored. It is necessary to have the four datasets in the same directory.

### Training
The model is fairly big and if it is trained on more than 1000 organisms with more than 1000 fragments per organism it should be trained on high performance graphic nodes.  Tesla A100 PCIE 80 GB on an LSF cluster was used to train the model and for that the modules are defined in requirements.txt. Probably, these will differ for different architectures so they should be adjusted if other graphic cards are used. jobscript_big.sh was used to submit the jobs which is based on the script src/training_model/training.py. There, one can define all relevant parameters for the training such as optimizer, batch size, data directory etc. 