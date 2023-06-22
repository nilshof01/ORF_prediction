import pandas as pd
import os


def log_training_info(filename, training_loss, training_accuracy, validation_loss, validation_accuracy, epoch_number,
                      learning_rate_value, weight_decay_value, batch_size_value, data_input):
    # Create an empty DataFrame
    df = pd.DataFrame()
   # training_loss = training_loss.tolist()
    #validation_loss = validation_loss.tolist()
    # If the file already exists, read the existing data
    if os.path.isfile(filename):
        df = pd.read_csv(filename)

    list_len = len(training_loss)
    print(validation_accuracy)
    print(validation_loss)
    print(training_loss)
    print(training_accuracy)
    # Create a dictionary with the data
    new_data = {
        'Epoch': [epoch_number] * list_len,
        'Training Loss': training_loss,
        'Training Accuracy': training_accuracy,
        "Validation Loss": validation_loss,
        'Validation Accuracy': validation_accuracy,
        'Learning Rate': [learning_rate_value] * list_len,
        'Weight Decay': [weight_decay_value] * list_len,
        'Batch Size': [batch_size_value] * list_len,
        'Data Input': [data_input] * list_len
    }

    # Create a DataFrame from the dictionary
    intermediate_frame = pd.DataFrame(new_data)

    # Concatenate the new DataFrame with the existing DataFrame
    df = pd.concat([df, intermediate_frame])

    # Save DataFrame to csv
    df.to_csv(filename, index=False)