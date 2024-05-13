"""
make_dataset.py

This module contains functions to process raw data and create a final dataset ready for analysis.
"""
# -*- coding: utf-8 -*-
import os
import pickle
import sys
import numpy as np
from lib_ml_team11 import Preprocessing

# Create an instance of the Preprocessing class
preprocessing = Preprocessing()

def read_data(data_file):
    """
    Processes raw data and creates a final dataset ready for analysis.
    """
    with open(data_file, "r", encoding="utf-8") as f:
        data_lines = f.readlines()[1:]

    data = [line.strip() for line in data_lines]
    
    return data


def process(input_filepath, output_filepath, model_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    train = read_data(os.path.join(input_filepath, 'train.txt'))
    val = read_data(os.path.join(input_filepath, 'val.txt'))
    test = read_data(os.path.join(input_filepath, 'test.txt'))

    x_train, y_train, x_val, y_val, x_test, y_test = preprocessing.fit_transform(train, test, val, sequence_length=200)

    # Save processed data
    os.makedirs(output_filepath, exist_ok=True)
    tokenizer = preprocessing.get_tokenizer()
    with open(os.path.join(model_path, 'char_index.pkl'), 'wb') as f:
        pickle.dump(tokenizer.word_index, f)
    with open(os.path.join(model_path, 'tokenizer.pkl'), 'wb') as f:
        pickle.dump(tokenizer, f)
    np.save(os.path.join(output_filepath, 'x_train.npy'), x_train)
    np.save(os.path.join(output_filepath, 'x_val.npy'), x_val)
    np.save(os.path.join(output_filepath, 'x_test.npy'), x_test)
    np.save(os.path.join(output_filepath, 'y_train.npy'), y_train)
    np.save(os.path.join(output_filepath, 'y_val.npy'), y_val)
    np.save(os.path.join(output_filepath, 'y_test.npy'), y_test)


if __name__ == '__main__':
    process(sys.argv[1], sys.argv[2], sys.argv[3])
