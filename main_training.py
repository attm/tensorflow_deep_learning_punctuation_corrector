from encdec_model_builder.lstm_atn_builder import build as build_lstm_atn
import tensorflow as tf
from os.path import join as pjoin
import os
import numpy as np


cwd = os.getcwd()
DATA_FOLDER = pjoin(cwd, "data")

# Preparing GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def load_datasets() -> np.ndarray:
    i1 = np.load(pjoin(DATA_FOLDER, "input1.npy"))
    i2 = np.load(pjoin(DATA_FOLDER, "input2.npy"))
    tg = np.load(pjoin(DATA_FOLDER, "target.npy"))
    return i1, i2, tg

def split_dataset(dataset : np.ndarray, head_count : int = -1, train_split : float = 0.8) -> np.ndarray:
    """
    Splitting dataset and taking only specifiend count of dataset.

    Parameters:
        dataset (np.ndarray) : original dataset
        head_count (int) : number of sample to be took
        train_split (float) : part of dataset to be train, test will be other part
    Returns:
        train_dataset (np.ndarray) : train dataset
        test_dataset (np.ndarray) : test dataset
    """
    # Check parameters
    if head_count > len(dataset):
        print("split_dataset: head_count is more than dataset len, returning None")
        return None
    
    if train_split <= 0.0 or train_split >= 1.0:
        print("split_dataset: train_split must be between 0.0 and 1.0, returning None")
        return None

    processed_dataset = dataset
    # Taking head count
    if head_count > 0:
        processed_dataset = processed_dataset[:head_count]

    train_count = int(len(processed_dataset) * train_split)

    train_dataset = processed_dataset[:train_count]
    test_dataset = processed_dataset[train_count:]
    
    return train_dataset, test_dataset


def main_train() -> None:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model, encoder, decoder = build_lstm_atn(optimizer=optimizer)

    i1, i2, tg = load_datasets()

    i1_train, i1_test = split_dataset(i1, head_count=20000, train_split=0.8)
    i2_train, i2_test = split_dataset(i2, head_count=20000, train_split=0.8)
    tg_train, tg_test = split_dataset(tg, head_count=20000, train_split=0.8)

    model.fit([i1_train, i2_train], tg_train, batch_size=16, epochs=1, validation_data=([i1_test, i2_test], tg_test))

if __name__ == "__main__":
    main_train()