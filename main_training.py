from encdec_model_builder.lstm_simple_builder import build as build_lstm
from encdec_model.predictor import EncDecPredictor
from tensorflow.keras.preprocessing.text import Tokenizer
from os.path import join as pjoin
from config import training_config

import os 
import pickle

import tensorflow as tf
import numpy as np


cwd = os.getcwd()
DATA_FOLDER = pjoin(cwd, "data")
SAVED_MODELS_FOLDER = pjoin(cwd, "saved_models")

# Dataset consts
SAMPLES_NUM = 500
TRAIN_SPLIT = 0.8

# Training consts
EPOCHS = 300
BATCH_SIZE = 8

# Preparing GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def load_datasets() -> np.ndarray:
    i1 = np.load(pjoin(DATA_FOLDER, "input1.npy"))
    i2 = np.load(pjoin(DATA_FOLDER, "input2.npy"))
    tg = np.load(pjoin(DATA_FOLDER, "target.npy"))
    return i1, i2, tg

def load_tokenizer() -> Tokenizer:
    path = pjoin(DATA_FOLDER, "tokenizer.pickle")
    if os.path.exists(path):
        with open(path, "rb") as f:
            tokenizer = pickle.load(f)
    return tokenizer

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
        raise AttributeError("split_dataset: head_count is more than dataset len")
    
    if train_split <= 0.0 or train_split >= 1.0:
        raise AttributeError("split_dataset: train_split must be between 0.0 and 1.0")

    processed_dataset = dataset
    # Taking head count
    if head_count > 0:
        processed_dataset = processed_dataset[:head_count]

    train_count = int(len(processed_dataset) * train_split)

    train_dataset = processed_dataset[:train_count]
    test_dataset = processed_dataset[train_count:]
    
    return train_dataset, test_dataset

def main_train() -> None:
    tf.keras.backend.clear_session()
    model, encoder, decoder = build_lstm(rnn_units=128, dense_units=512, vocab_size=20000)

    i1, i2, tg = load_datasets()
    tkn = load_tokenizer()

    i1_train, i1_test = split_dataset(i1, head_count=SAMPLES_NUM, train_split=TRAIN_SPLIT)
    i2_train, i2_test = split_dataset(i2, head_count=SAMPLES_NUM, train_split=TRAIN_SPLIT)
    tg_train, tg_test = split_dataset(tg, head_count=SAMPLES_NUM, train_split=TRAIN_SPLIT)

    model.fit(x=[i1_train, i2_train], y=tg_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

    model.save(pjoin(SAVED_MODELS_FOLDER, "composite"))
    encoder.save(pjoin(SAVED_MODELS_FOLDER, "encoder"))
    decoder.save(pjoin(SAVED_MODELS_FOLDER, "decoder"))


if __name__ == "__main__":
    main_train()