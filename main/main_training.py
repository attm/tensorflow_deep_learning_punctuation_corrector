from encdec_model_builder.lstm_simple_builder import build as build_lstm
from encdec_model.predictor import EncDecPredictor
from tensorflow.keras.preprocessing.text import Tokenizer
from os.path import join as pjoin
from config import training_config, dataset_builder_config

import os 
import pickle

import tensorflow as tf
import numpy as np


cwd = os.getcwd()
DATA_FOLDER = pjoin(cwd, "data")
SAVED_MODELS_FOLDER = pjoin(cwd, "saved_models")
COMPLETE_MODELS_FOLDER = pjoin(SAVED_MODELS_FOLDER, "complete_models")
COMPOSITE_MODEL_CHECKPOINT_PATH = pjoin(SAVED_MODELS_FOLDER, "model_checkpoints", "composite_model", "checkpoint.ckpt")

# Dataset consts
SAMPLES_NUM = 20000
TRAIN_SPLIT = 0.8

# Training consts
EPOCHS = 5
BATCH_SIZE = 64
USE_CHECKPOINT = True

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
    model, encoder, decoder = build_lstm(rnn_units=256, dense_units=512, vocab_size=dataset_builder_config.TOKENIZER_NUM_WORDS)

    print(model.summary())
    print(encoder.summary())
    print(decoder.summary())

    i1, i2, tg = load_datasets()
    tkn = load_tokenizer()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=COMPOSITE_MODEL_CHECKPOINT_PATH,
                                                    save_best_only=True,
                                                    save_weights_only=True)

    callbacks=[cp_callback]

    if USE_CHECKPOINT:
        cp_dirname = os.path.dirname(COMPOSITE_MODEL_CHECKPOINT_PATH)
        if os.path.exists(cp_dirname):
            model.load_weights(COMPOSITE_MODEL_CHECKPOINT_PATH)
            print("Loaded checkpoint from {0}".format(COMPOSITE_MODEL_CHECKPOINT_PATH))
        else:
            print("Checkpoint path not exists.")

    model.fit(  x=[i1[:SAMPLES_NUM], i2[:SAMPLES_NUM]],
                y=tg[:SAMPLES_NUM], 
                batch_size=BATCH_SIZE, 
                epochs=EPOCHS,
                validation_split=0.2, 
                callbacks=callbacks)

    encoder.save(pjoin(COMPLETE_MODELS_FOLDER, "encoder"))
    decoder.save(pjoin(COMPLETE_MODELS_FOLDER, "decoder"))


if __name__ == "__main__":
    main_train()