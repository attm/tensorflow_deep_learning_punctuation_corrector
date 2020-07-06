from corrector_dataset_builder.dataset_builder import build_from_sentences
import pandas as pd 
import numpy as np
from os.path import join as pjoin
import os
import pickle

# Path's setup
cwd = os.getcwd()
DATA_FOLDER = pjoin(cwd, "data")
RAW_DATA_FILE_PATH = pjoin(cwd, DATA_FOLDER, "eng_sentences.tsv")

def rebuild_datasets():
    # Loading raw sentences from .tsv
    df = pd.read_csv(RAW_DATA_FILE_PATH, delimiter="\t", error_bad_lines=False)
    texts = df.iloc[:, [2]].values
    texts = texts.reshape(texts.shape[0])
    texts_list = texts.tolist()

    # Building datasets
    i1, i2, tg, tokenizer = build_from_sentences(texts_list)

    # Saving datasets
    np.save(pjoin(DATA_FOLDER, "input1.npy"), i1)
    np.save(pjoin(DATA_FOLDER, "input2.npy"), i2)
    np.save(pjoin(DATA_FOLDER, "target.npy"), tg)

    # Saving tokenizer
    with open(pjoin(DATA_FOLDER, "tokenizer.pickle"), "wb") as f:
        pickle.dump(tokenizer, f)

    return i1, i2, tg, tokenizer

def main_dataset_build():
    i1, i2, tg, tokenizer = rebuild_datasets()

if __name__ == "__main__":
    main_dataset_build()