from corrector_dataset_builder.samples_generator import generate_samples
from utils.check_decorators import type_check
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# TOKENIZER CONST
TOKENIZER_NUM_WORDS = 10000
TOKENIZER_OOV = "oov"
TOKENIZER_FILTERS = '"#$%&()*+-/=@[\\]^_{|}~\t\n'

# PAD SEQUENCES CONST
PAD_MAXLEN = 32
PAD_PADDING = "post"

@type_check
def build_from_sentences(sentences : list, tokenizer : Tokenizer = None, num_words : int = TOKENIZER_NUM_WORDS, pad_maxlen : int = PAD_MAXLEN, padding : str = PAD_PADDING) -> np.ndarray:
    """
    Building datasets for training from list of raw sentences.

    Parameters:
        sentences (list) : List of raw sentences.
        tokenizer (keras Tokenizer instance) : tokenizer that will be fitted and used for tokenizing.
        num_words (int) : maximum number of words that will be used for tokenizing
        pad_maxlen (int) : maxlen parameter for pad_sequence function
        padding (str) : padding parameter for pad_sequence function ("post" or "pred")
    Returns:
        input1_dataset (np.ndarray) : tokenizer and padded input1 dataset
        input2_dataset (np.ndarray) : tokenizer and padded input2 dataset
        target_dataset (np.ndarray) : tokenizer and padded target dataset
        tokenizer (keras Tokenizer) : fitted tokenizer
    """
    # Processing sentences and creating raw datasets.
    i1, i2, tg, tk = generate_samples_datasets_from_sentences(sentences)
    print("dataset_builder.build_from_sentences: built sample sentences.")

    # Checking if tokenizer is ok, making default one if not.
    if not isinstance(tokenizer, Tokenizer):
        tokenizer = Tokenizer(num_words=num_words, oov_token=TOKENIZER_OOV, filters=TOKENIZER_FILTERS)
        print("dataset_builder.build_from_sentences: got wrong tokenizer, created default.")

    # Fitting tokenizer on special dataset.
    tokenizer.fit_on_texts(tk)
    print("dataset_builder.build_from_sentences: tokenizer fitted on texts.")

    # Tokenizing all datasets.
    i1 = tokenize_sample_dataset(i1, tokenizer)
    i2 = tokenize_sample_dataset(i2, tokenizer)
    tg = tokenize_sample_dataset(tg, tokenizer)
    print("dataset_builder.build_from_sentences: tokenized sample sentences.")

    # Padding all datasets.
    i1 = pad_sample_dataset(i1, maxlen=pad_maxlen, padding=padding)
    i2 = pad_sample_dataset(i2, maxlen=pad_maxlen, padding=padding)
    tg = pad_sample_dataset(tg, maxlen=pad_maxlen, padding=padding)
    print("dataset_builder.build_from_sentences: padded sample sentences.")

    return i1, i2, tg, tokenizer


@type_check
def generate_samples_datasets_from_sentences(sentences : list) -> list:
    """
    Generating samples datasets from sentences.

    Parameters:
        sentences (list) : List of raw sentences
    Returns:
        input1_dataset, input2_dataset, target_dataset, tokenizer_dataset (list) : processed datasets.
    """
    input1_dataset = []
    input2_dataset = []
    target_dataset = []
    tokenizer_dataset = []

    for s in sentences:
        i1, i2, tg, tk = generate_samples(s)

        input1_dataset.append(i1)
        input2_dataset.append(i2)
        target_dataset.append(tg)
        tokenizer_dataset.append(tk)

    return input1_dataset, input2_dataset, target_dataset, tokenizer_dataset

@type_check
def tokenize_sample_dataset(samples_dataset : list, tokenizer : Tokenizer) -> list:
    """
    Tokenizing given list of texts.
    
    Parameters:
        samples_dataset (list) : list of str's that need to be tokenized
        tokenizer (keras Tokenizer) : tokenizer that will tokenize dataset
    Returns:
        tokenized_sample_dataset (list) : tokenized dataset.
    """
    tokenized_sample_dataset = tokenizer.texts_to_sequences(samples_dataset)
    return tokenized_sample_dataset

@type_check
def pad_sample_dataset(samples_dataset : list, maxlen : int = 32, padding : str = "post") -> list:
    """
    Padding given tokenized dataset.

    Parameters:
        sample_dataset (list) : dataset to be padded.
        maxlen (int) : pad_sequence maxlen parameter.
        padding (str) : pad_sequence padding parameter, "post" or "pred
    Returns:
        samples_dataset (np.ndarray) : padded dataset.
    """
    samples_dataset = pad_sequences(samples_dataset, maxlen=maxlen, padding=padding)
    return samples_dataset

