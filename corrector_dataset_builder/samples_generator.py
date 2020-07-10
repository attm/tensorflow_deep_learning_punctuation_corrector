from corrector_dataset_builder.sentence_processors import pad_symbol, remove_multiple_spaces, add_before_uppercase
from utils.check_decorators import type_check
from config import dataset_builder_config

import re

PUNCTUATION_SYMBOLS = dataset_builder_config.DATASET_PUNCTUATION_SYMBOLS
START_TOKEN = dataset_builder_config.DATASET_START_TOKEN
END_TOKEN = dataset_builder_config.DATASET_END_TOKEN
NUM_TOKEN = dataset_builder_config.DATASET_NUM_TOKEN
UPPERCASE_TOKEN = dataset_builder_config.DATASET_UPPERCASE_TOKEN


# ============ SENTENCE_PROCESSING ============ #
@type_check
def pad_sentence(sentence : str, pad_symbols : list = PUNCTUATION_SYMBOLS) -> str:
    """
    Padds all punctuations symbols.

    Parameters:
        sentence (str) : sentence to be processed.
        pad_symbols (list) : list of symbols that will be padded.
    Returns:
        Processed sentence.
    """
    for symbol in pad_symbols:
        sentence = pad_symbol(sentence, symbol)
    sentence = sentence.rstrip()
    sentence = remove_multiple_spaces(sentence)
    return sentence

@type_check
def remove_punctuation_symbols(sentence : str, punctuation_symbols : list = PUNCTUATION_SYMBOLS) -> str:
    """
    Removes all punctuation symbols, by replacing with single space.

    Parameters:
        sentence (str) : sentence to be processed.
        punctuation_symbols (list) : list of symbols that will be removed.
    Returns:
        Processed sentence.
    """
    for symbol in punctuation_symbols:
        sentence = sentence.replace(symbol, " ")
    sentence = sentence.rstrip()
    sentence = remove_multiple_spaces(sentence)
    return sentence

# ============ SAMPLES GENERATING ============ #
@type_check
def generate_samples(sentence : str) -> str:
    """
    Generates 4 samples from raw sentence: Inpu1, Input2, Target, Tokenizer.

    Parameters: 
        sentence (str) : raw original sentence.
    Returns:
        input1_sentence (str) : input1-like sentence, lowercase with mistakes
        input2_sentence (str) : input2-like sentence, correct with start token
        target_sentence (str) : target-like sentence, correct with end token
        tokenizer_sentence (str) : combined sentence, with all lower and upper words and start and end tokens
    """
    input1_sentence = generate_input1_sample(sentence)
    input2_sentence = generate_input2_sample(sentence)
    target_sentence = generate_target_sample(sentence)
    tokenizer_sentence = generate_tokenizer_sample(input1_sentence, target_sentence)
    return input1_sentence, input2_sentence, target_sentence, tokenizer_sentence

@type_check
def generate_input1_sample(sentence : str, punctuation_symbols : list = PUNCTUATION_SYMBOLS) -> str:
    """
    Generates input1-like sentence, by making sentence lowercase, removing all punctuation.
    Input1 sentences are just punctuatinally incorrect sentences, that should be corrected.

    Parameters:
        sentence (str) : original, correct sentence.
        punctuation_symbols (list) : list of symbols that will be removed.
    Returns:
        Input1 sample.
    """
    sentence = sentence.lower()
    sentence = remove_punctuation_symbols(sentence, punctuation_symbols)
    sentence.lstrip()
    return sentence

@type_check
def generate_input2_sample(sentence : str, start_token : str = START_TOKEN, uppercase_token : str = UPPERCASE_TOKEN, pad_symbols : list = PUNCTUATION_SYMBOLS) -> str:
    """
    Generates input2-like sentence, by padding it and adding start token at the start.
    Input2-like sentences are correct sentences that will be used for teacher forcing.

    Parameters:
        sentence (str) : original, correct sentence.
        start_token (str) : start token, will be used for tokenizer and training.
        pad_symbols (list) : list of symbols that will be padded.
    Returns:
        Input2 sample.
    """
    sentence = pad_sentence(sentence, pad_symbols=pad_symbols)
    sentence = add_before_uppercase(sentence, add_str=uppercase_token + " ")
    sentence = start_token + " " + sentence
    sentence = sentence.lower()
    sentence.lstrip()
    return sentence

@type_check
def generate_target_sample(sentence : str, end_token : str = END_TOKEN, uppercase_token : str = UPPERCASE_TOKEN, pad_symbols : list = PUNCTUATION_SYMBOLS) -> str:
    """
    Generates target-like sentence, by padding it and adding end token.
    Target sentences are correct sentences that will be used for model fitting.

    Parametes:
        sentence (str) : original, correct sentence.
        end_token (str) : end token, will be used for tokenizer and training.
        pad_symbols (list) : list of symbols that will be padded.
    Returns:
        Target sample.
    """
    sentence = pad_sentence(sentence, pad_symbols=pad_symbols)
    sentence = add_before_uppercase(sentence, add_str=uppercase_token + " ")
    sentence = sentence + " " + end_token
    sentence = sentence.lower()
    sentence.lstrip()
    return sentence

@type_check
def generate_tokenizer_sample(target_sentence : str, start_token : str = START_TOKEN) -> str:
    """
    Generates sample for tokenizer fitting by simply adding start token for target sentence, so tokenizer will ger all symbols.

    Parameters:
        target_sentence (str) : Target sentence, correct with end token.
        start_token (str) : Start token that will be added.
    Returns:
        Combined sample for tokenizer fitting.
    """
    tokenizer_sentence = target_sentence + " " + start_token
    return tokenizer_sentence