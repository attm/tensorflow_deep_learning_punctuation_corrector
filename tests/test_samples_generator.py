import pytest
from corrector_dataset_builder.samples_generator import *
from config import dataset_builder_config


TEST_SENTENCE = "This is simple: basic, test sentence. Maybe Correct?"
TEST_PUNCTUATION_SYMBOLS = [",", "."]

START_TOKEN = dataset_builder_config.DATASET_START_TOKEN
END_TOKEN = dataset_builder_config.DATASET_END_TOKEN
UPPERCASE_TOKEN = dataset_builder_config.DATASET_UPPERCASE_TOKEN


# ============ TEST SENTENCE PROCESSING ============ #
def test_pad_sentence():
    ps = pad_sentence(TEST_SENTENCE)
    assert ps == "This is simple : basic , test sentence . Maybe Correct ?"

def test_remove_punctuation_symbols():
    s = remove_punctuation_symbols(TEST_SENTENCE, TEST_PUNCTUATION_SYMBOLS)
    assert s == "This is simple: basic test sentence Maybe Correct?"

# ============ TEST SAMPLE GENERATING ============ #

def test_generate_input1_sample():
    i1 = generate_input1_sample(TEST_SENTENCE, TEST_PUNCTUATION_SYMBOLS)
    assert i1 == "this is simple: basic test sentence maybe correct?"

def test_generate_input2_sample():
    i2 = generate_input2_sample(TEST_SENTENCE)
    assert i2 == "{0} {1} this is simple : basic , test sentence . {2} maybe {3} correct ?".format(START_TOKEN, UPPERCASE_TOKEN, UPPERCASE_TOKEN, UPPERCASE_TOKEN)

def test_generate_target_sample():
    tg = generate_target_sample(TEST_SENTENCE)
    assert tg == "{0} this is simple : basic , test sentence . {1} maybe {2} correct ? {3}".format(UPPERCASE_TOKEN, UPPERCASE_TOKEN, UPPERCASE_TOKEN, END_TOKEN)

def test_generate_tokenizer_sample():
    tk = generate_tokenizer_sample(TEST_SENTENCE, TEST_SENTENCE)
    tk is not None