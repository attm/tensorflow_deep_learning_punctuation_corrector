import pytest
from corrector_dataset_builder.samples_generator import *


TEST_SENTENCE = "This is simple: basic, test sentence. Maybe Correct?"
TEST_PUNCTUATION_SYMBOLS = [",", "."]


# ============ TEST SENTENCE PROCESSING ============ #
def test_pad_sentence():
    ps = pad_sentence(TEST_SENTENCE)
    assert ps == "This is simple : basic , test sentence . Maybe Correct ?"

def test_remove_punctuation_symbols():
    s = remove_punctuation_symbols(TEST_SENTENCE, TEST_PUNCTUATION_SYMBOLS)
    assert s == "This is simple: basic test sentence Maybe Correct?"

# ============ TEST SAMPLE GENERATING ============ #
def test_generate_samples():
    i1, i2, tg, tk = generate_samples(TEST_SENTENCE)
    assert i1 == "this is simple basic test sentence maybe correct"
    assert i2 == "<s> This is simple : basic , test sentence . Maybe Correct ?"
    assert tg == "This is simple : basic , test sentence . Maybe Correct ? <e>"
    # Hard to check, order is not always the same
    assert tk is not None

def test_generate_input1_sample():
    i1 = generate_input1_sample(TEST_SENTENCE, TEST_PUNCTUATION_SYMBOLS)
    assert i1 == "this is simple: basic test sentence maybe correct?"

def test_generate_input2_sample():
    i2 = generate_input2_sample(TEST_SENTENCE)
    assert i2 == "<s> This is simple : basic , test sentence . Maybe Correct ?"

def test_generate_target_sample():
    tg = generate_target_sample(TEST_SENTENCE)
    assert tg == "This is simple : basic , test sentence . Maybe Correct ? <e>"

def test_generate_tokenizer_sample():
    tk = generate_tokenizer_sample(TEST_SENTENCE, TEST_SENTENCE)
    tk is not None