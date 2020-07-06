import pytest
from corrector_dataset_builder.dataset_builder import generate_samples_datasets_from_sentences, build_from_sentences
import numpy as np


TEST_SENTENCE = "This is simple: basic, test sentence. Maybe Correct?"
TEST_PUNCTUATION_SYMBOLS = [",", "."]

TEST_SENTENCE_LIST = [TEST_SENTENCE]


def test_generate_samples_datasets_from_sentences():
    i1, i2, tg, tk = generate_samples_datasets_from_sentences(TEST_SENTENCE_LIST)
    assert i1 == ["this is simple basic test sentence maybe correct"]
    assert i2 == ["<s> This is simple : basic , test sentence . Maybe Correct ?"]
    assert tg == ["This is simple : basic , test sentence . Maybe Correct ? <e>"]

def test_build_from_sentences():
    i1, i2, tg, tk = build_from_sentences(TEST_SENTENCE_LIST)
    assert isinstance(i1, np.ndarray)
    assert isinstance(i2, np.ndarray)
    assert isinstance(tg, np.ndarray)

def test_build_from_sentences_tokenizer_wrong_type():
    tkn = ["Fake Tokenizer"]
    i1, i2, tg, tk = build_from_sentences(TEST_SENTENCE_LIST, tokenizer=tkn)
