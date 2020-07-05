import pytest
from corrector_dataset_builder.sentence_processors import pad_symbol, remove_multiple_spaces


# ============ PAD_SYMBOL TEST ============ #
def test_pad_symbol_bad_string():
    with pytest.raises(TypeError):
        pad_symbol("Test string", None) is None 

    with pytest.raises(TypeError):
        pad_symbol(None, 23) is None 

def test_pad_symbol():
    assert pad_symbol("Test str.", ".") == "Test str . "

# ============ REMOVE_MULTIPLE_SPACES TEST ============ #
def test_remove_multiple_spaces():
    assert remove_multiple_spaces("     Test     string   .   ") == " Test string . "
