import pytest
from corrector_dataset_builder.sentence_processors import pad_symbol, remove_multiple_spaces, add_before_uppercase, token_to_uppercase


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

# ============ ADD_BEFORE_UPPERCASE TEST ============ #
def test_add_before_uppercase():
    assert add_before_uppercase("This is Test", "<test> ") == "<test> This is <test> Test"

# ============ TOKEN_TO_UPPERCASE TEST ============ #
def test_token_to_uppercase():
    assert token_to_uppercase("<test> this is <test> test", "<test> ") == "This is Test"