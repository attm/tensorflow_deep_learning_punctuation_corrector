from corrector_dataset_builder.sentence_processors import pad_symbol, remove_multiple_spaces
from utils.check_decorators import type_check


@type_check
def pad_sentence(sentence : str) -> str:
    pass

@type_check
def generate_punctuation_mistakes(sentence : str) -> str:
    pass

@type_check
def generate_samples(sentence : str) -> str:
    pass

@type_check
def generate_input1_sample(sentence : str) -> str:
    pass

@type_check
def generate_input2_sample(sentence : str) -> str:
    pass

@type_check
def generate_target_sample(sentence : str) -> str:
    pass

@type_check
def generate_tokenizer_sample(sentence : str) -> str:
    pass