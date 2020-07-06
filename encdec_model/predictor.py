from tensorflow.keras.preprocessing.text  import Tokenizer
from corrector_dataset_builder.samples_generator import generate_input1_sample
import numpy as np


class EncDecPredictor():
    def __init__(self):
        pass

    # ============ PUBLIC ============ #
    def set_tokenizer(self, tokenizer : Tokenizer) -> None:
        pass

    def predict(self, sentence : str) -> str:
        pass

    def translate_sentence(self, sentence : np.ndarray) -> str:
        pass

    # ============ PRIVATE ============ #
    def _predict_sentence(self, sentence : str) -> str:
        pass

    def _process_sentence(self, sentence : str) -> str:
        pass