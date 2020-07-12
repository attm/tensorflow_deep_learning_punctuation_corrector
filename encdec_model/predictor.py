from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model
import numpy as np
import tensorflow as tf
from corrector_dataset_builder.samples_generator import generate_input1_sample
from corrector_dataset_builder.dataset_builder import pad_sample_dataset
from corrector_dataset_builder.sentence_processors import token_to_uppercase
from utils.check_decorators import type_check
from config import dataset_builder_config


UPPERCASE_TOKEN = dataset_builder_config.DATASET_UPPERCASE_TOKEN


class EncDecPredictor():
    def __init__(   self,
                    encoder : Model,
                    decoder : Model,
                    tokenizer : Tokenizer,
                    pad_maxlen : int = 32,
                    pad_padding : str = "post",
                    start_token : str = "<s>",
                    end_token : str = "<e>"):
        # Checking and setting components
        if not isinstance(encoder, Model):
            raise TypeError("Can't set encoder, expected tf.keras Model, got {0}".format(type(encoder)))
        else:
            self.encoder = encoder

        if not isinstance(decoder, Model):
            raise TypeError("Can't set decoder, expected tf.keras Model, got {0}".format(type(decoder)))
        else:
            self.decoder = decoder

        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("Can't set tokenizer, expected tf.keras Tokenizer, got {0}".format(type(tokenizer)))
        else:
            self.tokenizer = tokenizer
            self.inv_word_index = {v: k for k, v in self.tokenizer.word_index.items()}

        if not isinstance(pad_maxlen, int) or pad_maxlen <= 0:
            raise TypeError("Can't set pad_maxlen, must be >0 and type of int, got {0} type of {1}".format(pad_maxlen, type(pad_maxlen)))
        else:
            self.pad_maxlen = pad_maxlen

        if not isinstance(pad_padding, str):
            raise TypeError("Can't set pad_padding, must be str, got {0}".format(type(pad_padding)))
        else:
            self.pad_padding = pad_padding

        if not isinstance(start_token, str):
            raise TypeError("Can't set start_token, must be str, got {0}".format(type(start_token)))
        else:
            self.start_token = start_token

        if not isinstance(end_token, str):
            raise TypeError("Can't set end_token, must be str, got {0}".format(type(end_token)))
        else:
            self.end_token = end_token 

    # ============ PUBLIC ============ #
    def predict(self, sentence : str) -> str:
        sequence = self._sentence_to_sequence(sentence)
        prediction_sequence = self._predict_sequence_keras(sequence)
        prediction_sentence = self.translate_sequence(prediction_sequence)
        return prediction_sentence

    def translate_sequence(self, sequence : list) -> str:
        # translating tokenized seq (list of ints) to string
        # words = [self.inv_word_index.get(word) for word in sequence]
        # listToStr = ' '.join([str(elem) for elem in words if elem is not None])
        # return listToStr
        translated_sentence = self.tokenizer.sequences_to_texts([sequence])
        translated_sentence = "".join([str(word) for word in translated_sentence if word is not None])
        translated_sentence = token_to_uppercase(translated_sentence, UPPERCASE_TOKEN + " ")
        return translated_sentence

    # ============ PRIVATE ============ #
    def _predict_sequence_keras(self, sequence : np.ndarray) -> str:
        # Getting tokens numbers in word_index
        start_token = self.tokenizer.word_index[self.start_token]
        end_token = self.tokenizer.word_index[self.end_token]
        
        # Encode the input as state vectors.
        states_value, encoder_outputs = self.encoder.predict(sequence)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, 1))
        # Populate the first character of target sequence with the start token.
        target_seq[0, 0, 0] = start_token

        output = []
        for i in range(self.pad_maxlen):
            # Predicting sequnece, getting prediction (dense softmax output) and states
            # Inputs are sequence, states from encoder or from previous prediction, and encoder_outputs for attention
            output_tokens, h, c = self.decoder.predict([target_seq, states_value, encoder_outputs])
            # Getting word token from dense softmax output
            word_token = np.argmax(output_tokens[0, 0, :])
            # Break if sequnece end
            if word_token == end_token:
                break
            # Saving predicted word
            output.append(word_token)
            # Remaking sequnece
            target_seq = np.zeros((1, 1, 1))
            target_seq[0, 0, 0] = word_token
            # Saving state for next iteration
            states_value = [h, c]

        return output

    def _sentence_to_sequence(self, sentence : str) -> np.ndarray:
        sentence = generate_input1_sample(sentence)
        sentence = self.tokenizer.texts_to_sequences([sentence])
        sequence = pad_sample_dataset(sentence)
        return sequence
