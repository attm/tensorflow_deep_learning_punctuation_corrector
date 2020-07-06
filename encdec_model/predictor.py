from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras import Model
from corrector_dataset_builder.samples_generator import generate_input1_sample
from corrector_dataset_builder.dataset_builder import pad_sample_dataset
import numpy as np
from utils.check_decorators import type_check


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
        prediction_sequence = self._predict_sequence(sequence)
        prediction_sentence = self.translate_sequence(prediction_sequence)
        return prediction_sentence

    def translate_sequence(self, sequence : list) -> str:
        # translating tokenized seq (list of ints) to string
        # words = [self.inv_word_index.get(word) for word in sequence]
        # listToStr = ' '.join([str(elem) for elem in words if elem is not None])
        # return listToStr
        translated_sentence = self.tokenizer.sequences_to_texts([sequence])
        return translated_sentence

    # ============ PRIVATE ============ #
    def _predict_sequence(self, sequence : np.ndarray) -> str:
        # Getting tokens numbers in word_index
        start_token = self.tokenizer.word_index[self.start_token]
        end_token = self.tokenizer.word_index[self.end_token]
        # Reshaping sequence 
        seq = sequence.reshape(1, self.pad_maxlen)
        # Making first prediction, getting state
        state = self.encoder.predict(seq)
        # Making dummy sequnece
        target_seq = np.zeros(shape=(1, self.pad_maxlen))
        target_seq[0, [0]] = start_token
        # Defining output that contains predicted words
        output = []

        # Itterating and making predictions, saving last predicted words and use it for the next prediction
        for i in range(self.pad_maxlen):
            # Decoding and making prediction
            yh, h, c = self.decoder.predict([target_seq] + state)
            # Getting word from softmax dense output
            word_token = np.argmax(yh[0, 0, :])
            # If word is end_token, braking cycle
            if word_token == end_token:
                break
            # Appending output with new predicted word
            output.append(word_token)
            # Getting state
            state = [h, c]
            # Redefining new sequence and setting predicted word
            target_seq = np.zeros(shape=(1, 32))
            target_seq[0, [0]] = word_token
            print(word_token)

        print(output)
        return output

    def _sentence_to_sequence(self, sentence : str) -> np.ndarray:
        sentence = generate_input1_sample(sentence)
        sentence = self.tokenizer.texts_to_sequences([sentence])
        sequence = pad_sample_dataset(sentence)
        return sequence
