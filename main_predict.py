import tensorflow as tf
from encdec_model.predictor import EncDecPredictor
from os.path import join as pjoin
import os
import pickle


# Preparing GPU
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


cwd = os.getcwd()
SAVED_MODELS_FOLDER = pjoin(cwd, "saved_models")
ENCODER_MODEL_NAME = "encoder"
DECODER_MODEL_NAME = "decoder"
TOKENIZER_PATH = pjoin(cwd, "data", "tokenizer.pickle")

def load_models():
    #composite = tf.keras.models.load_model(pjoin(SAVED_MODELS_FOLDER, "composite"))
    encoder = tf.keras.models.load_model(pjoin(SAVED_MODELS_FOLDER, ENCODER_MODEL_NAME))
    decoder = tf.keras.models.load_model(pjoin(SAVED_MODELS_FOLDER, DECODER_MODEL_NAME))
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    return encoder, decoder, tokenizer

def main():
    tf.keras.backend.clear_session()
    enc, dec, tkn = load_models()

    predictor = EncDecPredictor(enc, dec, tkn)

    sentence = "i will show"

    print(predictor.predict(sentence))

if __name__ == "__main__":
    main()