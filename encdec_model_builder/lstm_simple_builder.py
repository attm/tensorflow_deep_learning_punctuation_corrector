from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, TimeDistributed
import tensorflow as tf


def build(vocab_size : int = 20000, rnn_units : int = 256, dense_units : int = 1024, seq_lenght : int = 32) -> Model:
    # ============ ENCODER-TRAINING MODEL ============ #
    # Encoder inputs, seq_lenght is maximum lenght of tokenizerd & padded sentence
    encoder_inputs = Input(shape=(seq_lenght, ))

    # Embedding layer, inputs is seq of integers
    encoder_embedding = Embedding(vocab_size, rnn_units, input_length=seq_lenght, mask_zero=False)
    encoder = encoder_embedding(encoder_inputs)

    # Encoder LSTM, getting states and passing it to decoder LSTM
    encoder_lstm = LSTM(rnn_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder)
    # Saving encoder LSTM states
    encoder_states = [state_h, state_c]

    # ============ DECODER-TRAINING MODEL ============ #
    # Decoder inputs, same as encoder - seq of ints
    decoder_inputs = Input(shape=(seq_lenght, ))

    # Decoder embedding layer
    decoder_embedding = Embedding(vocab_size, rnn_units, input_length=seq_lenght, mask_zero=False)
    decoder = decoder_embedding(decoder_inputs)

    # Decoder LSTM
    decoder_lstm = LSTM(rnn_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder,
                                        initial_state=encoder_states)
    
    #Decoder Dense layers output
    decoder_dense_1 = Dense(dense_units, activation="tanh")
    decoder_dense_2 = Dense(vocab_size, activation='softmax')

    decoder = decoder_dense_1(decoder_outputs)
    decoder_outputs = decoder_dense_2(decoder)

    # ============ COMPOSITE ENCODER-DECODER MODEL FOR TRAINING ============ #
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # ============ ENCODER MODEL FOR INFERENCE ============ #
    encoder_model = Model(encoder_inputs, encoder_states)

    # ============ DECODER MODEL FOR INFERENCE ============ #
    # Inference Decoder input states for LSTM layer
    decoder_state_input_h = Input(shape=(rnn_units,))
    decoder_state_input_c = Input(shape=(rnn_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # Inference Decoder embedding
    decoder_model = decoder_embedding(decoder_inputs)

    # Inference Decoder LSTM
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_model, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    # Inference Decoder Dense                        
    decoder_outputs = decoder_dense_1(decoder_outputs)
    decoder_outputs = decoder_dense_2(decoder_outputs)
    
    decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
    return model, encoder_model, decoder_model