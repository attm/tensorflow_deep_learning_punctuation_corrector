from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Bidirectional, LSTM, Dense
import tensorflow as tf 


def build(vocab_size : int = 20000, rnn_units : int = 256, dense_units : int = 1024, optimizer : object = None) -> Model:
    """
    Build enc-dec model with lstm rnn and attention.

    Parameters:
        vocab_size (int) : number of words for embedding, usually the same as tokenizer num_words
        rnn_units (int) : number of LSTM units
        dense_units (int) : number of dense units
        optimizer (tf.keras optimizer) : optimizer for model
    Returns:
        model (tf.keras.Model) : model for training
        encoder_inf_model (tf.keras.Model) : encoder for predicting
        decoder_inf_model (tf.keras.Model) : decoder for predicting

    """
    # LAYERS DEFINITION
    # Parameters definition

    # encoder & decoder inputs
    encoder_input = Input(shape=(None,))
    decoder_input = Input(shape=(None,))

    # Encoder & Decoder embedding layer (same for both)
    e_embedding = tf.keras.layers.Embedding(vocab_size, rnn_units, mask_zero=False)
    d_embedding = tf.keras.layers.Embedding(vocab_size, rnn_units, mask_zero=False)

    attention_layer = tf.keras.layers.AdditiveAttention

    # MODEL DEFINITION
    # encoder def
    encoder = e_embedding(encoder_input)
    e_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, return_state=True))(encoder)
    e_rnn = Bidirectional(LSTM(rnn_units, return_sequences=True, return_state=True))(e_rnn)
    e_rnn = Bidirectional(LSTM(rnn_units, return_state=True))(e_rnn)

    encoder_outs, f_encoder_h, f_encoder_c, b_encoder_h, b_encoder_c = e_rnn
    encoder_h = tf.keras.layers.concatenate([f_encoder_h, b_encoder_h])
    encoder_c = tf.keras.layers.concatenate([f_encoder_c, b_encoder_c])

    #decoder def
    decoder = d_embedding(decoder_input)
    d_rnn = LSTM(rnn_units * 2, return_sequences=True, return_state=True)(decoder, initial_state=[encoder_h, encoder_c])
    d_rnn = LSTM(rnn_units * 2, return_sequences=True, return_state=True)(d_rnn)
    d_rnn = LSTM(rnn_units * 2, return_sequences=True, return_state=True)(d_rnn)

    decoder_outs, _, _ = d_rnn

    #attention 
    attention = attention_layer()([decoder_outs, encoder_outs])
    context_combined = tf.keras.layers.concatenate([attention, decoder_outs])

    # decoder Dense output layer
    decoder_output = Dense(dense_units, activation="tanh")(context_combined)
    decoder_output = Dense(vocab_size, activation="softmax")(decoder_output)

    #encoder-decoder model
    model = tf.keras.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

    #encoder for inference
    encoder_inf_model = tf.keras.Model(encoder_input, outputs=[encoder_h, encoder_c])

    #decoder for inference
    decoder_inf_input_h = tf.keras.Input(shape=(rnn_units * 2))
    decoder_inf_input_c = tf.keras.Input(shape=(rnn_units * 2))
    decoder_inf_initial_states = [decoder_inf_input_h, decoder_inf_input_c]

    #decoder embedding
    decoder_emb = d_embedding(decoder_input)

    #decoder_inf 
    decoder_inf_outs, decoder_inf_h, decoder_inf_c = LSTM(rnn_units * 2, return_state=True, return_sequences=True)(decoder_emb, initial_state=decoder_inf_initial_states)
    decoder_inf_states = [decoder_inf_h, decoder_inf_c]

    #decoder_inf dense
    decoder_inf_output = tf.keras.layers.TimeDistributed(Dense(dense_units, activation="tanh"))(decoder_inf_outs)
    decoder_inf_output = tf.keras.layers.TimeDistributed(Dense(vocab_size, activation="softmax"))(decoder_inf_output)
    decoder_inf_model = tf.keras.Model([decoder_input] + decoder_inf_initial_states, [decoder_inf_output] + decoder_inf_states)

    if optimizer is None:
        print("lstm_atn_builder.build: optimizer is None, creating default.")
        optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model, encoder_inf_model, decoder_inf_model