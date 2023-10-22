# model.py

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Lambda, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# Define Siamese network architecture
def create_siamese_network(input_shape, embedding_dim):
    input_layer = Input(shape=input_shape)
    embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=embedding_dim)(input_layer)
    lstm_layer = LSTM(128)(embedding_layer)
    return Model(inputs=input_layer, outputs=lstm_layer)

def create_siamese_model(input_shape, embedding_dim):
    input_source = Input(shape=input_shape)
    input_target = Input(shape=input_shape)

    siamese_network = create_siamese_network(input_shape, embedding_dim)

    output_source = siamese_network(input_source)
    output_target = siamese_network(input_target)

    # Define the similarity metric (e.g., cosine similarity)
    def similarity(vectors):
        x, y = vectors
        cosine = tf.keras.losses.cosine_similarity(x, y)
        return 1 - cosine

    similarity_layer = Lambda(similarity)([output_source, output_target])

    model = Model(inputs=[input_source, input_target], outputs=similarity_layer)
    return model
