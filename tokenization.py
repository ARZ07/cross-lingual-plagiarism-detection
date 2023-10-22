# tokenization.py

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def tokenize_text(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    max_seq_length = max(len(text) for text in texts)
    text_sequences = [tokenizer.texts_to_sequences([text])[0] for text in texts]
    padded_sequences = pad_sequences(text_sequences, maxlen=max_seq_length)
    return tokenizer, padded_sequences
