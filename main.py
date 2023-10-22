# main.py

from src.translation import translate_to_english
from src.preprocessing import preprocess_text
from src.tokenization import tokenize_text
from src.model import create_siamese_model
import tensorflow as tf
import numpy as np

# Load and translate data
text1 = """Aazaad - Chrome.lnk" ... """  # Your text data
text2 = """Wings of Fire, une autobiographie ... """  # Your text data

text1 = translate_to_english(text1)
text2 = translate_to_english(text2)

# Preprocess and tokenize text data
text1 = preprocess_text(text1)
text2 = preprocess_text(text2)

texts = [text1, text2]
tokenizer, padded_sequences = tokenize_text(texts)

# Create and compile the Siamese model
input_shape = (padded_sequences.shape[1],)
embedding_dim = 50  # Adjust as needed
model = create_siamese_model(input_shape, embedding_dim)
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['mae'])

# Use the model for plagiarism detection
similarity = model.predict([padded_sequences[0], padded_sequences[1]])

# Define a threshold for plagiarism detection
threshold = 0.05  # Adjust as needed

# Check for plagiarism
if similarity > threshold:
    print("Plagiarism Detected.")
else:
    print("No Plagiarism Detected.")
