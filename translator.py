# translation.py

from googletrans import Translator

def translate_to_english(text):
    translator = Translator()
    translation = translator.translate(text, src='auto', dest='en')
    return translation.text
