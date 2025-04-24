import unittest
import pandas as pd
import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Functions from your main code
def to_lowercase(text):
    return text.lower()

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_digits(text):
    return re.sub(r'\d+', '', text)

lemmatizer = WordNetLemmatizer()
def apply_lemmatization(text):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(word) for word in tokens])

# Unit test class
class TestTextProcessing(unittest.TestCase):

    def test_to_lowercase(self):
        self.assertEqual(to_lowercase("Hello WORLD"), "hello world")

    def test_remove_punctuation(self):
        self.assertEqual(remove_punctuation("Hello, world!"), "Hello world")

    def test_remove_digits(self):
        self.assertEqual(remove_digits("This is 2024"), "This is ")

    def test_apply_lemmatization(self):
        self.assertEqual(apply_lemmatization("running jumps"), "running jump")

if __name__ == '__main__':
    unittest.main()
