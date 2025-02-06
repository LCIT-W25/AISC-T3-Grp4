import unittest
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

# Define text preprocessing pipeline
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Full text processing pipeline: Lowercase → Remove Punctuation → Remove Digits → Tokenization → Lemmatization"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Integration test class
class TestTextProcessingIntegration(unittest.TestCase):

    def test_full_pipeline(self):
        input_text = "Running Jumps in 2024!"
        expected_output = "running jump"  # Stopwords removed, lemmatized
        
        processed_text = preprocess_text(input_text)
        self.assertEqual(processed_text, expected_output)

if __name__ == '__main__':
    unittest.main()
