import re
import nltk
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        text = str(text).lower()
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def segment_sentences(text: str) -> list[str]:
        cleaned_text = TextPreprocessor.clean_text(text)
        return sent_tokenize(cleaned_text)
