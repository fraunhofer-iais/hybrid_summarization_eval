import string
from typing import List, Optional

from nltk import SnowballStemmer, word_tokenize
from nltk.corpus import stopwords

from konvens2020_summarization.data_classes import Corpus


class Preprocessor:
    def __init__(self,
                 pipeline: Optional[List] = None):
        self.pipeline = pipeline

    @staticmethod
    def remove_umlaut(text: str) -> str:
        text = text.replace('ä', 'ae')
        text = text.replace('ö', 'oe')
        text = text.replace('ü', 'ue')
        text = text.replace('Ä', 'Ae')
        text = text.replace('Ö', 'Oe')
        text = text.replace('Ü', 'Ue')
        text = text.replace('ß', 'ss')
        return text

    @staticmethod
    def to_lower(text: str) -> str:
        return text.lower()

    @staticmethod
    def remove_punctuation(text: str) -> str:
        return text.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def remove_digits(text: str) -> str:
        return text.translate(str.maketrans('', '', string.digits))

    @staticmethod
    def stem(text: str) -> str:
        stemmer = SnowballStemmer(language='german')
        return ' '.join([stemmer.stem(word) for word in text.split()])

    @staticmethod
    def remove_stopwords(text: str) -> str:
        # Load german stopwords
        try:
            german_stop_words = stopwords.words('german')
        except LookupError:
            import nltk
            nltk.download('stopwords')
            german_stop_words = stopwords.words('german')

        return ' '.join([token for token in word_tokenize(text) if token.lower() not in german_stop_words])

    @staticmethod
    def remove_multi_spaces(text: str) -> str:
        return ' '.join(text.split())

    def process_text(self, text: str) -> str:
        if isinstance(self.pipeline, List):
            for step in self.pipeline:
                text = getattr(Preprocessor, step)(text)
        return text

    def process(self,
                corpus: Corpus) -> Corpus:
        for doc in corpus.documents:
            doc.text_processed = self.process_text(doc.text)
            doc.ref_summary_processed = self.process_text(doc.ref_summary)
            for gen_sum in doc.gen_summaries:
                gen_sum.text_processed = self.process_text(gen_sum.text)
        return corpus
