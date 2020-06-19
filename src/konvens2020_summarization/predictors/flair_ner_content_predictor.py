from typing import Optional, Dict, Tuple, List

from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm

from konvens2020_summarization.data_classes import Corpus
from konvens2020_summarization.preprocessor import Preprocessor


def get_important_words(corpus: Corpus,
                        preprocess_pipeline: Optional[List] = None) -> Tuple[Dict, Corpus]:
    important_words = {}
    tagger = SequenceTagger.load('de-ner')
    preprocessor = Preprocessor(pipeline=preprocess_pipeline)
    corpus_processed = preprocessor.process(corpus=corpus)

    for doc in tqdm(corpus_processed.documents):
        sentence = Sentence(doc.text)
        tagger.predict(sentence)

        important_words[doc.id_] = [entity.text for entity in sentence.get_spans('ner')]

    return important_words, corpus_processed


def predict_flair_ner_content_score(corpus: Corpus,
                                    name: str,
                                    preprocessing: Optional[List] = None):

    important_words, corpus = get_important_words(corpus=corpus,
                                                  preprocess_pipeline=preprocessing)

    for doc in corpus.documents:
        important_doc_words = important_words[doc.id_]
        for gen_sum in doc.gen_summaries:
            gen_sum_counts = 0
            for imp_word in important_doc_words:
                if imp_word in gen_sum.text_processed.split(' '):
                    gen_sum_counts += 1

            try:
                gen_sum_score = gen_sum_counts / len(important_doc_words)
            except ZeroDivisionError:
                gen_sum_score = 0.

            gen_sum.predicted_scores[name] = gen_sum_score
    return corpus
