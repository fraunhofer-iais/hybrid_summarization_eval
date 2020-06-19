from typing import Optional, Dict, Tuple, List

import numpy as np

from konvens2020_summarization.data_classes import Corpus
from konvens2020_summarization.featurizer import TfidfFeaturizer
from konvens2020_summarization.preprocessor import Preprocessor


def get_important_words(corpus: Corpus,
                        k: Optional[int] = 10,
                        tfidf_args: Optional[Dict] = None,
                        preprocess_pipeline: Optional[List] = None) -> Tuple[Dict, Corpus]:

    if not tfidf_args:
        tfidf_args = {'max_df': 0.9}

    important_words = {}

    preprocessor = Preprocessor(pipeline=preprocess_pipeline)
    corpus_processed = preprocessor.process(corpus=corpus)

    featurizer = TfidfFeaturizer(train_documents=corpus_processed.documents,
                                 model_args=tfidf_args)
    corpus_featurized = featurizer.featurize(corpus=corpus_processed)

    feature_names = featurizer.feature_names

    for doc in corpus_featurized.documents:
        top_indices_text = np.argsort(doc.text_vectorized.flatten())[::-1][:k]
        top_indices_ref_sum = np.argsort(doc.ref_summary_vectorized.flatten())[::-1][:k]
        important_words[doc.id_] = {}

        important_words[doc.id_]['text'] = np.array(feature_names)[top_indices_text].tolist()
        important_words[doc.id_]['ref_sum'] = np.array(feature_names)[top_indices_ref_sum].tolist()
        important_words[doc.id_]['gen_sums'] = {}
        for gen_sum in doc.gen_summaries:
            top_indices_gen_sum = np.argsort(gen_sum.text_vectorized.flatten())[::-1][:k]
            important_words[doc.id_]['gen_sums'][gen_sum.id_] = np.array(feature_names)[top_indices_gen_sum].tolist()
    return important_words, corpus_processed


def predict_tfidf_content_reversed_score(corpus: Corpus,
                                         name: str,
                                         num_important_words: Optional[int] = 10,
                                         tfidf_args: Optional[Dict] = None,
                                         preprocessing: Optional[List] = None
                                         ):

    important_words, corpus = get_important_words(corpus=corpus,
                                                  k=num_important_words,
                                                  tfidf_args=tfidf_args,
                                                  preprocess_pipeline=preprocessing)

    # content_scores = {}
    for doc in corpus.documents:
        for gen_sum in doc.gen_summaries:
            gen_sum_counts = 0
            for imp_word in important_words[doc.id_]['gen_sums'][gen_sum.id_]:
                if imp_word in doc.ref_summary_processed.split(' '):
                    gen_sum_counts += 1

            try:
                gen_sum_score = gen_sum_counts / num_important_words
            except ZeroDivisionError:
                gen_sum_score = 0.

            gen_sum.predicted_scores[name] = gen_sum_score
    return corpus

