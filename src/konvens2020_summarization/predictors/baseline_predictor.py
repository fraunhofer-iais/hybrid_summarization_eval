import numpy as np

from konvens2020_summarization.data_classes import Corpus


def predict_baseline_score(corpus: Corpus, name: str, baseline_score: float):
    for d in corpus.documents:
        for s in d.gen_summaries:
            if s.predicted_scores is None:
                s.predicted_scores = {}
            s.predicted_scores[name] = baseline_score
    return corpus


def predict_random_score(corpus: Corpus, name: str):
    for d in corpus.documents:
        for s in d.gen_summaries:
            if s.predicted_scores is None:
                s.predicted_scores = {}
            s.predicted_scores[name] = np.random.rand()
    return corpus