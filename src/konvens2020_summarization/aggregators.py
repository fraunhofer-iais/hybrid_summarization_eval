from importlib import import_module
from typing import Union, List, Optional

import numpy as np

from konvens2020_summarization.data_classes import Corpus, default_corpus
from konvens2020_summarization.weight_parameter_regression import apply_linear_regression, apply_logit_regression, apply_polynomial_regression

AGGREGATORS = {'agg_min': ('konvens2020_summarization.aggregators',
                           'agg_min'),
               'agg_weighted': ('konvens2020_summarization.aggregators',
                                'agg_weighted'),
               'agg_linear_reg': ('konvens2020_summarization.aggregators',
                                  'agg_linear_reg'),
               'agg_logit_reg': ('konvens2020_summarization.aggregators',
                                 'agg_logit_reg'),
               'agg_poly_reg': ('konvens2020_summarization.aggregators',
                                'agg_poly_reg')}


def init_aggregator(name: str):
    try:
        module_name, method_name = AGGREGATORS[name]
    except KeyError:
        raise KeyError(f'Aggregator {name} is not implemented. Choose among the following aggregators:\n'
                       f'{list(AGGREGATORS.keys())}')

    module = import_module(module_name)
    aggregator = getattr(module, method_name)
    return aggregator


def agg_min(corpus: Corpus):
    for doc in corpus.documents:
        for gen_sum in doc.gen_summaries:
            predicted_scores = list(gen_sum.predicted_scores.values())
            gen_sum.output_score = min(predicted_scores)
    return corpus


def agg_weighted(corpus: Corpus,
                 weights: Optional[List[float]] = None,
                 prediction_types: Union[str, List[str]] = 'all'):
    if prediction_types == 'all':
        prediction_types = list(corpus[0].gen_summaries[0].predicted_scores.keys())
        for doc in corpus:
            for s in doc.gen_summaries:
                assert list(s.predicted_scores.keys()) == prediction_types
    else:
        for doc in corpus:
            for s in doc.gen_summaries:
                assert all([t in s.predicted_scores.keys() for t in prediction_types])

    if weights is None:
        weights = np.array(1/len(prediction_types)).repeat(len(prediction_types))
    else:
        weights = np.array(weights)
    assert len(weights) == len(prediction_types)

    for doc in corpus:
        for s in doc.gen_summaries:
            scores = [value for key, value in s.predicted_scores.items() if key in prediction_types]
            s.output_score = np.dot(weights, np.array(scores))
    return corpus


def agg_linear_reg(corpus: Corpus,
                   annotation_type: str = 'total_score',
                   prediction_types: Union[str, List[str]] = 'all'):
    corpus, result = apply_linear_regression(corpus, annotation_type, prediction_types)
    return corpus, result


def agg_logit_reg(corpus: Corpus,
                  annotation_type: str = 'total_score',
                  prediction_types: Union[str, List[str]] = 'all'):
    corpus, result = apply_logit_regression(corpus, annotation_type, prediction_types)
    return corpus, result


def agg_poly_reg(corpus: Corpus,
                 annotation_type: str = 'total_score',
                 prediction_types: Union[str, List[str]] = 'all'):
    corpus, _ = apply_polynomial_regression(corpus, annotation_type=annotation_type, prediction_types=prediction_types)
    return corpus, _


def main():
    corpus = default_corpus()
    prediction_types = ['first', 'second']
    for doc in corpus:
        for s in doc.gen_summaries:
            for t in prediction_types:
                s.predicted_scores[t] = np.random.rand()

    corpus = agg_weighted(corpus, weights=None, prediction_types=['first', 'second'])
    print(np.mean([s.output_score for doc in corpus for s in doc.gen_summaries]))
    corpus = agg_weighted(corpus, weights=None, prediction_types=['first'])
    print(np.mean([s.output_score for doc in corpus for s in doc.gen_summaries]))
    try:
        corpus = agg_weighted(corpus, weights=None, prediction_types=['third'])
    except AssertionError:
        print('exception raised')
    corpus = agg_weighted(corpus, weights=[0.1, 0.4], prediction_types='all')
    print(np.mean([s.output_score for doc in corpus for s in doc.gen_summaries]))


if __name__ == '__main__':
    main()
