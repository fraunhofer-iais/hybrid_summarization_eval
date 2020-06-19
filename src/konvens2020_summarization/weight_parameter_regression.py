from typing import List, Dict, Union, Tuple

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from konvens2020_summarization.data_classes import Corpus, default_corpus


def _get_values(corpus: Corpus,
                annotation_type: str = 'total_score',
                prediction_types: Union[List[str], str] = 'all'):
    assert prediction_types == 'all' or isinstance(prediction_types, list)
    summaries = [s for doc in corpus for s in doc.gen_summaries]

    try:
        y = [s.annotated_scores[annotation_type] for s in summaries]
    except KeyError:
        print(f'ERROR: Annotation type {annotation_type} not found in summaries.')
        for s in summaries:
            if not 'total_score' in s.annotated_scores:
                print(s.keys)
        raise
    y = np.array(y)

    x = []
    for s in summaries:
        if prediction_types != 'all':
            try:
                assert all([t in s.predicted_scores.keys() for t in prediction_types])
            except AssertionError:
                print(f'ERROR: summary {s.id_} does not contain all specified predictions.\nSpecified: {prediction_types}.\nPredicted: {list(s.predicted_scores.keys())}')
                raise
        x.append([value for key, value in s.predicted_scores.items() if
                  prediction_types == 'all' or key in prediction_types])

    x = np.array(x)
    return x, y


# def _calculate_linear_regression(x, y):
#     model = LinearRegression()
#     regression_model = model.fit(x, y)
#     y_pred = regression_model.predict(x)
#     return regression_model, y_pred


def _calculate_polynomial_regression(x, y, n=2):
    poly_reg = PolynomialFeatures(degree=n)
    x_poly = poly_reg.fit_transform(x)
    regression_model = LinearRegression()
    regression_model.fit(x_poly, y)
    y_pred = regression_model.predict(x_poly)
    return regression_model, y_pred


# class LogitRegression(LinearRegression):
#
#     def fit(self, x, y, eps=1e-6):
#         y = np.maximum(y, eps)
#         y = np.minimum(y, 1 - eps)
#         y = np.asarray(y)
#         y = np.log(y / (1 - y))
#         return super().fit(x, y)
#
#     def predict(self, x):
#         y = super().predict(x)
#         return 1 / (np.exp(-y) + 1)
#
#
# def _calculate_logit_regression(x, y):
#     model = LogitRegression()
#     regression_model = model.fit(x, y)
#     y_pred = regression_model.predict(x)
#     return regression_model, y_pred


def apply_linear_regression(corpus: Corpus,
                            annotation_type: str = 'total_score',
                            prediction_types: Union[List[str], str] = 'all'):
    x, y = _get_values(corpus, annotation_type, prediction_types)

    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    result = model.fit()
    y_pred = result.predict().tolist()
    for doc in corpus:
        for s in doc.gen_summaries:
            s.output_score = y_pred.pop(0)
    return corpus, result


# def apply_linear_regression(corpus: Corpus,
#                             annotation_type: str = 'total_score',
#                             prediction_types: Union[List[str], str] = 'all',
#                             return_model: bool = False):
#     x, y = _get_values(corpus, annotation_type, prediction_types)
#     regression_model, y_pred = _calculate_linear_regression(x, y)
#     y_pred = list(y_pred)
#     for doc in corpus:
#         for s in doc.gen_summaries:
#             s.output_score = y_pred.pop(0)
#     if return_model:
#         return corpus, regression_model
#     return corpus


def apply_polynomial_regression(corpus: Corpus,
                                n: int = 2,
                                annotation_type: str = 'total_score',
                                prediction_types: Union[List[str], str] = 'all',
                                return_model: bool = False):
    x, y = _get_values(corpus, annotation_type, prediction_types)
    regression_model, y_pred = _calculate_polynomial_regression(x, y, n)
    y_pred = list(y_pred)
    for doc in corpus:
        for s in doc.gen_summaries:
            s.output_score = y_pred.pop(0)
    if return_model:
        return corpus, regression_model
    return corpus


def apply_logit_regression(corpus: Corpus,
                           annotation_type: str = 'total_score',
                           prediction_types: Union[List[str], str] = 'all'):
    x, y = _get_values(corpus, annotation_type, prediction_types)

    x = sm.add_constant(x)
    model = sm.Logit(y, x)
    result = model.fit()
    y_pred = result.predict().tolist()
    for doc in corpus:
        for s in doc.gen_summaries:
            s.output_score = y_pred.pop(0)
    return corpus, result


# def apply_logit_regression(corpus: Corpus,
#                            annotation_type: str = 'total_score',
#                            prediction_types: Union[List[str], str] = 'all',
#                            return_model: bool = False):
#     x, y = _get_values(corpus, annotation_type, prediction_types)
#     regression_model, y_pred = _calculate_logit_regression(x, y)
#     y_pred = list(y_pred)
#     for doc in corpus:
#         for s in doc.gen_summaries:
#             s.output_score = y_pred.pop(0)
#     if return_model:
#         return corpus, regression_model
#     return corpus


def main():
    corpus = default_corpus()
    prediction_types = ['first', 'second']
    for doc in corpus:
        for s in doc.gen_summaries:
            for t in prediction_types:
                s.predicted_scores[t] = np.random.rand()

    x, y = _get_values(corpus, annotation_type='total_score', prediction_types='all')
    print(x.shape)
    regression_model, y_pred = _calculate_linear_regression(x, y)
    print(np.mean(y-y_pred))
    regression_model, y_pred = _calculate_logit_regression(x, y)
    print(np.mean(y-y_pred))

    x, y = _get_values(corpus, annotation_type='total_score', prediction_types=['first', 'second'])
    print(x.shape)
    x, y = _get_values(corpus, annotation_type='total_score', prediction_types=['first'])
    print(x.shape)
    try:
        x, y = _get_values(corpus, annotation_type='total_score', prediction_types=['first', 'third'])
        print(x.shape)
    except AssertionError:
        print('correct error raised')


if __name__ == '__main__':
    main()
