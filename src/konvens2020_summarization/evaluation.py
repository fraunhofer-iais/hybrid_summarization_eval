import json
import os
from typing import List, Dict, Tuple, Optional

import matplotlib.pyplot as plt
import scipy
import seaborn as sns
import numpy as np
import pandas as pd

from konvens2020_summarization.data_classes import Corpus


def collect_scores(corpus: Corpus,
                   predictor_names: List[str],
                   annotated_score_names: List[str]) -> Tuple[Dict[str, List],
                                                        Dict[str, List],
                                                        List]:

    annotated_scores = {name: [] for name in annotated_score_names}
    predicted_scores = {name: [] for name in predictor_names}
    output_scores = []

    for doc in corpus.documents:
        for gen_sum in doc.gen_summaries:
            for name in annotated_score_names:
                annotated_scores[name].append(gen_sum.annotated_scores[name])

            for name in predictor_names:
                predicted_scores[name].append(gen_sum.predicted_scores[name])

            output_scores.append(gen_sum.output_score)

    return annotated_scores, predicted_scores, output_scores


def evaluate(corpus: Corpus,
             predictor_names: List[str],
             save_dir: str,
             annotated_score_names: Optional[List[str]] = None):
    if annotated_score_names is None:
        annotated_score_names = ['total_score', 'content_score', 'grammar_score', 'compact_score', 'abstract_score']

    annotated_scores, predicted_scores, output_scores = collect_scores(corpus=corpus,
                                                                       predictor_names=predictor_names,
                                                                       annotated_score_names=annotated_score_names)

    print(f'\n'
          f'CORRELATION MATRIX OF PREDICTORS')
    pred_scores = np.array([pred for pred in predicted_scores.values()])
    correlation_matrix = pd.DataFrame(np.corrcoef(pred_scores), columns=predictor_names, index=predictor_names)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 1000)
    print(correlation_matrix)
    with open(os.path.join(save_dir, 'corr_matrix.tex'), 'w') as f:
        f.write(correlation_matrix.to_latex())
    with open(os.path.join(save_dir, 'corr_matrix.txt'), 'w') as f:
        f.write(correlation_matrix.to_string())

    # Plotting and Saving
    print(f'\n'
          f'CORRELATION RESULTS')

    results = {}

    message = ''
    for pred_score_name, pred_scores in predicted_scores.items():
        if pred_score_name not in results:
            results[pred_score_name] = {}
        message += f'{pred_score_name}\n'
        for anno_score_name, anno_scores in annotated_scores.items():
            if anno_score_name not in results[pred_score_name]:
                results[pred_score_name][anno_score_name] = []

            sns.jointplot(x=anno_scores,
                          y=pred_scores,
                          kind='hex',
                          space=0,
                          height=7,
                          ratio=2)

            pearson_r = scipy.stats.pearsonr(pred_scores, anno_scores)
            title = f"x={anno_score_name}\ny={pred_score_name}\nPearson's r={pearson_r[0]:.2f} p={pearson_r[1]:.2f}"
            plt.title(title, loc='left')
            plt.close('all')
            results[pred_score_name][anno_score_name].append(pearson_r[0])
            message += f'  {anno_score_name:<20} r: {pearson_r[0]:.4f}\tp: {pearson_r[1]:.4f}\n'
        message += '\n'

    sns.jointplot(x=annotated_scores['total_score'],
                  y=output_scores,
                  kind='hex',
                  space=0,
                  height=7,
                  ratio=2)

    json.dump(results, open(os.path.join(save_dir, 'results.json'), 'w'))
    pearson_r = scipy.stats.pearsonr(output_scores, annotated_scores['total_score'])
    title = f"x=output_score\ny=total_score\nPearson's r={pearson_r[0]:.2f} p={pearson_r[1]:.2f}"
    plt.title(title)
    plt.close('all')
    message += f'output_score\n'
    message += f'  {"total_score":<20} r: {pearson_r[0]:.4f}\tp: {pearson_r[1]:.4f}'

    print(message)
    with open(os.path.join(save_dir, 'pred_anno_correlations.txt'), 'w') as f:
        f.write(message)
