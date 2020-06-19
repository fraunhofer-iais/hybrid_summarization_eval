from argparse import ArgumentParser
from datetime import datetime
import json
import os
from typing import Dict

import yaml

from konvens2020_summarization import project_path, corpus_path
from konvens2020_summarization.aggregators import init_aggregator
from konvens2020_summarization.data_classes import Corpus, GenSummary
from konvens2020_summarization.evaluation import evaluate
from konvens2020_summarization.predictors import init_predictor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        default=os.path.join(project_path, 'config.yaml'),
                        help='Path to config file (yaml format).',
                        type=str)
    return parser.parse_args()


def run_pipeline(config: Dict):
    save_dir = os.path.join(project_path, 'experiments', f'{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}')
    os.makedirs(save_dir, exist_ok=True)
    json.dump(config, open(os.path.join(save_dir, 'config.json'), 'w'))

    if os.path.exists(os.path.join(project_path, 'data', 'saved_corpora', config['output'])):
        corpus = Corpus.from_json(os.path.join(project_path, 'data', 'saved_corpora', config['output']))
    else:
        # Parsing
        corpus = Corpus.from_json(file=corpus_path)
        if config['include_reference_summaries']:
            for doc in corpus:
                reference_summary = GenSummary(id_=-1,
                                               text=doc.ref_summary,
                                               annotated_scores={key: 1 for key in doc.gen_summaries[0].annotated_scores.keys()})
                doc.gen_summaries.append(reference_summary)

        # Run Predictors
        print('RUN PREDICTORS')
        for predictor_name, predictor_kwargs in config['predictors'].items():
            print(f'{predictor_name}')
            predictor = init_predictor(name=predictor_name)
            if not predictor_kwargs:
                predictor_kwargs = {}
            corpus = predictor(corpus=corpus,
                               name=predictor_name,
                               **predictor_kwargs)

        if 'output' in config:
            print(f'\nwriting to {config["output"]}...\n')
            corpus.to_json(os.path.join(project_path, 'data', 'saved_corpora', config['output']))
    remove_predictors = ['baseline_predictor']
    for doc in corpus:
        for s in doc.gen_summaries:
            for pred in remove_predictors:
                try:
                    del s.predicted_scores[pred]
                except KeyError:
                    pass

    # Run Aggregator
    predictor_names = list(corpus[0].gen_summaries[0].predicted_scores.keys())
    aggregator_kwargs = config['aggregator']
    aggregator = init_aggregator(name=aggregator_kwargs.pop('name'))
    corpus, result = aggregator(corpus=corpus, **aggregator_kwargs)
    regression_summary = result.summary(xname=predictor_names[:0] + ['constant'] + predictor_names[0:])
    print(regression_summary)
    with open(os.path.join(save_dir, 'reg_summary.tex'), 'w') as f:
        f.write(regression_summary.as_latex())
    with open(os.path.join(save_dir, 'reg_summary.txt'), 'w') as f:
        f.write(regression_summary.as_text())

    # Run Evaluation
    evaluate(corpus=corpus,
             annotated_score_names=None,
             predictor_names=predictor_names,
             save_dir=save_dir)

    # Export to CSV
    corpus.to_submission_csv(os.path.join(save_dir, 'submission.csv'))


def main():
    args = parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    run_pipeline(config)


if __name__ == '__main__':
    main()
