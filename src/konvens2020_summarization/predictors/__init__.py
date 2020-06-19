from importlib import import_module

PREDICTORS = {'tfidf_content_predictor': ('konvens2020_summarization.predictors.tfidf_content_predictor',
                                          'predict_tfidf_content_score'),
              'tfidf_content_reversed_predictor': ('konvens2020_summarization.predictors.tfidf_content_reversed_predictor',
                                                   'predict_tfidf_content_reversed_score'),
              'flair_ner_content_predictor': ('konvens2020_summarization.predictors.flair_ner_content_predictor',
                                              'predict_flair_ner_content_score'),
              'flair_grammar_predictor': ('konvens2020_summarization.predictors.flair_grammar_predictor',
                                          'predict_flair_grammar_score'),
              'compression_predictor': ('konvens2020_summarization.predictors.compression_predictor',
                                        'predict_compression_score'),
              'grammar_predictor': ('konvens2020_summarization.predictors.grammar_predictor',
                                    'predict_grammar_score'),
              # 'spacy-transformer_predictor': ('konvens2020_summarization.predictors.spacy-transformer_predictor',
              #                                 'predict_similarity_score'),
              'baseline_predictor': ('konvens2020_summarization.predictors.baseline_predictor',
                                     'predict_baseline_score'),
              'sbert_predictor': ('konvens2020_summarization.predictors.sbert_predictor',
                                              'predict_similarity_score'),
              'random_predictor': ('konvens2020_summarization.predictors.baseline_predictor',
                                   'predict_random_score'),
              'sentence_similarity_predictor': ('konvens2020_summarization.predictors.sentence_similarity_predictor',
                                                'predict_sentence_similarity_score'),
              'nmf_content_predictor': ('konvens2020_summarization.predictors.nmf_content_predictor',
                                        'predict_nmf_content_score'),
              'rouge_predictor': ('konvens2020_summarization.predictors.rouge_predictor',
                                  'predict_rouge_score'),
              'number_matching_predictor': ('konvens2020_summarization.predictors.number_predictor',
                                            'predict_number_matching_score')}


def init_predictor(name: str):
    try:
        module_name, method_name = PREDICTORS[name]
    except KeyError:
        raise KeyError(f'Predictor {name} is not implemented. Choose among the following predictors:\n'
                       f'{list(PREDICTORS.keys())}')

    module = import_module(module_name)
    predictor = getattr(module, method_name)
    return predictor
