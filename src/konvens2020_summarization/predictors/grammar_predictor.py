import numpy as np
from pylanguagetool import api
from tqdm import tqdm
from konvens2020_summarization.data_classes import Corpus


def predict_grammar_score(corpus: Corpus, name: str, grammar_penalty: float):
    for d in tqdm(corpus.documents):
        for s in d.gen_summaries:
            n_issues = len(api.check(s.text, api_url='https://languagetool.org/api/v2/', lang='de')["matches"])
            text_len = len(s.text)
            s.predicted_scores[name] = max(0, 1 - grammar_penalty*n_issues/np.log(text_len))
    return corpus
