from konvens2020_summarization.data_classes import Corpus


def predict_compression_score(corpus: Corpus, name: str):
    for d in corpus.documents:
        org_len = len(d.text)
        for s in d.gen_summaries:
            summary_len = len(s.text)
            s.predicted_scores[name] = (org_len - summary_len)/org_len
    return corpus
