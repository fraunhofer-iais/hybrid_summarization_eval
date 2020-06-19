import nltk

from konvens2020_summarization import data_classes

sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')


class SentenceSimilarityEvaluator:

    def __init__(self, method='ratio'):
        assert method in ['ratio', 'first', 'consecutive', 'consecutive_not_first']
        self.method = method

    def __call__(self, doc: data_classes.Document, summary: data_classes.GenSummary):
        ratio, first_sentences, consecutive_sentences, indices = sentence_similarity(original_text=doc.text,
                                                                                     summary_text=summary.text)
        if self.method == 'ratio':
            return ratio
        if self.method == 'first':
            return float(first_sentences)
        if self.method == 'consecutive':
            return float(consecutive_sentences)
        if self.method == 'consecutive_not_first':
            return float(consecutive_sentences and not first_sentences)


def split_sentences(text):
    return sentence_detector.tokenize(text)


def sentence_similarity(original_text: str, summary_text: str):
    original_sentences = split_sentences(original_text)
    summary_sentences = split_sentences(summary_text)
    indices = []
    for sentence in summary_sentences:
        try:
            indices.append(original_sentences.index(sentence))
        except ValueError:
            pass
    first_sentences = set(indices) == set(range(len(indices))) if len(indices) > 0 else False
    consecutive_sentences = max(indices) - min(indices) == len(indices) - 1 if len(indices) > 0 else False
    ratio = len(indices) / len(summary_sentences)
    return ratio, first_sentences, consecutive_sentences, indices


def predict_sentence_similarity_score(corpus: data_classes.Corpus,
                                      name: str = 'sentence_similarity',
                                      method: str = 'consecutive_not_first') -> data_classes.Corpus:
    evaluator = SentenceSimilarityEvaluator(method=method)

    for doc in corpus:
        for summary in doc.gen_summaries:
            summary.predicted_scores[name] = evaluator(doc=doc, summary=summary)

    return corpus


def main():
    import os
    from konvens2020_summarization import project_path

    corpus = data_classes.Corpus.from_json(os.path.join(project_path, 'data/raw_annotated/corpus.json'))
    corpus_predicted = predict_sentence_similarity_score(corpus, 'sentence_similarity', method='consecutive_not_first')


if __name__ == '__main__':
    main()
