from konvens2020_summarization.data_classes import Corpus, default_corpus
import re

re_number = re.compile('[^0-9]')


def is_number(x: str) -> bool:
    if not x:
        return False
    return re_number.search(x) is None


def predict_number_matching_score(corpus: Corpus,
                                  name: str) -> Corpus:
    for doc in corpus.documents:
        numbers_original = [token for token in doc.text.split() if is_number(token)]
        for gen_sum in doc.gen_summaries:
            numbers_summary = [token for token in gen_sum.text.split() if is_number(token)]
            numbers_summary_matched = [number for number in numbers_summary if number in numbers_original]
            score = len(numbers_summary_matched) / len(numbers_summary) if numbers_summary else 0
            gen_sum.predicted_scores[name] = score
    return corpus


if __name__ == '__main__':
    corpus = default_corpus()
    corpus = predict_number_matching_score(corpus, name='number_matching')
    for doc in corpus:
        for s in doc.gen_summaries:
            print(s.predicted_scores['number_matching'])
