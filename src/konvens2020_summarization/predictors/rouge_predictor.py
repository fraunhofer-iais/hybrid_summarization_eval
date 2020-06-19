from typing import List, Tuple, Optional
from rouge import Rouge
from tqdm import tqdm
from konvens2020_summarization.data_classes import Corpus

rouge = Rouge()


def rouge_score(reference: str, hypothesis: str, metric: str = 'rouge-2') -> float:
    assert metric in ['rouge-1', 'rouge-2', 'rouge-l']
    scores = rouge.get_scores(reference, hypothesis)
    return scores[0][metric]['f']


def predict_rouge_score(corpus: Corpus,
                        name: Optional[str] = None,
                        methods: List[Tuple[str]] = [('rouge-2', 'reference')]):
    assert all([method[0] in ['rouge-1', 'rouge-2', 'rouge-l'] for method in methods])
    assert all([method[1] in ['summary', 'fulltext'] for method in methods])
    for method in methods:
        metric = method[0]
        reference = method[1]
        name = metric + '-' + reference
        for d in tqdm(corpus.documents):
            for s in d.gen_summaries:
                reference_text = d.text if reference == 'fulltext' else d.ref_summary
                s.predicted_scores[name] = rouge_score(reference=reference_text, hypothesis=s.text, metric=metric)
    return corpus
