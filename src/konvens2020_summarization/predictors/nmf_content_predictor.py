from collections import Counter, defaultdict
import random
from typing import List, Optional, Dict, Tuple
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

from konvens2020_summarization.data_classes import Corpus
from konvens2020_summarization.preprocessor import Preprocessor


def create_cooc_matrix(tokens: List[str],
                       window_size: int):

    word_counter = Counter()
    word_counter.update(tokens)

    words = [word for word, count in word_counter.most_common()]
    random.seed(42)
    random.shuffle(words)

    word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for w, i in word2id.items()}
    vocab_len = len(word2id)

    id_tokens = [word2id[w] for w in tokens]

    cooc_mat = defaultdict(Counter)
    for i, w in enumerate(id_tokens):
        start_i = max(i - window_size, 0)
        end_i = min(i + window_size + 1, len(id_tokens))
        for j in range(start_i, end_i):
            if i != j:
                c = id_tokens[j]
                cooc_mat[w][c] += 1 / abs(j - i)

    i_idx = list()
    j_idx = list()
    xij = list()

    # Create indexes and x values tensors
    for w, cnt in cooc_mat.items():
        for c, v in cnt.items():
            i_idx.append(w)
            j_idx.append(c)
            xij.append(v)

    cooc = csr_matrix((xij, (i_idx, j_idx)), shape=(vocab_len, vocab_len)).toarray()
    return cooc, id2word


def get_input_matrix(text: str,
                     matrix_type: str,
                     window_size: int):

    tokens = text.split()
    cooc, id2word = create_cooc_matrix(tokens=tokens,
                                       window_size=window_size)

    if matrix_type == 'cooc':
        output = cooc
    elif matrix_type == 'ppmi':
        total_sum = cooc.sum()
        row_sums = cooc.sum(axis=1)
        col_sums = cooc.sum(axis=0)

        pxy = cooc / total_sum
        px = np.tile((row_sums / total_sum), (cooc.shape[0], 1))
        py = np.tile((col_sums / total_sum), (cooc.shape[1], 1)).T

        pmi = np.log((pxy / (px * py)) + 1e-8)
        ppmi = np.where(pmi < 0, 0., pmi)
        output = ppmi
    else:
        raise ValueError(f'Matrix type {matrix_type} is not defined.')

    return output, id2word


def run_nmf(M: np.array,
            n_topics: int):

    nmf = NMF(n_components=n_topics, init='random', random_state=42, shuffle=True, solver='mu')
    W = nmf.fit_transform(M)
    H = nmf.components_
    return W, H


def get_topics(M: np.array,
               n: int,
               id2word: Dict[int, str]) -> Dict[str, List[str]]:
    M = M if M.shape[0] > M.shape[1] else M.T

    topics_paper = {}
    for dim in range(M.shape[1]):
        topics_paper[f'Topic {dim + 1}'] = []

        embeddings = M[np.argwhere(M.argmax(axis=1) == dim).flatten()]
        indices = np.argwhere(M.argmax(axis=1) == dim).flatten()

        new_sorting_order = np.argsort(embeddings[:, dim], axis=0)[::-1]

        embeddings_sorted = embeddings[new_sorting_order].astype(dtype='float64')
        indices_sorted = indices[new_sorting_order]

        counter = 0
        for embedding, id_ in zip(embeddings_sorted, indices_sorted):

            if counter < n:
                topics_paper[f'Topic {dim + 1}'].append(id2word[id_])

            counter += 1

        if embeddings_sorted.shape[0] < n:
            while len(topics_paper[f'Topic {dim + 1}']) < (n * 2):
                topics_paper[f'Topic {dim + 1}'].append('N/A')

    return topics_paper


def get_important_words(corpus: Corpus,
                        window_size: int = 5,
                        matrix_type: str = 'ppmi',
                        n_topics: int = 6,
                        n_topic_words: Optional[int] = 10,
                        preprocess_pipeline: Optional[List] = None,
                        ) -> Tuple[Dict, Corpus]:

    preprocessor = Preprocessor(pipeline=preprocess_pipeline)
    corpus_processed = preprocessor.process(corpus=corpus)

    important_words = {}
    for doc in corpus_processed.documents:
        matrix, id2word = get_input_matrix(text=doc.text_processed,
                                           matrix_type=matrix_type,
                                           window_size=window_size)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            W, H = run_nmf(M=matrix, n_topics=n_topics)

        topics_W = get_topics(M=W,
                              n=n_topic_words,
                              id2word=id2word)
        topics_H = get_topics(M=H,
                              n=n_topic_words,
                              id2word=id2word)

        imp_words = []
        for topic_W, topic_H in zip(topics_W.values(), topics_H.values()):
            imp_words.extend(topic_W)
            imp_words.extend(topic_H)
        imp_words = list(set(imp_words))

        important_words[doc.id_] = imp_words

    return important_words, corpus_processed


def predict_nmf_content_score(corpus: Corpus,
                              name: str,
                              n_topic_words: Optional[int] = 10,
                              n_topics: int = 6,
                              window_size: int = 5,
                              matrix_type: str = 'ppmi',
                              preprocessing: Optional[List] = None):

    important_words, corpus = get_important_words(corpus=corpus,
                                                  n_topic_words=n_topic_words,
                                                  n_topics=n_topics,
                                                  window_size=window_size,
                                                  matrix_type=matrix_type,
                                                  preprocess_pipeline=preprocessing)

    for doc in corpus.documents:
        important_doc_words = important_words[doc.id_]
        for gen_sum in doc.gen_summaries:
            gen_sum_counts = 0
            for imp_word in important_doc_words:
                if imp_word in gen_sum.text_processed.split(' '):
                    gen_sum_counts += 1

            try:
                gen_sum_score = gen_sum_counts / len(important_doc_words)
            except ZeroDivisionError:
                gen_sum_score = 0.

            gen_sum.predicted_scores[name] = gen_sum_score
    return corpus
