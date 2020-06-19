from sentence_transformers import SentenceTransformer
import scipy.spatial
from tqdm import tqdm
from konvens2020_summarization.data_classes import Corpus
import numpy as np
from sentence_transformers import models

multilingual_embedder = SentenceTransformer("distiluse-base-multilingual-cased")

word_embedding_model = models.Transformer("dbmdz/bert-base-german-uncased")
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=True)
uncased_embedder = SentenceTransformer(modules=[word_embedding_model, pooling_model])


def predict_similarity_score(corpus: Corpus, name: str, model="german-uncased"):
    if model == "german-uncased":
        embedder = uncased_embedder
    else:
        embedder = multilingual_embedder

    for d in tqdm(corpus.documents):
        doc_embedding = np.mean(embedder.encode(d.text), axis=0)
        for s in d.gen_summaries:
            summary_embedding = np.mean(embedder.encode(s.text), axis=0)
            s.predicted_scores[name] = 1-scipy.spatial.distance.cosine(summary_embedding, doc_embedding)

    return corpus
