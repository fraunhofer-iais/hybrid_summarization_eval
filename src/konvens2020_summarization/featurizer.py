import pickle
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer

from konvens2020_summarization.data_classes import Corpus, Document


class TfidfFeaturizer:
    def __init__(self,
                 train_documents: Optional[List[Document]] = None,
                 model_path: Optional[str] = None,
                 model_args: Optional[Dict] = None
                 ):
        self.model_path = model_path

        try:
            self.model = self._load()
        except (FileNotFoundError, TypeError):
            tfidf_args = model_args if model_args is not None else {}
            if not train_documents:
                raise ValueError('If not loading from an existing model, training corpus must be provided.')
            self.model = self._train(train_documents, tfidf_args)

    def _load(self) -> TfidfVectorizer:
        return pickle.load(open(self.model_path, 'rb'))

    def _save(self, model: TfidfVectorizer):
        pickle.dump(model, open(self.model_path), 'wb')

    def _train(self,
               train_corpus: List[Document],
               tfidf_args: Dict) -> TfidfVectorizer:
        inputs = [doc.text_processed for doc in train_corpus]
        vectorizer = TfidfVectorizer(**tfidf_args,
                                     lowercase=False)
        model = vectorizer.fit(inputs)
        if self.model_path:
            self._save(model)
        return model

    @property
    def size(self) -> int:
        return len(self.model.vocabulary_)

    @property
    def vocab(self) -> Dict:
        return self.model.vocabulary_

    @property
    def feature_names(self) -> List:
        return self.model.get_feature_names()

    def featurize(self,
                  corpus: Corpus) -> Corpus:

        for doc in corpus.documents:
            doc.text_vectorized = self.model.transform([doc.text_processed]).toarray()
            doc.ref_summary_vectorized = self.model.transform([doc.ref_summary_processed]).toarray()
            for gen_sum in doc.gen_summaries:
                gen_sum.text_vectorized = self.model.transform([doc.text_processed]).toarray()
        return corpus
