from collections import Counter
from dataclasses import dataclass, field
import os
import json
from typing import List, Optional, Dict

import nltk
import pandas as pd
import numpy as np

from konvens2020_summarization import project_path


@dataclass
class Sentence:
    id_: int
    text: str
    text_processed: Optional[str] = None

    def to_dict(self):
        return {'id_': self.id_,
                'text': self.text,
                'text_processed': self.text_processed}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass
class GenSummary:
    id_: int  # KONVENS sample id
    text: str
    text_processed: Optional[str] = None
    text_vectorized: Optional[np.array] = None
    _sentences: List[Sentence] = None

    annotated_scores: Dict = field(default_factory=dict)
    predicted_scores: Dict = field(default_factory=dict)
    output_score: Optional[float] = None

    @property
    def sentences(self):
        if self._sentences is None:
            self._sentences = split_sentences(self.text)
        return self._sentences

    def to_dict(self):
        predicted_scores = {key: float(value) for key, value in self.predicted_scores.items()}
        annotated_scores = {key: float(value) for key, value in self.annotated_scores.items()}
        data = {
            'id_': self.id_,
            'text': self.text,
            'text_processed': self.text_processed,
            '_sentences': self._sentences,
            'annotated_scores': annotated_scores,
            'predicted_scores': predicted_scores,
            'output_score': float(self.output_score) if self.output_score is not None else None
        }
        return data

    @classmethod
    def from_dict(cls, data):
        if data.get('_sentences') is not None:
            data['sentences'] = [Sentence.from_dict(sentence) for sentence in data['_sentences']]
        return cls(**data)


@dataclass
class Document:
    id_: int  # our internal document id
    text: str
    ref_summary: str
    gen_summaries: List[GenSummary]
    ref_summary_processed: Optional[str] = None
    ref_summary_vectorized: Optional[np.array] = None
    text_processed: Optional[str] = None
    text_vectorized: Optional[np.array] = None
    _sentences: List[Sentence] = None

    @property
    def sentences(self):
        if self._sentences is None:
            self._sentences = split_sentences(self.text)
        return self._sentences

    def to_dict(self):
        data = {
            'id_': self.id_,
            'text': self.text,
            'ref_summary': self.ref_summary,
            'gen_summaries': [summary.to_dict() for summary in self.gen_summaries],
            'ref_summary_processed': self.ref_summary_processed,
            'text_processed': self.text_processed,
            '_sentences': [sentence.to_dict() for sentence in self.sentences],
        }
        return data

    @classmethod
    def from_dict(cls, data):
        data['gen_summaries'] = [GenSummary.from_dict(summary) for summary in data['gen_summaries']]
        if data.get('_sentences') is not None:
            data['_sentences'] = [Sentence.from_dict(sentence) for sentence in data['_sentences']]
        return cls(**data)


@dataclass
class Corpus:
    documents: List[Document]

    def to_dict(self):
        data = {
            'documents': [doc.to_dict() for doc in self.documents]
        }
        return data

    @classmethod
    def from_dict(cls, data):
        data['documents'] = [Document.from_dict(doc) for doc in data['documents']]
        return cls(**data)

    def to_json(self, file):
        json.dump(self.to_dict(), open(file, 'w', encoding='utf-8'))

    @classmethod
    def from_json(cls, file):
        data = json.load(open(file, 'r'))
        return cls.from_dict(data)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        return self.documents[idx]

    def to_submission_csv(self, path):
        summaries = [summary for document in self.documents for summary in document.gen_summaries]
        summaries = sorted(summaries, key=lambda summary: summary.id_)
        with open(path, 'w') as f:
            f.write('id,result\n')
            for summary in summaries:
                # Remove reference summaries
                if summary.id_ == -1:
                    continue
                f.write(f'{summary.id_},{np.round(summary.output_score, 2)}\n')

    @classmethod
    def from_excel(cls,
                   path: Optional[str] = None):

        # Parse excel with Pandas
        if path:
            raw_data = pd.read_excel(path, encoding='utf-8').values
        else:
            raw_data_path = os.path.join(project_path,
                                         f'data{os.sep}raw_annotated{os.sep}competition_data_annotated_full.xlsx')
            raw_data = pd.read_excel(raw_data_path, encoding='utf-8').values

        # Create mapping dict from document text to id
        text_to_id = {text: i for i, text in enumerate(Counter(raw_data[:, 1].tolist()).keys())}

        data = {}
        for sample in raw_data:
            id_ = sample[0]
            text = sample[1]
            ref_summary = sample[2]
            gen_summary = sample[3]
            total_score = sample[4]
            content_score = sample[5]
            grammar_score = sample[6]
            compact_score = sample[7]
            abstract_score = sample[8]

            if text_to_id[text] not in data:
                data[text_to_id[text]] = {'id_': text_to_id[text],
                                          'value': text,
                                          'ref_summary': ref_summary,
                                          'gen_summaries': {id_: {'text': gen_summary,
                                                                  'total_score': total_score,
                                                                  'content_score': content_score,
                                                                  'grammar_score': grammar_score,
                                                                  'compact_score': compact_score,
                                                                  'abstract_score': abstract_score}}}
            else:
                data[text_to_id[text]]['gen_summaries'].update({id_: {'text': gen_summary,
                                                                      'total_score': total_score,
                                                                      'content_score': content_score,
                                                                      'grammar_score': grammar_score,
                                                                      'compact_score': compact_score,
                                                                      'abstract_score': abstract_score}})

        # Create dataclass structure
        documents = []
        for doc in data.values():
            gen_summaries = [GenSummary(id_=id_,
                                        text=gen_summary['text'],
                                        annotated_scores={'total_score': gen_summary['total_score'],
                                                          'content_score': gen_summary['content_score'],
                                                          'grammar_score': gen_summary['grammar_score'],
                                                          'compact_score': gen_summary['compact_score'],
                                                          'abstract_score': gen_summary['abstract_score']})
                             for id_, gen_summary in doc['gen_summaries'].items()]

            documents.append(Document(id_=doc['id_'],
                                      text=doc['value'],
                                      ref_summary=doc['ref_summary'],
                                      gen_summaries=gen_summaries))

        return cls(documents=documents)

    @classmethod
    def from_csv(cls,
                 path: Optional[str] = None):

        # Parse csv with Pandas
        if path:
            raw_data = pd.read_csv(path, sep=',', encoding='utf-8').values
        else:
            raw_data_path = os.path.join(project_path, f'data{os.sep}raw{os.sep}competition_data.csv')
            raw_data = pd.read_csv(raw_data_path, sep=',', encoding='utf-8').values

        # Create mapping dict from document text to id
        text_to_id = {text: i for i, text in enumerate(Counter(raw_data[:, 1].tolist()).keys())}

        data = {}
        for id_, text, ref_summary, gen_summary in raw_data:

            if text_to_id[text] not in data:
                data[text_to_id[text]] = {'id_': text_to_id[text],
                                          'value': text,
                                          'ref_summary': ref_summary,
                                          'gen_summaries': {id_: gen_summary}}
            else:
                data[text_to_id[text]]['gen_summaries'].update({id_: gen_summary})

        # Create dataclass structure
        documents = []
        for doc in data.values():
            gen_summaries = [GenSummary(id_=id_,
                                        text=gen_summary) for id_, gen_summary in doc['gen_summaries'].items()]

            documents.append(Document(id_=doc['id_'],
                                      text=doc['value'],
                                      ref_summary=doc['ref_summary'],
                                      gen_summaries=gen_summaries))

        return cls(documents=documents)


def split_sentences(text: str) -> List[Sentence]:
    sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle')
    sentences = sentence_detector.tokenize(text)
    return [Sentence(id_=i, text=sentence) for i, sentence in enumerate(sentences)]


def default_corpus():
    return Corpus.from_json(os.path.join(project_path, 'data', 'raw_annotated', 'corpus.json'))