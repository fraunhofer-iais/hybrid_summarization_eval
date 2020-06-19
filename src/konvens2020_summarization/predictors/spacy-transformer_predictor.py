# import spacy
# from spacy.cli.download import download
# from konvens2020_summarization.data_classes import Corpus
# from tqdm import tqdm
#
# try:
#     nlp = spacy.load("de_trf_bertbasecased_lg")
# except OSError:
#     download("de_trf_bertbasecased_lg")
#     nlp = spacy.load("de_trf_bertbasecased_lg")
#
#
# def predict_similarity_score(corpus: Corpus, name: str):
#     for d in tqdm(corpus.documents):
#         doc_tensor = nlp(d.text)
#         for s in d.gen_summaries:
#             summary_tensor = nlp(s.text)
#             s.predicted_scores[name] = doc_tensor[0].similarity(summary_tensor[0])
#     return corpus
