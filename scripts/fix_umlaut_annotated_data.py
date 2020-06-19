import unicodedata
from konvens2020_summarization.data_classes import Corpus


def fix_unicode(text):
    text = unicodedata.normalize('NFC', text)
    return text


corpus = Corpus.from_excel('../data/raw_annotated/competition_data_annotated_full.xlsx')

for doc in corpus:
    doc.text = fix_unicode(doc.text)
    doc.ref_summary = fix_unicode(doc.ref_summary)
    for summary in doc.gen_summaries:
        summary.text = fix_unicode(summary.text)

# for char in corpus[0].text[14:20]:
#     print(char)
# print()
# for char in corpus[1].ref_summary[20:26]:
#     print(char)
# print()
# for char in corpus[1].gen_summaries[2].text[78:85]:
#     print(char)

corpus.to_json('../data/raw_annotated/corpus.json')
