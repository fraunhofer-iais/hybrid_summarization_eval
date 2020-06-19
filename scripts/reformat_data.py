import os
import csv

from konvens2020_summarization import project_path

with open(os.path.join(project_path, 'data/raw/competition_data.csv'), encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    rows = [row for row in reader]

data = {}
for row in rows:
    id = row[0]
    text = row[1]
    reference = row[2]
    summary = row[3]
    if text not in data:
        data[text] = {'id': id,
                      'reference': reference,
                      'summaries': {id: summary}}
    else:
        data[text]['summaries'].update({id: summary})

data = {entry['id']: {'text': text, 'reference': entry['reference'], 'summaries': entry['summaries']} for text, entry in data.items()}

os.makedirs(os.path.join(project_path, 'data/parsed/'), exist_ok=True)

for id, entry in data.items():
    with open(os.path.join(project_path, 'data/parsed/', f'{str(id).zfill(3)}.txt'), 'w') as f:
        f.write('Original Text:' + '\n')
        f.write(entry['text'] + '\n')
        f.write('\n')
        f.write('Reference Summary:' + '\n')
        f.write(entry['reference'] + '\n')
        f.write('\n')
        for summary_id, summary in entry['summaries'].items():
            f.write(f'Generated Summary {summary_id}:' + '\n')
            f.write(summary + '\n')
            f.write('\n')
