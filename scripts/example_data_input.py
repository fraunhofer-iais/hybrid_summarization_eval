"""
Parsing file provided by the conference organizers
"""
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--id', type=int, default=0)
args = parser.parse_args()

# Read in data and print the 10th row
text_id = args.id

with open('../data/raw/competition_data.csv', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(reader)
    rows = [row for row in reader]

print('Text with id ' + str(text_id) + ':')
print(rows[text_id][1])

print('Reference summary:')
print(rows[text_id][2])

print('Generated summary:')
print(rows[text_id][3])


# Write an example submission file. Replace "0.5" with your prediction
with open('example_submission2.csv', 'w', newline = '', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter = ',', quotechar = '"')
    writer.writerow(['id', 'result'])
    for row in rows:
        writer.writerow([row[0], 0.5])
