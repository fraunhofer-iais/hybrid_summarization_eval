import os
package_path = os.path.dirname(os.path.abspath(__file__))
project_path = os.path.abspath(os.path.join(package_path, os.pardir, os.pardir))
data_path = os.path.join(project_path, f'data')
corpus_path = os.path.join(data_path, f'raw_annotated{os.sep}corpus.json')
