from setuptools import setup, find_packages

setup(
    name='hybrid_summarization_eval',
    version='0.0.1',
    author='David Biesner, Eduardo Brito, Lars Hillebrand',
    author_email='lars.patrick.hillebrand@iais.fraunhofer.de',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'flair',
        'nltk',
        'numpy',
        'pandas',
        'pylanguagetool',
        'pyyaml',
        'scipy',
        'seaborn',
        'sklearn',
        'spacy',
        'torch',
        'tqdm',
        'xlrd'
    ]
)
