Hybrid Ensemble Predictor as Quality Metric for German Text Summarization: Fraunhofer IAIS at GermEval 2020 Task 3
======

This repository reproduces our work on automatic quality assesment of automatically generated German summaries.

Installation
--------
It can be installed in development mode with:

```sh

   $ git clone https://github.com/fraunhofer-iais/hybrid_summarization_eval.git
   $ cd hybrid_summarization_eval
   $ pip install -e .
```
The ``-e`` dynamically links the code in the git repository to the Python site-packages so your changes get
reflected immediately.


How to use
--------
The results obtained for the shared task can be reproduced by executing the script:
```sh

   $ python src/konvens2020_summarization/run_pipeline.py
```
Citation
--------
If you find our software useful in your work, please consider citing:

[1] Biesner, D., *et al.*: Hybrid Ensemble Predictor as Quality Metric for German Text Summarization: Fraunhofer IAIS at GermEval 2020 Task 3. 
