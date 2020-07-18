# vue4logs-python
The python version of vue4logs code converted from Ipython notebooks. These codes are used in creating benchmarks.


# Directory structure

1. library/ : codes of creating vocabulary and other low level components
2. results/ : Benchmarks of different experiments

vue4logs.ipynb : end-to-end baseline implementation notebook
vue4logs.py : end-to-end baseline implementation python

vue4logs_biLSTM_encoder.ipynb : end-to-end biLSTM encoder implementation notebook
vue4logs_biLSTM_encoder.py : end-to-end biLSTM encoder implementation python

- create_embeddings.py : STEP 1 implementation
- cluster.py : STEP 2 implementation
- get_log_templates.py : STEP 3 implementation
- configs.py : configurations for experiments
- init.py : Run all three steps at once

# Usage

- configure dataset_nr, epochs and experiment type in configs.
- python init.py
