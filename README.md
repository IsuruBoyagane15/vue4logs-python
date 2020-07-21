# vue4logs-python
The python version of vue4logs code converted from Ipython notebooks. These codes are used in creating benchmarks.


# Directory structure

1. library/ : codes of creating vocabulary and other low level components
2. results/ : Benchmarks of different experiments

## Sub-directories in results/
resulsts/**{experiment_type}_{epochs}**

Example: results/**baseline_1**
 - Results of experiment type = baseline and epochs = 1
  


# Files

- vue4logs.ipynb : end-to-end baseline implementation notebook
- vue4logs.py : end-to-end baseline implementation python

- vue4logs_biLSTM_encoder.ipynb : end-to-end biLSTM encoder implementation notebook
- vue4logs_biLSTM_encoder.py : end-to-end biLSTM encoder implementation python

- create_embeddings.py : STEP 1 implementation
- cluster.py : STEP 2 implementation
- get_log_templates.py : STEP 3 implementation
- configs.py : configurations for experiments
- init.py : Run all three steps at once

# Usage

To run all steps at once :
- configure dataset_nr, epochs and experiment type in configs.
- python init.py

To run steps separately :
- configure dataset_nr, epochs and experiment type in configs.
- step 1 : python create_embeddings.py
- step 2 : python cluster.py (embeddings.csv file created by step 1 should be there in relevant results directory to run this)
- step 3 : python get_log_templates.py (predicted_labels.csv file should be there in relevant results directory to run this)