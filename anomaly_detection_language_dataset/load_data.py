import os
import pandas as pd

data_folder = "data"
seed = 2023

error_msg = "run language_dataset_anomalies_data.ipynb first"

# obtain the full corpus of all words
corpus_file = f"{data_folder}/corpus.pkl"
if os.path.isfile(corpus_file):
    print(f"loading in english_train from {corpus_file}")
    corpus_df = pd.read_pickle(corpus_file)

# obtain the training data for the language model (large sample of the english words)
english_train_pickle_file = f"{data_folder}/english_train.pkl"
if os.path.isfile(english_train_pickle_file):
    print(f"loading in english_train from {english_train_pickle_file}")
    english_train = pd.read_pickle(english_train_pickle_file)

# obtain the test corpus of inliers and outliers words
corpus_sample_pickle_file = f"{data_folder}/corpus_sample.pkl"
if os.path.isfile(corpus_sample_pickle_file):
    print(f"loading in corpus_sample_df from {corpus_sample_pickle_file}")
    corpus_sample_df = pd.read_pickle(corpus_sample_pickle_file)
else:
    raise ValueError(error_msg)
