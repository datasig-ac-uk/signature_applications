import os
import math
import pandas as pd

from nlpsig import set_seed

data_folder = "data"
seed = 2023

# load in the wordlists for each language
wordlist_files = [
    f"{data_folder}/wordlist_de.txt",
    f"{data_folder}/wordlist_en.txt",
    f"{data_folder}/wordlist_fr.txt",
    f"{data_folder}/wordlist_it.txt",
    f"{data_folder}/wordlist_pl.txt",
    f"{data_folder}/wordlist_sv.txt",
]

wordlist_dfs = []
for filename in wordlist_files:
    with open(filename, "r") as f:
        words = f.read().splitlines()
        words_df = pd.DataFrame({"word": words})
        words_df["language"] = filename.split("_")[1][0:2]
        wordlist_dfs.append(words_df)

# concatenate all wordlists into one dataframe
corpus_df = pd.concat(wordlist_dfs).reset_index(drop=True)

# obtain the training data for the language model (large sample of the english words)
english_train_pickle_file = f"{data_folder}/english_train.pkl"
if os.path.isfile(english_train_pickle_file):
    print(f"loading in english_train from {english_train_pickle_file}")
    english_train = pd.read_pickle(english_train_pickle_file)
else:
    print(f"creating english_train dataframe with seed {seed}")
    # set seed for sampling
    set_seed(seed)
    n_words = len(corpus_df[corpus_df["language"]=="en"])-10000
    
    # sample english words from the corpus
    english_train = corpus_df[corpus_df["language"]=="en"].sample(n_words, random_state=seed)
    english_train = english_train.reset_index(drop=True)
    
    print(f"saving english_train to {english_train_pickle_file}")
    # save data for later
    english_train.to_pickle(english_train_pickle_file)
    
# remove the words that we use to train language model from the corpus
cond = corpus_df["word"].isin(english_train["word"])
corpus_df = corpus_df.drop(corpus_df[cond].index)
corpus_df = corpus_df.reset_index(drop=True)
corpus_df["language"].value_counts()

# obtain the sample of words for the anomaly detection task (10000 english words, 10000 non-english words)
corpus_sample_pickle_file = f"{data_folder}/corpus_sample.pkl"
if os.path.isfile(corpus_sample_pickle_file):
    print(f"loading in corpus_sample_df from {corpus_sample_pickle_file}")
    corpus_sample_df = pd.read_pickle(corpus_sample_pickle_file)
else:
    print(f"creating corpus_sample_df dataframe with seed {seed}")
    # set seed for sampling
    set_seed(seed)
    n_english = 10000
    n_remaining = 10000
    # sampling non-english words
    languages = corpus_df["language"].unique()
    words_per_language = math.floor(n_remaining / (len(languages)-1))
    non_english_df = pd.concat(
        [corpus_df[corpus_df["language"]==lang].sample(words_per_language, random_state=seed)
         for lang in languages if lang != "en"]
    )
    # take the remaining english words
    english_df = corpus_df[corpus_df["language"]=="en"]
    corpus_sample_df = pd.concat([non_english_df, english_df]).reset_index(drop=True)

    print(f"saving corpus_sample_df to {corpus_sample_pickle_file}")
    # save data for later
    corpus_sample_df.to_pickle(corpus_sample_pickle_file)
