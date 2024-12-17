# FETCH-Dog-Whistle

## Set up

To set up the virtual environment

```
conda env create -f environment.yml
```

This should create an environment named dogwhistle

## Setting up evaluation files

## Running Word2Vec

### Preparing Data

To prepare data for word2vec use either `src/euphemism_detection/word2vec/prepare_data.py` or `src/euphemism_detection/word2vec/other_prepare_data.py`

#### prepare_data.py

Other prepare data is for preparing gab/twitter compressed files for Word2Vec training. Deduplicates using bloom filter

Three command line parameters
- tweet_files: list of .gz files from twitter/gab
- output_folder: output folder for prepared data
- id: id of process to distinguish between different processes outputting to same folder

#### other_prepare_data.py

Other prepare data is for preparing CSVs and parquet files for Word2Vec training

Two command line parameters
- file: CSV or parquet file with either tweet or content as its single column name
- output_folder: output folder for prepared data

### Building Corpora

After building the data you need to create the corpora for Word2Vec/Phrase2Vec training

#### Building unigram corpus

Run `src/euphemism_detection/word2vec/counts.py` on each file in the output output folder

Two parameters
- input_file: input txt file from `prepare_data.py` or `other_prepare_data.py`
- output_file: output json file of counts \

Then run `src/euphemism_detection/word2vec/prune.py` on all the counts files

Two parameters
- input_files: list of count json files
- output_file: output json file of total counts

#### Building 2+-gram corpora

To build Phrase2Vec run `src/euphemism_detection/word2vec/phrase.py`

Three parameters
- input_files: n-1th list of text files generated by `phrase.py` or by `prepare_data.py`/`other_prepare_data.py`
- output_folder: output folder of new txt files
- vocab: n-1th gram counts json file

Then run `src/euphemism_detection/word2vec/counts.py` on phrase output
Then run `src/euphemism_detection/word2vec/prune.py` on counts output

Repeat till your desired length

### Training Models

To train Word2Vec, run `src/word2vec/train_word2vec.py`

Two parameters
- input_dir: directory for text files generated by `prepare_data.py` or `other_prepare_data.py` or `phrase.py`
- output_dir: output directory for saved model

### Prediction

To get predictions from Word2Vec run `src/word2vec/predict.py`

Five parameters
- model: path to trained word2vec model
- keywords: path to seed dogwhistle file
- output_dir: where to output results
- n_rounds: how many rounds of expansions 
- top_k: how many terms to expand within each round

### Evaluation

To get evaluation from Word2Vec/Phrase2Vec run `src/word2vec/word2vec_eval.py`

- dogwhistle_file_path: allen ai glossary.tsv or reddit parquet file containing all the ground truth dog whistles
- extrapolating_dogwhistle_path: path to extrapolating dogwhistle file
- expansions_path: path to expansions from prediction step
- ngrams_possible: ngrams that you model predicts

## Running Neural Methods (MLM and EPD)



### MLM

To run the EPD experiments use `src/neural/run_single.py`

The parameters for the file are:
- dogwhistle_file
- dogwhistle_path
- phrase_candidate_file
- word2vec_file
- model_name
- tweet_files
- file
- sampled_file
- output_path

### EPD

To run the EPD experiments use `src/neural/run_multiple.py`

The parameters for the file are:
- dogwhistle_file
- dogwhistle_path
- phrase_candidate_file
- word2vec_file
- model_name
- tweet_files
- file
- sampled_file
- output_path

## EarShot

### Setting up EarShot

### Running EarShot
