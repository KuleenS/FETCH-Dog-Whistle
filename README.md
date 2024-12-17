# FETCH-Dog-Whistle

## Set up

To set up the virtual environment

```
conda env create -f environment.yml
```

This should create an environment named dogwhistle

## Setting up evaluation files

### Checking recall of your dataset (WIP for other data types)

To check the number of possible dog whistles in your dataset please run `src/scripts/recall.py`

Three parameters
- file: csv file with tweet column
- dogwhistle_file_path: allen ai glossary.tsv or reddit parquet file containing all the ground truth dog whistles
- output_folder: output folder for dogwhistles.recall file. Lists all dog whistles in your dataset

### Splitting dog whistles

To obtain a split of dog whistles to run fetch please run `src/scripts/split_train_test.py`

Three parameters
- dogwhistle_file_path: allen ai glossary.tsv or reddit parquet file containing all the ground truth dog whistles
- recall_file: file of the list of dog whistles from the groundtruth found in your data generated by `src/scripts/recall.py` (optional for reddit)
- output_folder: output folder for extrapolating.dogwhistles and given.dogwhistles files


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

### Setting up EarShot Database

Run either `src/earshot/scripts/embed.py` or `src/earshot/scripts/embed_other.py`

#### embed.py

Four parameters
- output_path: output path for output npz files
- input_files: list of .gz files from twitter/gab
- dogwhistle_file_path: allen ai glossary.tsv or reddit parquet file containing all the ground truth dog whistles
- id: id of process to distinguish between different processes outputting to same folder

#### embed_other.py

Four parameters
- output_path: output path for output npz files
- input_file: CSV or parquet file with either tweet or content as its single column name
- dogwhistle_file_path: allen ai glossary.tsv or reddit parquet file containing all the ground truth dog whistles
- id: id of process to distinguish between different processes outputting to same folder

### Running EarShot

To run earshot run `src/earshot/scripts/earshot.py`

- embedding_folder: folder of npz files from embed step
- dogwhistle_file: allen ai glossary.tsv or reddit parquet file containing all the ground truth dog whistles
- dogwhistle_path: path to extrapolating.dogwhistles and given.dogwhistles files
- collection_name: name of collection of database to save/load
- output_folder: path to save/load database lookup
- db_path: path to save/load database
- filtering_method: what earshot method to try
    - choices
        - bert-train: train a bert with seed posts as positive examples and random sample as negative samples 
        - bert-predict: use a model to predict yes or no to dog whistle in post
        - twostep: using LLM to predict yes or no to dog whistle in post
            - gemini-twostep: use gemini for this task
            - chatgpt-twostep: use chatgpt/openai for this task
            - offline-twostep: use offline huggingface model for this task
        - direct: directly asking LLM for dog whistle
            - offline-direct: use offline huggingface model for this task
            - chatgpt-direct: use chatgpt/openai for this task
            - gemini-direct: use gemini for this task
        - none: no filtering done beforehand
- llm_batch_size: batch size for LLMs
- models: LLMs/BERT to try 
- temperature: temperature of sampling from LLM
- bert-train parameters
    - lr: learning rate
    - weight_decay: L2 penalty
    - batch_size: batch size for BERT
    - epochs: number of epochs to train model
    - model_output_folder: where to save BERT models
- extraction_n_grams: list of max number of ngrams to extract using keyword extraction models