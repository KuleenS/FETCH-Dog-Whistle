import argparse

import os

import re

import matplotlib.pyplot as plt

import pandas as pd 

import seaborn as sns

def main(args):
    dataframes = []

    for tweet_file in args.tweet_files:
        df = pd.read_csv(tweet_file)

        df.index = pd.to_datetime(df['date'])

        dataframes.append(df)
    
    total_df = pd.concat(dataframes)
    
    resampled = total_df.resample('M')['match'].value_counts().to_csv(os.path.join(args.output_dir, "timeline.csv"))

    resampled["date"] = pd.to_datetime(resampled['date'])

    top_10 = resampled[["count", "match"]].groupby("match").sum().reset_index().sort_values("count", ascending=False).iloc[:5]["match"].tolist()

    a4_dims = (12, 9)
    fig, ax = plt.subplots(figsize=a4_dims)

    sns.lineplot(ax=ax, x='date',y='count',hue='match',data=resampled[resampled.match.isin(top_10)])

    plt.xticks(rotation=90)

    ax.legend(loc="upper left", title="Dogwhistle")

    ax.set_title("Top 5 Dogwhistles over Time")

    fig.savefig(os.path.join(args.output_dir, "timeline.png")) 

    import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
import pandas as pd
import string
 
nltk.download('stopwords')
nltk.download('punkt')
nltkstopwords = stopwords.words("english")
tweet_tok = TweetTokenizer()
df = pd.read_csv("filtered_tweets_dates_1.csv")
df.index = pd.to_datetime(df['date'])
import preprocessor as p
import nltk
from collections import Counter
df['tokenized'] = df['tweet'].apply(lambda x: p.clean(x)).apply(lambda x: [x.lower() for x in tweet_tok.tokenize(x)]).apply(lambda x: Counter([word for word in x if word not in string.punctuation and word not in nltkstopwords]))
from tqdm import tqdm
for u,v in df.groupby(pd.Grouper(freq="M")):
    blank = Counter()
    tweets = v['tokenized'].tolist()
    for tweet in tqdm(tweets):
        blank += tweet
    print(blank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--output_dir')

    args = parser.parse_args()

    main(args)