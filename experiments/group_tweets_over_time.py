import argparse

import os

import re

import matplotlib.pyplot as plt

import pandas as pd 

import seaborn as sns

def main(args):
    dogwhistle_glossary_df = pd.read_csv(args.dogwhistle_file_path, sep="\t")

    dogwhistle_set = dogwhistle_glossary_df["Surface Forms"].str.split(";").tolist()

    comparison_set = None

    comparison_set = dogwhistle_glossary_df["Dogwhistle"].tolist()
    
    surface_form_comparison_pairs = dict()

    for i in range(len(dogwhistle_set)):
        surface_forms = dogwhistle_set[i]
        for j in range(len(surface_forms)):
            surface_form_comparison_pairs[re.escape(surface_forms[j].lower().strip()).encode("utf-8")] = comparison_set[i]

    dataframes = []

    for tweet_file in args.tweet_files:
        df = pd.read_csv(tweet_file)

        df.index = pd.to_datetime(df['date'])

        df["match"] = df["match"].str.encode("utf-8").map(surface_form_comparison_pairs)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tweet_files', nargs='+')
    parser.add_argument('--dogwhistle_file_path')
    parser.add_argument('--output_dir')

    args = parser.parse_args()

    main(args)