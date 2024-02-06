import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

twitter_timeline = pd.read_csv("/export/b05/ksasse1/dogwhistle-kg/filtered_tweets_date/timeline.csv")

gab_timeline = pd.read_csv("/export/b05/ksasse1/dogwhistle-kg/filtered_gabs_timeline/timeline.csv")
twitter_timeline = twitter_timeline.set_index(pd.DatetimeIndex(pd.to_datetime(twitter_timeline["date"])).tz_localize(None))
gab_timeline = gab_timeline.set_index(pd.DatetimeIndex(pd.to_datetime(gab_timeline["date"])).tz_localize(None))

bottom = max(gab_timeline.index.values[0], twitter_timeline.index.values[0])
top = min(gab_timeline.index.values[-1], twitter_timeline.index.values[-1])
twitter_filtered = twitter_timeline[bottom: top]
gab_filtered = gab_timeline[bottom: top]
twitter_filtered.reset_index(drop=True, inplace=True)
gab_filtered.reset_index(drop=True, inplace=True)
merge = twitter_filtered.merge(gab_filtered, how="left", left_on=["date", "match"], right_on=["date", "match"])
merge["count_z"] = merge["count_x"] + merge["count_y"]
top5 = merge.groupby("match")["count_z"].sum().sort_values(ascending=False)[0:5].index.values


a4_dims = (12, 9)
fig, ax = plt.subplots(figsize=a4_dims)

sns.lineplot(ax=ax, x='date',y='count_x',hue='match', data=merge[merge.match.isin(top5)], palette=sns.color_palette("Set1"))
sns.lineplot(ax=ax, x='date',y='count_y',hue='match', data=merge[merge.match.isin(top5)], palette=sns.color_palette("Set2"))

plt.xticks(rotation=90)

ax.legend(loc="upper left", title="Dogwhistle")

ax.set_title("Top 5 Dogwhistles over Time")