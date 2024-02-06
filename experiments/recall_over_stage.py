import json

import pandas as pd

from collections import defaultdict

df = pd.read_json('expansion.json', orient="records", lines=True)
dogwhistles_df = pd.read_csv("glossary.tsv", sep="\t")
dogwhistle_set = dogwhistles_df["Surface Forms"].str.split(";").tolist()

comparison_set = dogwhistles_df["Dogwhistle"].tolist()

dogwhistles = dict()

for i in range(len(dogwhistle_set)):
    for surface_form in dogwhistle_set[i]:
        dogwhistles[surface_form.strip().lower()] = comparison_set[i].strip().lower()
with open("extrapolating.dogwhistles", "r") as f:
    data = f.readlines()
data = [x.strip().lower() for x in data]
extrapolating_dogwhistle_map = {k : dogwhistles[k] for k in data}
len(set(extrapolating_dogwhistle_map.values()))
df[df["term"].map(extrapolating_dogwhistle_map).notna()]["source"].value_counts()