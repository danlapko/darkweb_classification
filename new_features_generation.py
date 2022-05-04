import pandas as pd
from transformers import pipeline
from tqdm import tqdm

tqdm.pandas()

df = pd.read_csv("exploit_hacking_set Data Scientist.csv",
                 converters={"type": str, "label": str, "sub_label": str, "text": str, "title": str, "site": str})

# Load zero-shot-classifier
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli", device=0)


def get_label_score(text, candidate_labels):
    if text is None:
        text = ' '
    res = classifier(text, candidate_labels, multi_label=False)
    d = {label: score for label, score in zip(res['labels'], res['scores'])}
    first_label = res['labels'][0]
    return first_label, d


# Apply z-s classifier to Text column

# Chose this labels as the most frequent in dataset
# With more labels zero-shot classifier works bad
candidate_labels = [
    'hacking',
    'fraud',
    'adult',
]

# df = df.head(20)

df['zero_shot_tmp'] = df['text'].progress_apply(
    lambda text: get_label_score(text, candidate_labels))

df['zero_shot_1st_label'] = df['zero_shot_tmp'].progress_apply(
    lambda x: x[0])

for label in candidate_labels:
    df[f'zero_shot_{label}_score'] = df['zero_shot_tmp'].progress_apply(lambda x: x[1][label])

df.drop(columns=['zero_shot_tmp'])

df.to_csv("new_features.csv", index=False)

print(df)


# Apply z-s classifier to Title column

candidate_labels = ['other',
                    'carding',
                    'exploit',
                    'payments',
                    ]

# df = df.head(20)

df['zero_shot_tmp'] = df['text'].progress_apply(
    lambda text: get_label_score(text, candidate_labels))

df['zero_shot_1st_sub_label'] = df['zero_shot_tmp'].progress_apply(
    lambda x: x[0])

for label in candidate_labels:
    df[f'zero_shot_{label}_score'] = df['zero_shot_tmp'].progress_apply(lambda x: x[1][label])

df.drop(columns=['zero_shot_tmp'])

print(df)

# store dataset

df.to_csv("new_features.csv", index=False)
