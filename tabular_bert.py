import pandas as pd
from multimodal_transformers.data import load_data
from transformers import AutoTokenizer

from transformers import TrainingArguments

from cross_val import cross_val_torch

model_name = "distilbert-base-uncased"  # 'bert-base-uncased'
lr = 5e-5
batch_size = 20
num_epochs = 2
# combine_feat_method='text_only',
# combine_feat_method='concat',
combine_feat_method='mlp_on_categorical_then_concat'
# combine_feat_method='gating_on_cat_and_num_feats_then_sum'
# combine_feat_method = 'weighted_feature_sum_on_transformer_cat_and_numerical_feats'

df = pd.read_csv("exploit_hacking_set Data Scientist.csv",
                 converters={"type": str, "label": str, "sub_label": str, "text": str, "title": str, "site": str})

df["spont_score"] = df["spont_score"].fillna(value=0.239)
df.drop(columns=['sub_label'])
df['label'] = df['label'] == 'Hacking'
df['label'] = df['label'].astype(int)
df['text_len'] = df['text'].apply(len)
df['title_len'] = df['title'].apply(len)

text_cols = ['text', 'title']
label_col = 'label'
categorical_cols = ['type', 'site', 'spont_score']
numerical_cols = ['text_len', 'title_len']
label_list = ['not Hacking', 'Hacking']  # what each label class represents
num_labels = 2

# make sure NaN values for cat columns are filled before passing to load_data
for c in categorical_cols:
    df.loc[:, c] = df.loc[:, c].astype(str).fillna("-9999999")

for c in text_cols:
    df.loc[:, c] = df.loc[:, c].astype(str).fillna(" ")

# df = df.head(2000)

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_data(
    df,
    text_cols,
    tokenizer,
    label_col=label_col,
    categorical_cols=categorical_cols,
    numerical_cols=numerical_cols,
    sep_text_token_str=tokenizer.sep_token,
)

training_args = TrainingArguments(
    output_dir="./logs/model_name",
    logging_dir="./logs/runs",
    overwrite_output_dir=True,
    do_train=True,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    evaluate_during_training=False,
    learning_rate=lr,
    logging_steps=25,
)

cross_val_torch(df, dataset, model_name, training_args, combine_feat_method)