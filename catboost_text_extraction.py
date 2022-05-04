import pandas as pd
from catboost import CatBoostClassifier
from cross_val import cross_val

df = pd.read_csv("old_n_new_features.csv",
                 converters={"type": str, "label": str, "sub_label": str, "text": str, "title": str, "site": str})

df["spont_score"] = df["spont_score"].fillna(value=0.239)

y_hac = df['label'] == 'Hacking'
y_hac = y_hac.astype(int)
y_exp = df['sub_label'] == 'Exploit'
y_exp = y_exp.astype(int)

y_hac_exp = df['sub_label'] == 'Exploit'
y_hac_exp[y_hac == 1] = 1
y_hac_exp[y_exp == 1] = 2
y_hac_exp = y_hac_exp.astype(int)

df_X = df.drop(columns=['label', 'sub_label'])
if 'zero_shot_tmp' in df_X.columns.values:
    df_X = df_X.drop(columns=['zero_shot_tmp'])

df_X['text_len'] = df_X['text'].apply(len)
df_X['title_len'] = df_X['title'].apply(len)

df_X['text'] = df_X['text'].apply(lambda x: x[:2000])
df_X['title'] = df_X['title'].apply(lambda x: x[:300])

clf_factory = lambda: CatBoostClassifier(iterations=500,
                                         learning_rate=0.2,
                                         loss_function='Logloss',
                                         verbose=False,
                                         text_features=['text', 'title'],
                                         cat_features=['type', 'site',
                                                       'zero_shot_1st_label', 'zero_shot_1st_sub_label'],
                                         feature_calcers=['BoW:top_tokens_count=2000', 'BoW:top_tokens_count=500'],
                                         )
cross_val(clf_factory, df_X, y_exp)
