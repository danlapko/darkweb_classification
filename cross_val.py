from typing import Union
import numpy as np
import pandas as pd
from multimodal_transformers.data import TorchTabularTextDataset
from multimodal_transformers.model import AutoModelWithTabular, TabularConfig
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Subset
from transformers import Trainer, TrainingArguments, AutoConfig


def cross_val(clf_factory, X: Union[np.ndarray, pd.DataFrame], y, n_splits=5, hierarchical=False):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=239)
    # avg = None if max(y) > 1 else 'binary'
    avg = 'binary'
    print(f"Cross Validation k_folds={n_splits}, n_classes={max(y) + 1}:")
    acc_pr = []
    acc_rec = []
    acc_f1 = []

    for i, (train, test) in enumerate(kf.split(X)):
        if isinstance(X, np.ndarray):
            X_train, X_test = X[train], X[test]
        else:
            X_train, X_test = X.loc[train], X.loc[test]
        y_train, y_test = y[train], y[test]
        clf = clf_factory()
        clf = clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hierarchical:
            pr_1 = precision_score(np.isin(y_test, [1, 2]), np.isin(y_pred, [1, 2]), average=avg)
            rec_1 = recall_score(np.isin(y_test, [1, 2]), np.isin(y_pred, [1, 2]), average=avg)
            f1_1 = f1_score(np.isin(y_test, [1, 2]), np.isin(y_pred, [1, 2]), average=avg)
            pr_2 = precision_score(y_test == 2, y_pred == 2, average=avg)
            rec_2 = recall_score(y_test == 2, y_pred == 2, average=avg)
            f1_2 = f1_score(y_test == 2, y_pred == 2, average=avg)
            pr = np.array([pr_1, pr_2])
            rec = np.array([rec_1, rec_2])
            f1 = np.array([f1_1, f1_2])
        else:
            pr = precision_score(y_test, y_pred, average=avg)
            rec = recall_score(y_test, y_pred, average=avg)
            f1 = f1_score(y_test, y_pred, average=avg)
        acc_pr.append(pr)
        acc_rec.append(rec)
        acc_f1.append(f1)
        print(f"\t{i} pr: {pr} rec: {rec} f1: {f1}")

    print(f"avg: pr={sum(acc_pr) / n_splits} rec={sum(acc_rec) / n_splits} f1={sum(acc_f1) / n_splits}")


def cross_val_torch(df: pd.DataFrame, dataset: TorchTabularTextDataset,
                    model_name: str, training_args: TrainingArguments,
                    combine_feat_method: str = 'weighted_feature_sum_on_transformer_cat_and_numerical_feats',
                    n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=239)
    print(f"Cross Validation k_folds={n_splits}, n_classes={2}:")
    acc_pr = []
    acc_rec = []
    acc_f1 = []

    for i, (train, test) in enumerate(kf.split(df)):
        train_dataset = Subset(dataset, train)
        test_dataset = Subset(dataset, test)
        model_config = AutoConfig.from_pretrained(model_name)
        tabular_config = TabularConfig(
            num_labels=2,
            cat_feat_dim=dataset.cat_feats.shape[1],
            numerical_feat_dim=dataset.numerical_feats.shape[1],
            combine_feat_method=combine_feat_method,

        )
        model_config.tabular_config = tabular_config
        model = AutoModelWithTabular.from_pretrained(model_name, config=model_config)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset
        )

        trainer.train()

        preds = trainer.predict(test_dataset)

        gt_labels = preds.label_ids
        preds = np.argmax(preds.predictions, axis=-1)

        pr = precision_score(gt_labels.astype(int), preds)
        rec = recall_score(gt_labels.astype(int), preds)
        f1 = f1_score(gt_labels.astype(int), preds)

        acc_pr.append(pr)
        acc_rec.append(rec)
        acc_f1.append(f1)
        print(f"\t{i} pr: {pr} rec: {rec} f1: {f1}")

    print(f"avg: pr={sum(acc_pr) / n_splits} rec={sum(acc_rec) / n_splits} f1={sum(acc_f1) / n_splits}")
