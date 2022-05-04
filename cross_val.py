from typing import Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score


def cross_val(clf_factory, X: Union[np.ndarray, pd.DataFrame], y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=239)
    avg = None if max(y) > 1 else 'binary'
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
        pr = precision_score(y_test, y_pred, average=avg)
        rec = recall_score(y_test, y_pred, average=avg)
        f1 = f1_score(y_test, y_pred, average=avg)
        acc_pr.append(pr)
        acc_rec.append(rec)
        acc_f1.append(f1)
        print(f"\t{i} pr: {pr} rec: {rec} f1: {f1}")

    print(f"avg: pr={sum(acc_pr) / n_splits} rec={sum(acc_rec) / n_splits} f1={sum(acc_f1) / n_splits}")
