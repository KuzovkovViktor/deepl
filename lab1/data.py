"""Завантаження Iris, стратифікований поділ 70/30 і стандартизація."""
import numpy as np
from sklearn.datasets import load_iris


def load_data():
    X, y = load_iris(return_X_y=True)
    X = X.astype(np.float64)

    rng = np.random.default_rng(0)
    train_idx, test_idx = [], []
    for c in (0, 1, 2):                      # класи в порядку 0, 1, 2
        idx = np.where(y == c)[0]
        idx = rng.permutation(idx)           # перемішуємо індекси класу
        train_idx.append(idx[:35])
        test_idx.append(idx[35:])
    train_idx = np.concatenate(train_idx)
    test_idx = np.concatenate(test_idx)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    mean = X_tr.mean(axis=0)                 # статистики лише з train
    std = X_tr.std(axis=0, ddof=0)
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std               # ті самі статистики для test
    return X_tr, y_tr, X_te, y_te
