"""Метрики, вартість і пошук порога для ЛР2.

Правило рішення: прогноз «шахрайство» (1), якщо s_i >= t.
Вартість: C(t) = 10*FN(t) + FP(t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FN_COST = 10   # пропущене шахрайство
FP_COST = 1    # хибна тривога

def confusion(y: np.ndarray, s: np.ndarray, t: float) -> tuple[int, int, int, int]:
    """TP, FP, FN, TN для порога t за правилом s >= t."""
    y = np.asarray(y)
    pred = np.asarray(s, dtype=np.float64) >= t
    pos = y == 1
    tp = int(np.count_nonzero(pred & pos))
    fp = int(np.count_nonzero(pred & ~pos))
    fn = int(np.count_nonzero(~pred & pos))
    tn = int(np.count_nonzero(~pred & ~pos))
    return tp, fp, fn, tn


def cost(fn: int, fp: int) -> int:
    return FN_COST * fn + FP_COST * fp


def precision(tp: int, fp: int) -> float | None:
    """None, якщо TP+FP = 0 (жодної позитивної тривоги: частка не визначена)."""
    return tp / (tp + fp) if (tp + fp) > 0 else None


def recall(tp: int, fn: int) -> float | None:
    """None, якщо TP+FN = 0 (у вибірці немає шахрайств)."""
    return tp / (tp + fn) if (tp + fn) > 0 else None


# ----------------------------------------------------------------------------
# Швидкий пошук: сортування + кумулятивні суми, O(n log n)
# ----------------------------------------------------------------------------
def candidate_table(y: np.ndarray, s: np.ndarray) -> pd.DataFrame:
    """Таблиця всіх кандидатів: threshold, TP, FP, FN, TN, C.

    Кандидати: усі унікальні оцінки та +inf. Рядки впорядковано за спаданням
    порога (першим іде +inf). Рівні оцінки отримують однаковий прогноз,
    тому кумулятивні суми беруться лише в кінці кожної групи рівних оцінок.
    """
    y = np.asarray(y).astype(np.int64)
    s = np.asarray(s, dtype=np.float64)
    n = len(s)
    if n == 0 or len(y) != n:
        raise ValueError("y і s мають бути непорожніми та однакової довжини")
    if not np.isfinite(s).all():
        raise ValueError("Оцінки мають бути скінченними")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("Мітки мають бути 0/1")

    order = np.argsort(-s, kind="stable")          # за спаданням оцінки
    s_sorted = s[order]
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)                   # позитивів серед перших k
    cum_fp = np.cumsum(1 - y_sorted)               # негативів серед перших k

    # кінець групи рівних оцінок: наступна оцінка менша (або кінець масиву)
    is_last = np.ones(n, dtype=bool)
    is_last[:-1] = s_sorted[:-1] != s_sorted[1:]

    thr = np.concatenate(([np.inf], s_sorted[is_last]))
    tp = np.concatenate(([0], cum_tp[is_last]))    # при t=+inf ніхто не позитивний
    fp = np.concatenate(([0], cum_fp[is_last]))
    n_pos = int(y.sum())
    n_neg = n - n_pos
    fn = n_pos - tp
    tn = n_neg - fp
    c = FN_COST * fn + FP_COST * fp
    return pd.DataFrame(
        {"threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": c}
    )


def choose_threshold(table: pd.DataFrame) -> float:
    """Поріг з мінімальною вартістю; за рівної вартості - найбільший."""
    best = table["C"].min()
    return float(table.loc[table["C"] == best, "threshold"].max())


# ----------------------------------------------------------------------------
# Повільний незалежний перебір (ЛИШЕ для малих наборів з п. 3 завдання)
# ----------------------------------------------------------------------------
def brute_force_table(y: list[int], s: list[float]) -> pd.DataFrame:
    """Кандидати й лічильники прямим застосуванням правила s_i >= t.

    Не використовує сортування, кумулятивні суми та жодних результатів
    швидкого алгоритму: чистий Python, подвійний цикл.
    """
    candidates = [float("inf")] + sorted(set(float(v) for v in s), reverse=True)
    rows = []
    for t in candidates:
        tp = fp = fn = tn = 0
        for yi, si in zip(y, s):
            predicted_fraud = si >= t
            if predicted_fraud and yi == 1:
                tp += 1
            elif predicted_fraud and yi == 0:
                fp += 1
            elif (not predicted_fraud) and yi == 1:
                fn += 1
            else:
                tn += 1
        rows.append((t, tp, fp, fn, tn, 10 * fn + 1 * fp))
    return pd.DataFrame(rows, columns=["threshold", "TP", "FP", "FN", "TN", "C"])


def brute_force_choice(table: pd.DataFrame) -> float:
    """Незалежний вибір порога: мінімум C; при нічиїй - найбільший поріг."""
    best_c, best_t = None, None
    for t, c in zip(table["threshold"], table["C"]):
        if best_c is None or c < best_c or (c == best_c and t > best_t):
            best_c, best_t = c, t
    return float(best_t)
