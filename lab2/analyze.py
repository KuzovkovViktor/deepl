"""ЛР2, пп. 2-4: аналіз зі збережених прогнозів.

    uv run python analyze.py --saved results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from lab2_starter import load_predictions, split_summary
from thresholds import (
    brute_force_choice,
    brute_force_table,
    candidate_table,
    choose_threshold,
    confusion,
    cost,
    precision,
    recall,
)

# Малі набори з п. 3 завдання (набір C: 0.8; одинадцять разів 0.5; 0.2,
# причому перша з одинадцяти п'ятірок має мітку 1, решта десять - мітку 0).
SMALL_SETS = {
    "A": ([0.9, 0.7, 0.7, 0.4, 0.2, 0.1], [0, 1, 0, 1, 0, 0]),
    "B": ([0.9, 0.6, 0.3], [0, 0, 0]),
    "C": ([0.8] + [0.5] * 11 + [0.2], [1] + [1] + [0] * 10 + [0]),
}
# Очікування, отримані вручну (незалежно від коду): (t*, мінімальна C)
EXPECTED = {"A": (0.4, 2), "B": (float("inf"), 0), "C": (0.8, 10)}


def fmt(v, digits=6):
    if v is None:
        return "не визначена"
    if isinstance(v, float) and np.isinf(v):
        return "+inf" if v > 0 else "-inf"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.{digits}g}"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def row_for_rule(name, y, s, t):
    tp, fp, fn, tn = confusion(y, s, t)
    return {
        "Правило": name, "Поріг": fmt(t, 8), "TP": tp, "FP": fp, "FN": fn,
        "TN": tn, "C": cost(fn, fp),
        "precision": fmt(precision(tp, fp), 4), "recall": fmt(recall(tp, fn), 4),
    }


# ----------------------------------------------------------------------------
def plot_curves(y, s, marks, out: Path):
    prec, rec, _ = precision_recall_curve(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    auc = roc_auc_score(y, s)
    ap = average_precision_score(y, s)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=1.5, label=f"PR (AP = {ap:.4f})")
    ax.axhline(y.mean(), color="gray", ls=":", label=f"базова точність = {y.mean():.4f}")
    for label, t, color in marks:
        tp, fp, fn, _ = confusion(y, s, t)
        if precision(tp, fp) is None:
            continue
        ax.scatter([recall(tp, fn)], [precision(tp, fp)], s=60, color=color,
                   zorder=3, label=f"{label}: t = {fmt(t, 4)}")
    ax.set_xlabel("Повнота (recall)"); ax.set_ylabel("Точність (precision)")
    ax.set_title("PR-крива, validation"); ax.grid(alpha=.3); ax.legend(loc="lower left")
    fig.tight_layout(); fig.savefig(out / "pr_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=1.5, label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", ls=":", label="випадкове вгадування")
    for label, t, color in marks:
        tp, fp, fn, tn = confusion(y, s, t)
        ax.scatter([fp / (fp + tn)], [recall(tp, fn)], s=60, color=color,
                   zorder=3, label=f"{label}: t = {fmt(t, 4)}")
    ax.set_xlabel("FPR = FP / (FP + TN)"); ax.set_ylabel("TPR = recall")
    ax.set_title("ROC-крива, validation"); ax.grid(alpha=.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(out / "roc_curve.png", dpi=150); plt.close(fig)
    return auc, ap


# ----------------------------------------------------------------------------
def verify_small_sets(lines: list[str]) -> bool:
    all_ok = True
    for name, (s_list, y_list) in SMALL_SETS.items():
        fast = candidate_table(np.array(y_list), np.array(s_list))
        slow = brute_force_table(y_list, s_list)
        t_fast, t_slow = choose_threshold(fast), brute_force_choice(slow)
        t_exp, c_exp = EXPECTED[name]

        same_table = (
            fast.shape == slow.shape
            and np.array_equal(fast["threshold"].to_numpy(), slow["threshold"].to_numpy())
            and all(np.array_equal(fast[c].to_numpy(), slow[c].to_numpy())
                    for c in ("TP", "FP", "FN", "TN", "C"))
        )
        c_min = int(fast["C"].min())
        ok = same_table and t_fast == t_slow == t_exp and c_min == c_exp
        all_ok &= ok

        view = slow.copy()
        view["threshold"] = view["threshold"].map(lambda v: fmt(float(v), 6))
        view["збіг зі швидким"] = [
            "так" if np.array_equal(fast.iloc[i].to_numpy(), slow.iloc[i].to_numpy()) else "НІ"
            for i in range(len(slow))
        ]
        lines += [
            f"### Набір {name}", "",
            f"Оцінки: `{s_list}`  ", f"Мітки: `{y_list}`", "",
            md_table(view), "",
            f"- таблиці (кандидати та всі лічильники) збігаються точно: **{'так' if same_table else 'НІ'}**",
            f"- t* швидкий / перебір / очікуваний вручну: {fmt(t_fast)} / {fmt(t_slow)} / {fmt(t_exp)}",
            f"- мінімальна C: {c_min} (очікувано {c_exp}); перевірку пройдено: **{'так' if ok else 'НІ'}**", "",
        ]
    return all_ok


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saved", type=Path, required=True, help="тека з predictions.npz")
    ap.add_argument("--out", type=Path, default=None,
                    help="куди писати результати (за замовчуванням <saved>/analysis)")
    args = ap.parse_args()
    out = args.out or (args.saved / "analysis")
    out.mkdir(parents=True, exist_ok=True)

    data = load_predictions(args.saved)
    report: list[str] = []
    summary: dict = {}

    # --- розмір вибірок --------------------------------------------------
    sp = split_summary(data)
    sp_view = sp.copy()
    sp_view["positive_fraction"] = sp_view["positive_fraction"].map(lambda v: f"{v:.6f} ({v*100:.4f}%)")
    report += ["## Поділ", "", md_table(sp_view), ""]
    print(sp.to_string(index=False))

    # --- п. 2: validation ------------------------------------------------
    yv = data["validation_y"].astype(np.int64)
    sv = data["validation_scores"].astype(np.float64)   # float32 -> float64 точно
    table = candidate_table(yv, sv)
    table.to_csv(out / "validation_candidates.csv", index=False)
    t_star = choose_threshold(table)
    n_tied = int((table["C"] == table["C"].min()).sum())

    # t* має бути рядком таблиці; перехресна перевірка зі «слабким» підрахунком
    row = table[table["threshold"] == t_star].iloc[0]
    assert (int(row.TP), int(row.FP), int(row.FN), int(row.TN)) == confusion(yv, sv, t_star)

    rules = pd.DataFrame([
        row_for_rule("Стандартний поріг", yv, sv, 0.5),
        row_for_rule("Мінімум вартості", yv, sv, t_star),
        row_for_rule("Усі негативні", yv, sv, float("inf")),
    ])
    report += ["## Validation: порівняння правил", "", md_table(rules), "",
               f"Кандидатів у таблиці: {len(table)}; рядків з мінімальною вартістю: {n_tied}.", ""]
    print(rules.to_string(index=False))

    auc, ap_score = plot_curves(
        yv, sv,
        [("стандартний", 0.5, "tab:orange"), ("мінімум C (t*)", t_star, "tab:red")], out)
    report += [f"ROC-AUC (validation) = {auc:.6f}; Average precision = {ap_score:.6f}", ""]
    print(f"t* = {t_star!r}; ROC-AUC = {auc:.6f}; AP = {ap_score:.6f}")

    (out / "threshold.json").write_text(json.dumps({
        "t_star": t_star, "rule": "predict fraud if score >= t_star",
        "selected_on": "validation", "validation_min_cost": int(table["C"].min()),
        "tied_minimum_rows": n_tied}, indent=2) + "\n", encoding="utf-8")

    # --- п. 3: перевірка пошуку --------------------------------------------
    report += ["## Перевірка пошуку порога (набори A/B/C)", ""]
    if not verify_small_sets(report):
        (out / "report_tables.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        raise SystemExit("ПЕРЕВІРКА ПОШУКУ НЕ ПРОЙДЕНА - тестова оцінка не виконується.")
    print("Перевірки A/B/C пройдено.")

    # --- п. 4: test ---------------------------------------------------------
    yt = data["test_y"].astype(np.int64)
    st = data["test_scores"].astype(np.float64)
    rid = data["test_row_ids"]
    amt = data["test_amount"]
    tp, fp, fn, tn = confusion(yt, st, t_star)
    prec, rec = precision(tp, fp), recall(tp, fn)
    c_test = cost(fn, fp)
    test_rules = pd.DataFrame([
        row_for_rule("Стандартний поріг", yt, st, 0.5),
        row_for_rule("t* (зафіксований на validation)", yt, st, t_star),
        row_for_rule("Усі негативні", yt, st, float("inf")),
    ])
    report += ["## Test (поріг t* зафіксовано на validation)", "", md_table(test_rules), "",
               f"precision = {fmt(prec, 6)}"
               + ("" if prec is not None else " (TP+FP = 0: жодної тривоги)")
               + f"; recall = {fmt(rec, 6)}"
               + ("" if rec is not None else " (TP+FN = 0: у test немає шахрайств)"), ""]
    print(test_rules.to_string(index=False))

    pred = st >= t_star
    kinds = np.where(pred & (yt == 0), "FP", np.where(~pred & (yt == 1), "FN", ""))
    def logit(v):                         # відстань у логітах: шкала нейромережі
        v = np.clip(v, 1e-12, 1 - 1e-12)
        return np.log(v / (1 - v))

    err = pd.DataFrame({"row": rid, "type": kinds, "label": yt, "score": st,
                        "amount": amt, "score_minus_t": st - t_star,
                        "logit_minus_logit_t": logit(st) - logit(t_star)})
    err = err[err["type"] != ""].sort_values("row")
    err.to_csv(out / "test_errors_all.csv", index=False)
    first = pd.concat([err[err["type"] == "FP"].head(5), err[err["type"] == "FN"].head(5)])
    first.to_csv(out / "test_errors_first5.csv", index=False)
    view = first.copy()
    view["score"] = view["score"].map(lambda v: f"{v:.6g}")
    view["amount"] = view["amount"].map(lambda v: f"{v:.2f}")
    view["score_minus_t"] = view["score_minus_t"].map(lambda v: f"{v:+.4g}")
    view["logit_minus_logit_t"] = view["logit_minus_logit_t"].map(lambda v: f"{v:+.2f}")
    view.columns = ["Рядок CSV", "Тип", "Мітка", "Оцінка", "Amount", "s − t*", "logit(s) − logit(t*)"]
    report += ["## Перші п'ять FP і перші п'ять FN у test", "", md_table(view), ""]

    stats = []
    for kind in ("FP", "FN"):
        e = err[err["type"] == kind]
        stats.append({
            "Тип": kind, "кількість": len(e),
            "медіана Amount": f"{e['amount'].median():.2f}" if len(e) else "-",
            "середнє Amount": f"{e['amount'].mean():.2f}" if len(e) else "-",
            "медіана |s − t*|": f"{e['score_minus_t'].abs().median():.4g}" if len(e) else "-",
            "макс |s − t*|": f"{e['score_minus_t'].abs().max():.4g}" if len(e) else "-",
            "медіана |Δlogit|": f"{e['logit_minus_logit_t'].abs().median():.2f}" if len(e) else "-",
        })
    base = {"Тип": "усі test", "кількість": len(yt),
            "медіана Amount": f"{np.median(amt):.2f}", "середнє Amount": f"{amt.mean():.2f}",
            "медіана |s − t*|": "-", "макс |s − t*|": "-", "медіана |Δlogit|": "-"}
    stats.append(base)
    report += ["### Зведення за всіма помилками test", "", md_table(pd.DataFrame(stats)), ""]
    print(md_table(view)); print(md_table(pd.DataFrame(stats)))

    summary.update({
        "t_star": t_star, "roc_auc_validation": auc, "ap_validation": ap_score,
        "test": {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "C": c_test,
                 "precision": prec, "recall": rec}})
    (out / "results_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (out / "report_tables.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nУсі результати записано в {out.resolve()}")


if __name__ == "__main__":
    main()
