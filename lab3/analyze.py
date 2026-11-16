"""ЛР3, команда аналізу: журнали -> таблиця, криві, ефекти, перевірка вибору,
одноразова тестова оцінка lab3_model.pt на CPU, матриця плутанини, п'ять помилок.

    uv run python analyze.py --data /шлях/до/intel-image-classification

Єдине місце, де завантажується test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix

import common as C

COLORS = {"scratch": "tab:gray", "frozen": "tab:blue", "unfreeze": "tab:red"}
STYLES = {1: "-", 2: "--", 3: ":"}


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


def load_logs() -> dict[tuple[str, int], dict]:
    logs = {}
    for regime, seed in C.RUN_PLAN:
        p = Path("runs") / C.run_name(regime, seed) / "run.json"
        if not p.exists():
            raise SystemExit(f"Немає журналу {p}: спершу виконайте команду навчання.")
        logs[(regime, seed)] = json.loads(p.read_text(encoding="utf-8"))
    return logs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="тека з seg_train/ і seg_test/")
    ap.add_argument("--out", type=Path, default=Path("analysis"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: list[str] = []

    # --- 0. версія даних ---------------------------------------------------------
    logs = load_logs()
    cur_hash = C.sha256_file(Path("split.csv"))
    hashes = {k: v["split_sha256"] for k, v in logs.items()}
    if len(set(hashes.values())) != 1 or set(hashes.values()) != {cur_hash}:
        raise SystemExit("ХЕШ split.csv не збігається у журналах і/або з поточним split.csv. "
                         "Запуски на різних поділах порівнювати не можна - перенавчіть усі сім.")
    print("Хеш split.csv однаковий у всіх семи журналах і збігається з поточним файлом.")
    report += [f"SHA-256 `split.csv` (однаковий у всіх 7 журналах і в поточному файлі): `{cur_hash}`", ""]
    commits = {(v["git"]["commit"], v["git"]["dirty"]) for v in logs.values()}
    report += [f"Хеші комітів у журналах: {sorted(map(str, commits))}", ""]
    for k, v in logs.items():
        s = v["settings"]
        if s["epochs"] != C.EPOCHS or s["train_per_class"] != C.TRAIN_PER_CLASS or s["skip_split_control"]:
            print(f"УВАГА: запуск {k} виконано з нестандартними налаштуваннями (димовий тест?): "
                  f"epochs={s['epochs']}, train_per_class={s['train_per_class']}, skip_split_control={s['skip_split_control']}")

    # --- 3. таблиця семи запусків ------------------------------------------------------
    F = {k: v["final_val_macro_f1"] for k, v in logs.items()}
    rows = []
    for i, (regime, seed) in enumerate(C.RUN_PLAN, 1):
        v = logs[(regime, seed)]
        rows.append({"#": i, "Режим": regime, "Зерно": seed,
                     "Втрата train, епоха 8": f"{v['final_train_loss']:.4f}",
                     "macro-F1 validation, епоха 8": f"{v['final_val_macro_f1']:.4f}",
                     "Час навчання, с": f"{v['train_seconds']:.1f}"})
    report += ["## Сім запусків", "", md_table(rows), ""]
    print(md_table(rows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for (regime, seed), v in logs.items():
        ep = np.arange(1, len(v["epoch_train_loss"]) + 1)
        kw = dict(color=COLORS[regime], ls=STYLES[seed], lw=1.6, label=f"{regime}, зерно {seed}")
        axes[0].plot(ep, v["epoch_train_loss"], **kw)
        axes[1].plot(ep, v["epoch_val_macro_f1"], **kw)
    axes[0].set_title("Втрата на train"); axes[0].set_ylabel("CrossEntropy (середнє за епоху)")
    axes[1].set_title("macro-F1 на validation"); axes[1].set_ylabel("macro-F1")
    for ax in axes:
        ax.set_xlabel("Епоха"); ax.set_xticks(range(1, C.EPOCHS + 1)); ax.grid(alpha=.3)
    axes[1].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(args.out / "curves.png", dpi=140); plt.close(fig)

    # --- 4. розклад ефектів ------------------------------------------------------------
    fro = [F[("frozen", s)] for s in (1, 2, 3)]
    unf = [F[("unfreeze", s)] for s in (1, 2, 3)]
    F_fro, F_unf, F_scr = float(np.mean(fro)), float(np.mean(unf)), F[("scratch", 1)]
    R_fro, R_unf = max(fro) - min(fro), max(unf) - min(unf)
    R = max(R_fro, R_unf)
    d_pre, d_unf = F_fro - F_scr, F_unf - F_fro

    def verdict(d):
        return "|Δ| ≤ R: не відрізняється від розкиду (результату немає)" if abs(d) <= R \
            else "|Δ| > R: ефект вартий перевірки, але не доведений"

    eff = [
        {"Величина": "F̄_frozen", "Значення": f"{F_fro:.4f}"},
        {"Величина": "F̄_unfreeze", "Значення": f"{F_unf:.4f}"},
        {"Величина": "F_scratch,1", "Значення": f"{F_scr:.4f}"},
        {"Величина": "R_frozen", "Значення": f"{R_fro:.4f}"},
        {"Величина": "R_unfreeze", "Значення": f"{R_unf:.4f}"},
        {"Величина": "R = max(R_frozen, R_unfreeze)", "Значення": f"{R:.4f}"},
        {"Величина": "Δ_pre = F̄_frozen − F_scratch,1", "Значення": f"{d_pre:+.4f}  ({verdict(d_pre)})"},
        {"Величина": "Δ_unf = F̄_unfreeze − F̄_frozen", "Значення": f"{d_unf:+.4f}  ({verdict(d_unf)})"},
        {"Величина": "Δ_pre + Δ_unf = F̄_unfreeze − F_scratch,1", "Значення": f"{d_pre + d_unf:+.4f}  (перевірка: {F_unf - F_scr:+.4f})"},
    ]
    report += ["## Розклад ефектів", "", md_table(eff), ""]
    print(md_table(eff))

    # час навчання по режимах (для питання про бюджет)
    tim = []
    for regime in C.REGIMES:
        ts = [v["train_seconds"] for (r, _), v in logs.items() if r == regime]
        tim.append({"Режим": regime, "середній час навчання, с": f"{np.mean(ts):.1f}", "мін–макс, с": f"{min(ts):.1f}–{max(ts):.1f}"})
    report += ["Час навчання за режимами:", "", md_table(tim), ""]
    print(md_table(tim))

    # --- 5. вибір і перевірка ---------------------------------------------------------------
    regime, seed, means = C.select_run(F)
    ck = torch.load("lab3_model.pt", map_location="cpu")
    if (ck["regime"], ck["seed"]) != (regime, seed) or ck["classes"] != C.CLASSES or ck["img_size"] != C.IMG_SIZE:
        raise SystemExit(f"lab3_model.pt ({ck['regime']}, {ck['seed']}) не відповідає правилу вибору ({regime}, {seed}).")
    sel_msg = (f"Вибір за validation: режим **{regime}** (середні F1: "
               + ", ".join(f"{k} {v:.4f}" for k, v in means.items())
               + f"), зерно **{seed}** (F1 = {F[(regime, seed)]:.4f}). `lab3_model.pt` відповідає правилу: так.")
    if abs(d_unf) <= R:
        sel_msg += " Оскільки |Δ_unf| ≤ R, вибір між frozen і unfreeze визначив шум."
    report += ["## Вибір конфігурації", "", sel_msg, ""]
    print(sel_msg)

    # --- 5. тест: один раз, CPU -----------------------------------------------------------------
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rows_all = C.read_split(Path("split.csv"))
    model = torchvision.models.resnet18(weights=None)
    model.fc = torch.nn.Linear(512, 6)
    model.load_state_dict(ck["state_dict"])
    model.to("cpu").eval()

    val_rows = C.rows_of(rows_all, "validation")
    val_ds = C.SplitDataset(val_rows, args.data, C.eval_transform())
    val_logits = C.predict_logits(model, C.make_eval_loader(val_ds), torch.device("cpu"))
    saved = np.load(Path("runs") / C.run_name(regime, seed) / "val_logits.npy")
    vdiff = float(np.abs(val_logits - saved).max())
    f1_val_cpu = C.macro_f1(val_ds.targets, val_logits)
    report += [f"Контроль: логіти validation цієї моделі на CPU відрізняються від збережених на {vdiff:.2e} (макс.), "
               f"macro-F1 validation на CPU = {f1_val_cpu:.4f}.", ""]
    print(f"validation на CPU: max|Δlogit|={vdiff:.2e}, F1={f1_val_cpu:.4f}")

    test_rows = C.rows_of(rows_all, "test")
    test_ds = C.SplitDataset(test_rows, args.data, C.eval_transform())
    logits = C.predict_logits(model, C.make_eval_loader(test_ds), torch.device("cpu"))
    y = np.array(test_ds.targets)
    pred = logits.argmax(1)
    f1_test = C.macro_f1(y, logits)
    acc = float((pred == y).mean())
    n = len(y)
    se = float(np.sqrt(acc * (1 - acc) / n))
    f1_val = F[(regime, seed)]
    cm = confusion_matrix(y, pred, labels=list(range(6)))

    res = [{"Величина": "macro-F1 test", "Значення": f"{f1_test:.4f}"},
           {"Величина": "accuracy test (p)", "Значення": f"{acc:.4f}"},
           {"Величина": f"SE = sqrt(p(1−p)/n), n = {n}", "Значення": f"{se:.4f}"},
           {"Величина": "macro-F1 validation вибраного запуску (епоха 8)", "Значення": f"{f1_val:.4f}"},
           {"Величина": "macro-F1 validation − macro-F1 test", "Значення": f"{f1_val - f1_test:+.4f}"}]
    report += ["## Test (одноразово, CPU)", "", md_table(res), ""]
    print(md_table(res))

    cm_rows = [{"справжній \\ передбачений": C.CLASSES[i], **{C.CLASSES[j]: int(cm[i, j]) for j in range(6)}} for i in range(6)]
    report += ["Матриця плутанини (рядки - справжній клас, стовпці - передбачений):", "", md_table(cm_rows), ""]
    print(md_table(cm_rows))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6), C.CLASSES, rotation=45, ha="right"); ax.set_yticks(range(6), C.CLASSES)
    ax.set_xlabel("Передбачений клас"); ax.set_ylabel("Справжній клас")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    ax.set_title(f"Матриця плутанини, test (macro-F1 = {f1_test:.4f})")
    fig.colorbar(im, ax=ax, fraction=0.046); fig.tight_layout()
    fig.savefig(args.out / "confusion_matrix.png", dpi=140); plt.close(fig)

    # --- п'ять характерних помилок ---------------------------------------------------------------
    pairs = [(int(cm[i, j]), i, j) for i in range(6) for j in range(6) if i != j and cm[i, j] > 0]
    pairs.sort(key=lambda t: (-t[0], t[1], t[2]))          # частота спадає; потім індекси зростають
    files_by_pair = {}
    for (cnt, i, j) in pairs:
        idx = [k for k in range(n) if y[k] == i and pred[k] == j]
        files_by_pair[(i, j)] = sorted(idx, key=lambda k: test_rows[k][0])   # за іменем як рядком
    chosen = []
    top = pairs[:5]
    for (cnt, i, j) in top:
        chosen.append((i, j, files_by_pair[(i, j)][0], cnt))
    rank = 1
    while len(chosen) < 5 and pairs:
        progressed = False
        for (cnt, i, j) in pairs:
            lst = files_by_pair[(i, j)]
            if len(lst) > rank and len(chosen) < 5:
                chosen.append((i, j, lst[rank], cnt)); progressed = True
        if not progressed:
            break
        rank += 1
    probs = np.exp(logits - logits.max(1, keepdims=True)).astype(np.float64)
    probs /= probs.sum(1, keepdims=True)
    err_rows = []
    for (i, j, k, cnt) in chosen:
        err_rows.append({"Файл": test_rows[k][0], "Справжній клас": C.CLASSES[i],
                         "Передбачений клас": C.CLASSES[j],
                         "Імовірність передбаченого класу": f"{probs[k, j]:.4f}",
                         "Кількість таких помилок у test": cnt})
    report += ["## П'ять характерних помилок", "", md_table(err_rows), ""]
    print(md_table(err_rows))
    fig, axes = plt.subplots(1, len(chosen), figsize=(3.4 * len(chosen), 3.9))
    axes = np.atleast_1d(axes)
    for ax, (i, j, k, cnt), r in zip(axes, chosen, err_rows):
        f, label, part = test_rows[k]
        with Image.open(C.part_dir(args.data, part, label) / f) as im_:
            ax.imshow(im_.convert("RGB"))
        ax.set_title(f"{f}\nсправжній: {C.CLASSES[i]}\nпрогноз: {C.CLASSES[j]} (p={probs[k, j]:.2f})", fontsize=8)
        ax.axis("off")
    fig.tight_layout(); fig.savefig(args.out / "errors.png", dpi=140); plt.close(fig)

    (args.out / "report_tables.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps({
        "F_frozen_mean": F_fro, "F_unfreeze_mean": F_unf, "F_scratch1": F_scr,
        "R_frozen": R_fro, "R_unfreeze": R_unf, "R": R, "delta_pre": d_pre, "delta_unf": d_unf,
        "selected_regime": regime, "selected_seed": seed, "f1_val_selected": f1_val,
        "f1_test": f1_test, "accuracy_test": acc, "se_accuracy": se,
        "confusion_matrix": cm.tolist()}, indent=2) + "\n", encoding="utf-8")
    print(f"\nРезультати записано в {args.out.resolve()}")


if __name__ == "__main__":
    main()
