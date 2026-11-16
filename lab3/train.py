"""ЛР3, команда навчання: поділ, перевірки пп. 1-2, сім запусків, журнали, вибір.

    uv run python train.py --data /шлях/до/intel-image-classification

--data - тека, у якій лежать seg_train/seg_train/... і seg_test/seg_test/...
Тест у цій команді не завантажується й не використовується.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import common as C


def md_table(rows: list[dict]) -> str:
    cols = list(rows[0].keys())
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Перевірки до навчання (п. 1)
# ----------------------------------------------------------------------------
def save_batch_figure(train_ds, path: Path):
    torch.manual_seed(1)
    x, y = next(iter(C.make_train_loader(train_ds)))
    mean = torch.tensor(C.MEAN).view(3, 1, 1)
    std = torch.tensor(C.STD).view(3, 1, 1)
    fig, axes = plt.subplots(4, 4, figsize=(8, 8.8))
    for i, ax in enumerate(axes.ravel()):
        img = (x[i] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()   # обернене нормування
        ax.imshow(img)
        ax.set_title(C.CLASSES[int(y[i])], fontsize=9)
        ax.axis("off")
    fig.suptitle("Перші 16 зображень навчального пакета (після аугментацій)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


@torch.no_grad()
def initial_loss(regime, device, train_ds) -> float:
    model, _, _ = C.build_model(regime, 1, device)
    loader = C.make_train_loader(train_ds)
    x, y = next(iter(loader))
    C.set_train_mode(model, regime)
    # цей екземпляр далі не використовується: прямий прохід у train-режимі оновлює BN
    return torch.nn.CrossEntropyLoss()(model(x.to(device)), y.to(device)).item()


# ----------------------------------------------------------------------------
# Перевірки режимів (п. 2)
# ----------------------------------------------------------------------------
def param_check(regime, device):
    model, opt, groups = C.build_model(regime, 1, device)
    total, trainable = C.count_params(model)
    exp_total, exp_train = C.EXPECTED_PARAMS[regime]
    in_opt = {id(p) for g in opt.param_groups for p in g["params"]}
    leak = any((not p.requires_grad) and id(p) in in_opt for p in model.parameters())
    opt_params = sum(g["n_params"] for g in groups)
    ok = (total == exp_total and trainable == exp_train and opt_params == trainable and not leak)
    return {"Режим": regime, "Усього параметрів": f"{total:,}".replace(",", " "),
            "requires_grad=True": f"{trainable:,}".replace(",", " "),
            "Групи оптимізатора": "; ".join(f"{g['name']}: {g['n_params']:,} @ {g['lr']:g}".replace(",", " ") for g in groups),
            "Заморожене в оптимізаторі": "ні" if not leak else "ТАК",
            "Збігається з таблицею": "так" if ok else "НІ"}, ok


_STAT = ("running_mean", "running_var", "num_batches_tracked")


def state_change_check(regime, device, train_ds, proper=True):
    """5 кроків на перших 5 пакетах; порівняння state_dict до/після (torch.equal)."""
    model, opt, _ = C.build_model(regime, 1, device)
    loader = C.make_train_loader(train_ds)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if proper:
        C.set_train_mode(model, regime)
    else:
        model.train()
    crit = torch.nn.CrossEntropyLoss()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i == 5:
            break
        opt.zero_grad()
        loss = crit(model(x.to(device)), y.to(device))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    after = model.state_dict()
    rows, all_ok = [], True
    for m in C.TOP_MODULES:
        pk = [k for k in before if k.startswith(m + ".") and not k.endswith(_STAT)]
        sk = [k for k in before if k.startswith(m + ".") and k.endswith(_STAT)]
        p_ch = any(not torch.equal(before[k], after[k]) for k in pk)
        s_ch = any(not torch.equal(before[k], after[k]) for k in sk) if sk else None
        frozen = m in C.EVAL_MODULES[regime]
        if frozen:
            ok = (not p_ch) and (s_ch in (False, None))
        else:
            ok = p_ch and (s_ch in (True, None))
        all_ok &= ok
        rows.append({"Режим": regime, "Модуль": m,
                     "Параметри змінилися": "так" if p_ch else "ні",
                     "Ковзні статистики BN змінилися": "немає BN" if s_ch is None else ("так" if s_ch else "ні"),
                     "Збігається з очікуваним": "так" if ok else "ні"})
    bn = {}
    for stat in ("running_mean", "running_var"):
        k = f"layer1.0.bn1.{stat}"
        bn[stat] = float((after[k].float() - before[k].float()).abs().max())
    nbt = int(after["layer1.0.bn1.num_batches_tracked"]) - int(before["layer1.0.bn1.num_batches_tracked"])
    return rows, all_ok, bn, nbt, losses


# ----------------------------------------------------------------------------
# Запуски
# ----------------------------------------------------------------------------
def settings_dict(args) -> dict:
    bn = torch.nn.BatchNorm2d(1)
    opt_defaults = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1.0, weight_decay=C.WEIGHT_DECAY).defaults
    return {
        "transforms_train": "RandomResizedCrop(160, scale=(0.7, 1.0)) -> RandomHorizontalFlip() -> ToTensor() -> Normalize(mean, std)",
        "transforms_eval": "Resize((160, 160)) -> ToTensor() -> Normalize(mean, std)",
        "mean": C.MEAN, "std": C.STD, "img_size": C.IMG_SIZE,
        "model": "torchvision resnet18, fc = Linear(512, 6) стандартної ініціалізації",
        "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1 (frozen, unfreeze); None (scratch)",
        "trainable": C.TRAINABLE_PREFIXES, "eval_modules": C.EVAL_MODULES,
        "lr_head": C.LR_HEAD, "lr_backbone_layer4": C.LR_BACKBONE,
        "optimizer": "AdamW", "optimizer_defaults": {k: (list(v) if isinstance(v, tuple) else v) for k, v in opt_defaults.items() if k != "lr"},
        "weight_decay": C.WEIGHT_DECAY,
        "batchnorm": {"momentum": bn.momentum, "eps": bn.eps},
        "criterion": "CrossEntropyLoss(), середнє за пакетом, без вагових коефіцієнтів, без згладжування",
        "train_loader": {"batch_size": C.BATCH_TRAIN, "shuffle": True, "drop_last": False, "num_workers": 0,
                         "generator/sampler": "немає (глобальний генератор PyTorch)"},
        "eval_loader": {"batch_size": C.BATCH_EVAL, "shuffle": False, "num_workers": 0},
        "epochs": C.EPOCHS, "train_per_class": args.train_per_class,
        "steps_per_epoch_last_batch_note": "останній неповний пакет не відкидається",
        "dtype": "float32", "lr_schedule": None, "grad_clip": None, "amp": False,
        "label_smoothing": 0.0, "class_weights": None, "early_stopping": False,
        "skip_split_control": bool(args.skip_split_control),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="тека з seg_train/ і seg_test/")
    ap.add_argument("--device", default=None, help="cpu / cuda / mps (за замовчуванням - найкращий доступний)")
    ap.add_argument("--resume", action="store_true", help="пропустити вже завершені запуски (той самий split.csv)")
    # Лише для димового тесту; для здачі не використовувати (фіксовані значення з умови):
    ap.add_argument("--epochs", type=int, default=C.EPOCHS, help=argparse.SUPPRESS)
    ap.add_argument("--train-per-class", type=int, default=C.TRAIN_PER_CLASS, help=argparse.SUPPRESS)
    ap.add_argument("--skip-split-control", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    C.EPOCHS = args.epochs

    device = C.pick_device(args.device)
    print(f"Пристрій: {C.device_name(device)}")
    report: list[str] = []

    # --- поділ ---------------------------------------------------------------
    rows = C.build_split(args.data, args.train_per_class)
    split_path = Path("split.csv")
    C.write_split(rows, split_path)
    split_hash = C.sha256_file(split_path)
    table, split_ok = C.split_control(rows)
    disjoint = C.check_disjoint(rows)
    report += ["## 1. Контроль поділу (кількість / сума числових імен)", "", md_table(table), "",
               f"Усі 18 клітинок і підсумки збігаються з умовою: **{'так' if split_ok else 'НІ'}**; "
               f"жоден файл не входить у дві частини (імена унікальні): **{'так' if disjoint else 'НІ'}**.",
               f"SHA-256 `split.csv`: `{split_hash}`", ""]
    print(md_table(table))
    print("Поділ збігається з умовою:", split_ok, "| без перетинів:", disjoint)
    if not disjoint:
        raise SystemExit("Один файл потрапив у кілька частин - помилка поділу.")
    if not split_ok and not args.skip_split_control:
        raise SystemExit("Контрольна таблиця не збіглася - знайдіть помилку поділу перед подальшими кроками.")

    train_ds = C.SplitDataset(C.rows_of(rows, "train"), args.data, C.train_transform())
    val_ds = C.SplitDataset(C.rows_of(rows, "validation"), args.data, C.eval_transform())
    # Тест тут НЕ завантажується.

    # --- пакет після перетворень -------------------------------------------------
    save_batch_figure(train_ds, Path("figures/train_batch.png"))
    report += ["Рисунок пакета після перетворень: `figures/train_batch.png`.", ""]

    # --- початкова втрата --------------------------------------------------------
    ln6 = math.log(6)
    il = []
    for regime in ("scratch", "frozen"):
        L0 = initial_loss(regime, device, train_ds)
        il.append({"Режим": regime, "L0 (перший пакет, 0 кроків)": f"{L0:.4f}", "ln 6": f"{ln6:.4f}",
                   "L0 − ln 6": f"{L0 - ln6:+.4f}"})
    report += ["## 1. Початкова втрата", "", md_table(il), ""]
    print(md_table(il))

    # --- параметри та групи ------------------------------------------------------
    prs, ok_all = [], True
    for regime in C.REGIMES:
        row, ok = param_check(regime, device)
        prs.append(row)
        ok_all &= ok
    report += ["## 2. Параметри й групи оптимізатора", "", md_table(prs), ""]
    print(md_table(prs))

    # --- що змінюється під час навчання -------------------------------------------
    sc_rows = []
    losses_ok = {}
    for regime in ("frozen", "unfreeze"):
        r, ok, _, _, ls = state_change_check(regime, device, train_ds, proper=True)
        sc_rows += r
        ok_all &= ok
        losses_ok[regime] = ls
    report += ["## 2. Що змінюється за 5 кроків (set_train_mode)", "", md_table(sc_rows), ""]
    print(md_table(sc_rows))

    # --- дослід із навмисною помилкою ----------------------------------------------
    bad_rows, bad_ok, bn, nbt, bad_losses = state_change_check("frozen", device, train_ds, proper=False)
    detected = (bn["running_mean"] > 0 and bn["running_var"] > 0 and nbt == len(bad_losses))
    report += ["## 2. Дослід із навмисною помилкою (frozen, звичайний model.train())", "",
               md_table(bad_rows), "",
               f"Усі рядки збігаються з очікуваним: **{'так' if bad_ok else 'ні'}** (очікується «ні»: помилку виявлено = **{'так' if detected else 'НІ'}**).",
               f"layer1.0.bn1 після 5 кроків: max|Δrunning_mean| = {bn['running_mean']:.6e}; "
               f"max|Δrunning_var| = {bn['running_var']:.6e}; num_batches_tracked +{nbt}.",
               f"Втрати на 5 кроках, правильно: {[round(v, 4) for v in losses_ok['frozen']]}; "
               f"з model.train(): {[round(v, 4) for v in bad_losses]}.", ""]
    print(md_table(bad_rows))
    print(f"layer1.0.bn1: dmean={bn['running_mean']:.3e}, dvar={bn['running_var']:.3e}, dNBT={nbt}")

    Path("checks").mkdir(exist_ok=True)
    Path("checks/report_checks.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    if not (ok_all and detected):
        raise SystemExit("Перевірки пп. 1-2 не збіглися з очікуваним - основний експеримент не запускається.")
    print("Перевірки пп. 1-2 пройдено.\n")

    # --- сім запусків ---------------------------------------------------------------
    git = C.git_info()
    env = C.env_info(device)
    settings = settings_dict(args)
    Path("runs").mkdir(exist_ok=True)
    finals: dict[tuple[str, int], float] = {}
    for regime, seed in C.RUN_PLAN:
        d = Path("runs") / C.run_name(regime, seed)
        d.mkdir(parents=True, exist_ok=True)
        rj = d / "run.json"
        if args.resume and rj.exists() and (d / "weights.pt").exists():
            prev = json.loads(rj.read_text(encoding="utf-8"))
            if prev.get("split_sha256") == split_hash:
                print(f"== {regime} seed={seed}: уже завершено, пропуск")
                finals[(regime, seed)] = prev["final_val_macro_f1"]
                continue
        print(f"== запуск {regime} seed={seed}")
        model, logits, losses, f1s, tsec, total_sec, groups = C.run_one(regime, seed, device, train_ds, val_ds)
        np.save(d / "val_logits.npy", logits.astype(np.float32))
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, d / "weights.pt")
        log = {"regime": regime, "seed": seed, "split_sha256": split_hash, "git": git, "env": env,
               "settings": settings, "optimizer_groups": groups,
               "epoch_train_loss": losses, "epoch_val_macro_f1": f1s,
               "final_train_loss": losses[-1], "final_val_macro_f1": f1s[-1],
               "train_seconds": tsec, "total_seconds_with_validation": total_sec}
        rj.write_text(json.dumps(log, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        finals[(regime, seed)] = f1s[-1]
        del model

    # --- вибір конфігурації (лише за validation) -----------------------------------------
    regime, seed, means = C.select_run(finals)
    print(f"Вибір: режим {regime}, зерно {seed}; середні F1: { {k: round(v, 4) for k, v in means.items()} }")
    sd = torch.load(Path("runs") / C.run_name(regime, seed) / "weights.pt", map_location="cpu")
    torch.save({"state_dict": sd, "regime": regime, "seed": seed,
                "classes": C.CLASSES, "img_size": C.IMG_SIZE}, "lab3_model.pt")
    print("Збережено lab3_model.pt. Test не використовувався.")


if __name__ == "__main__":
    main()
