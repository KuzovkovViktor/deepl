"""ЛР3: спільний код (поділ, дані, модель, режими навчання, оцінка)."""
from __future__ import annotations

import csv
import hashlib
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

# ----------------------------------------------------------------------------
# Фіксовані налаштування з умови
# ----------------------------------------------------------------------------
CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 160
TRAIN_PER_CLASS = 200
BATCH_TRAIN = 32
BATCH_EVAL = 64
EPOCHS = 8
WEIGHT_DECAY = 1e-4
LR_HEAD = 3e-3        # fc (у всіх режимах) і вся мережа в scratch
LR_BACKBONE = 3e-4    # layer4 у режимі unfreeze
REGIMES = ("scratch", "frozen", "unfreeze")
RUN_PLAN = [("scratch", 1), ("frozen", 1), ("frozen", 2), ("frozen", 3),
            ("unfreeze", 1), ("unfreeze", 2), ("unfreeze", 3)]

# Модулі, які мають бути в eval() під час навчання (стовпець «У eval()»)
EVAL_MODULES = {
    "scratch": [],
    "frozen": ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"],
    "unfreeze": ["conv1", "bn1", "layer1", "layer2", "layer3"],
}
# Що навчається (requires_grad=True)
TRAINABLE_PREFIXES = {
    "scratch": None,                 # усе
    "frozen": ("fc.",),
    "unfreeze": ("layer4.", "fc."),
}
TOP_MODULES = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"]

EXPECTED_PARAMS = {  # (усього, навчається)
    "scratch": (11_179_590, 11_179_590),
    "frozen": (11_179_590, 3_078),
    "unfreeze": (11_179_590, 8_396_806),
}

# Контрольна таблиця поділу: (клас, частина) -> (кількість, сума числових імен)
EXPECTED_SPLIT = {
    ("buildings", "train"): (200, 1996761), ("buildings", "validation"): (218, 4837323),
    ("buildings", "test"): (219, 4837559),
    ("forest", "train"): (200, 2032625), ("forest", "validation"): (237, 5276962),
    ("forest", "test"): (237, 5291326),
    ("glacier", "train"): (200, 2183923), ("glacier", "validation"): (276, 6135652),
    ("glacier", "test"): (277, 6157250),
    ("mountain", "train"): (200, 1950345), ("mountain", "validation"): (262, 5789535),
    ("mountain", "test"): (263, 5849765),
    ("sea", "train"): (200, 1987613), ("sea", "validation"): (255, 5655068),
    ("sea", "test"): (255, 5668011),
    ("street", "train"): (200, 1841817), ("street", "validation"): (250, 5541845),
    ("street", "test"): (251, 5547533),
}
EXPECTED_SPLIT_TOTALS = {"train": (1200, 11993084), "validation": (1498, 33236385),
                         "test": (1502, 33351444)}


# ----------------------------------------------------------------------------
# Поділ
# ----------------------------------------------------------------------------
def _sorted_jpgs(directory: Path) -> list[str]:
    # sorted() як рядків - так само впорядковує ImageFolder
    return sorted(p.name for p in directory.iterdir() if p.suffix.lower() == ".jpg")


def part_dir(root: Path, part: str, label: str) -> Path:
    base = "seg_train" if part == "train" else "seg_test"
    return root / base / base / label


def build_split(root: Path, train_per_class: int = TRAIN_PER_CLASS) -> list[tuple[str, str, str]]:
    """Рядки (file, label, part) за процедурою з умови."""
    rows: list[tuple[str, str, str]] = []

    rng = np.random.default_rng(0)               # один генератор на всі класи train
    for label in CLASSES:
        files = _sorted_jpgs(part_dir(root, "train", label))
        idx = np.arange(len(files))
        rng.shuffle(idx)
        chosen = sorted(files[i] for i in idx[:train_per_class])
        rows += [(f, label, "train") for f in chosen]

    rng = np.random.default_rng(0)               # новий генератор для seg_test
    val_rows, test_rows = [], []
    for label in CLASSES:
        files = _sorted_jpgs(part_dir(root, "test", label))
        idx = np.arange(len(files))
        rng.shuffle(idx)
        cut = len(files) // 2
        val_rows += [(f, label, "validation") for f in sorted(files[i] for i in idx[:cut])]
        test_rows += [(f, label, "test") for f in sorted(files[i] for i in idx[cut:])]
    return rows + val_rows + test_rows


def write_split(rows, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["file", "label", "part"])
        w.writerows(rows)


def read_split(path: Path) -> list[tuple[str, str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        return [(d["file"], d["label"], d["part"]) for d in r]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_control(rows):
    """Таблиця «кількість / сума числових імен» і ознака збігу з умовою."""
    got = {}
    for f, label, part in rows:
        n, s = got.get((label, part), (0, 0))
        got[(label, part)] = (n + 1, s + int(Path(f).stem))
    table, all_ok = [], True
    for label in CLASSES:
        row = {"Клас": label}
        for part in ("train", "validation", "test"):
            g = got.get((label, part), (0, 0))
            e = EXPECTED_SPLIT[(label, part)]
            ok = g == e
            all_ok &= ok
            row[part] = f"{g[0]} / {g[1]:,}".replace(",", " ") + ("" if ok else f"  ≠ {e[0]} / {e[1]:,}".replace(",", " "))
        table.append(row)
    tot = {"Клас": "Разом"}
    for part in ("train", "validation", "test"):
        n = sum(v[0] for (l, p), v in got.items() if p == part)
        s = sum(v[1] for (l, p), v in got.items() if p == part)
        e = EXPECTED_SPLIT_TOTALS[part]
        ok = (n, s) == e
        all_ok &= ok
        tot[part] = f"{n} / {s:,}".replace(",", " ") + ("" if ok else f"  ≠ {e[0]} / {e[1]:,}".replace(",", " "))
    table.append(tot)
    return table, all_ok


def check_disjoint(rows) -> bool:
    names = [f for f, _, _ in rows]
    if len(names) != len(set(names)):
        return False
    return True


# ----------------------------------------------------------------------------
# Дані
# ----------------------------------------------------------------------------
def train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


class SplitDataset(Dataset):
    """Зображення за рядками split.csv. Декодовані RGB-зображення кешуються в
    пам'яті (це лише прискорення: перетворення застосовуються щоразу заново)."""

    def __init__(self, rows, root: Path, transform):
        self.rows = list(rows)
        self.transform = transform
        self.targets = [CLASSES.index(label) for _, label, _ in self.rows]
        self.images = []
        for f, label, part in self.rows:
            with Image.open(part_dir(root, part, label) / f) as im:
                self.images.append(im.convert("RGB"))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.targets[i]


def make_train_loader(ds):
    # без власного generator/sampler: випадковість із глобального генератора PyTorch
    return DataLoader(ds, batch_size=BATCH_TRAIN, shuffle=True, drop_last=False, num_workers=0)


def make_eval_loader(ds):
    return DataLoader(ds, batch_size=BATCH_EVAL, shuffle=False, num_workers=0)


def rows_of(rows, part):
    return [r for r in rows if r[2] == part]


# ----------------------------------------------------------------------------
# Модель і режими
# ----------------------------------------------------------------------------
def pick_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{device.type} ({platform.processor() or platform.machine()})"


def build_model(regime: str, seed: int, device: torch.device):
    """Кроки 1-5 з умови. Повертає (model, optimizer, groups)."""
    assert regime in REGIMES
    torch.manual_seed(seed)                                         # 1
    np.random.seed(seed)
    weights = None if regime == "scratch" else ResNet18_Weights.IMAGENET1K_V1
    model = torchvision.models.resnet18(weights=weights)            # 2
    model.fc = torch.nn.Linear(512, 6)                              # 3
    prefixes = TRAINABLE_PREFIXES[regime]                           # 4
    for name, p in model.named_parameters():
        p.requires_grad = True if prefixes is None else name.startswith(prefixes)
    model.to(device)                                                # 5
    named = dict(model.named_parameters())
    if regime == "scratch":
        groups = [("all", [p for p in model.parameters()], LR_HEAD)]
    elif regime == "frozen":
        groups = [("fc", list(model.fc.parameters()), LR_HEAD)]
    else:
        groups = [("fc", list(model.fc.parameters()), LR_HEAD),
                  ("layer4", list(model.layer4.parameters()), LR_BACKBONE)]
    optimizer = torch.optim.AdamW(
        [{"params": ps, "lr": lr} for _, ps, lr in groups], weight_decay=WEIGHT_DECAY)
    group_info = [{"name": n, "n_params": sum(p.numel() for p in ps), "lr": lr}
                  for n, ps, lr in groups]
    return model, optimizer, group_info


def set_train_mode(model: torch.nn.Module, regime: str) -> None:
    """model.train(), після чого заморожені модулі повертаються в eval(), щоб
    їхні ковзні статистики BN не оновлювалися."""
    model.train()
    for name in EVAL_MODULES[regime]:
        getattr(model, name).eval()


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ----------------------------------------------------------------------------
# Навчання й оцінка
# ----------------------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train_epoch(model, optimizer, loader, regime, device, n_samples):
    set_train_mode(model, regime)
    criterion = torch.nn.CrossEntropyLoss()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(y)        # втрата пакета до кроку оптимізатора
    return total / n_samples


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    out = [model(x.to(device)).float().cpu().numpy() for x, _ in loader]
    return np.concatenate(out, axis=0)


def macro_f1(y_true, logits) -> float:
    pred = np.asarray(logits).argmax(axis=1)
    return float(f1_score(y_true, pred, labels=list(range(6)), average="macro", zero_division=0))


def run_one(regime, seed, device, train_ds, val_ds, log=print):
    """Один запуск за протоколом: 8 епох, після кожної - validation."""
    model, optimizer, groups = build_model(regime, seed, device)
    loader = make_train_loader(train_ds)       # після кроків 1-5: глобальний RNG уже засіяно
    val_loader = make_eval_loader(val_ds)
    losses, f1s, train_seconds = [], [], 0.0
    t_total = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        _sync(device)
        t0 = time.perf_counter()
        loss = train_epoch(model, optimizer, loader, regime, device, len(train_ds))
        _sync(device)
        train_seconds += time.perf_counter() - t0
        logits = predict_logits(model, val_loader, device)
        f1 = macro_f1(val_ds.targets, logits)
        losses.append(loss)
        f1s.append(f1)
        log(f"    {regime} seed={seed} епоха {epoch}/{EPOCHS}: loss={loss:.4f}  val macro-F1={f1:.4f}")
    total_seconds = time.perf_counter() - t_total
    return model, logits, losses, f1s, train_seconds, total_seconds, groups


# ----------------------------------------------------------------------------
# Журнал: git, версії
# ----------------------------------------------------------------------------
def git_info() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    try:
        return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}
    except Exception:
        return {"commit": "unknown (не git-репозиторій)", "dirty": None}


def env_info(device) -> dict:
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "numpy": np.__version__,
            "device": device_name(device)}


def run_name(regime, seed) -> str:
    return f"{regime}_seed{seed}"


def select_run(results: dict[tuple[str, int], float]):
    """Правило вибору за validation. results: (режим, зерно) -> F1 на епосі 8.

    Режим із найбільшим середнім F1 (scratch - його єдине значення); серед його
    запусків - той, що має найбільший F1 (за рівності - менше зерно)."""
    means = {}
    for r in REGIMES:
        vals = [v for (reg, _), v in results.items() if reg == r]
        if vals:
            means[r] = float(np.mean(vals))
    best_regime = max(means, key=lambda r: (means[r], -REGIMES.index(r)))
    cand = [(v, -s) for (reg, s), v in results.items() if reg == best_regime]
    best_seed = -max(cand)[1]
    return best_regime, best_seed, means
