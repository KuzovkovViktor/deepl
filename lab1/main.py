"""Запуск перевірок ЛР1.

    uv run python main.py            # правильна реалізація (за замовчуванням)
    uv run python main.py --faulty   # дослід 4: без ділення на N
"""
import argparse
import copy
import numpy as np

from data import load_data
from mlp_numpy import init_params, forward, loss_fn, backward
from mlp_torch import torch_loss_and_grads

TOL_TORCH = 1e-12
TOL_NUM = 1e-7
EPS = 1e-6
NAMES = {"W1": "dW1", "b1": "db1", "W2": "dW2", "b2": "db2"}
CHECK = {"W1": (0, 0), "b1": (0,), "W2": (0, 0), "b2": (0,)}


def ok(flag):
    return "так" if flag else "НІ"


def main(faulty):
    X, y, X_te, y_te = load_data()
    print(f"train: {X.shape}, test: {X_te.shape}; "
          f"train class counts {np.bincount(y)}, test {np.bincount(y_te)}")
    p = init_params()

    # --- NumPy forward/backward ---
    Z2, cache = forward(X, p)
    loss_np = loss_fn(X, y, p)
    g = backward(y, p, cache, faulty=faulty)
    print(f"\nРежим: {'НАВМИСНА ПОМИЛКА (без /N)' if faulty else 'правильна реалізація'}")
    print("Розмірності:", {k: v.shape for k, v in cache.items() if k != 'X'},
          {NAMES[k]: v.shape for k, v in g.items()})
    print("Кількість нульових елементів градієнтів:",
          {NAMES[k]: int((v == 0).sum()) for k, v in g.items()})

    # --- 2. PyTorch ---
    loss_t, g_t = torch_loss_and_grads(X, y, p)
    print("\n== 2. Звірка з PyTorch ==")
    print(f"Втрата NumPy  : {loss_np!r}")
    print(f"Втрата PyTorch: {loss_t!r}")
    all_finite = np.isfinite(loss_np) and np.isfinite(loss_t) and \
        all(np.isfinite(v).all() for v in g.values()) and \
        all(np.isfinite(v).all() for v in g_t.values())
    print(f"Усі значення скінченні: {all_finite}")
    d = abs(loss_np - loss_t)
    print(f"{'Величина':10s} {'max |NumPy-PyTorch|':>22s}  Пройдено")
    print(f"{'Втрата':10s} {d:22.3e}  {ok(d <= TOL_TORCH)}")
    for k in ("W1", "b1", "W2", "b2"):
        dk = np.abs(g[k] - g_t[k]).max()
        print(f"{'Градієнт '+NAMES[k]:10s} {dk:22.3e}  {ok(dk <= TOL_TORCH)}")
        print(f"   max|g_numpy| = {np.abs(g[k]).max():.6e}, "
              f"max|g_torch| = {np.abs(g_t[k]).max():.6e}")
    if faulty:
        for k in g:
            m = g_t[k] != 0
            r = g[k][m] / g_t[k][m]
            print(f"   {NAMES[k]}: відношення NumPy/PyTorch на ненульових: "
                  f"min={r.min():.6f}, max={r.max():.6f}")

    # --- 3. Чисельне диференціювання ---
    print("\n== 3. Центральна різниця, eps = 1e-6 ==")
    print(f"{'Параметр':8s} {'backward()':>18s} {'числ. похідна':>18s} "
          f"{'|різниця|':>11s}  Пройдено")
    for k, idx in CHECK.items():
        q = copy.deepcopy(p)
        theta0 = q[k][idx]
        q[k][idx] = theta0 + EPS
        Lp = loss_fn(X, y, q)
        q[k][idx] = theta0 - EPS
        Lm = loss_fn(X, y, q)
        q[k][idx] = theta0
        gnum = (Lp - Lm) / (2 * EPS)
        gman = g[k][idx]
        diff = abs(gnum - gman)
        label = f"{k}[{', '.join(map(str, idx))}]"
        print(f"{label:8s} {gman:18.10e} {gnum:18.10e} {diff:11.3e}  {ok(diff <= TOL_NUM)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--faulty", action="store_true",
                    help="дослід 4: градієнт за логітами без ділення на N")
    main(ap.parse_args().faulty)
