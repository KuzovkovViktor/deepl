"""Ручна реалізація MLP  X -> Linear(4,8) -> ReLU -> Linear(8,3)  у NumPy.
Автоматичне диференціювання не використовується."""
import numpy as np


def init_params():
    rng = np.random.default_rng(0)           # новий генератор; W1, потім W2
    W1 = rng.normal(0.0, np.sqrt(2.0 / 4), size=(4, 8))           # He
    b1 = np.zeros(8)
    W2 = rng.normal(0.0, np.sqrt(2.0 / (8 + 3)), size=(8, 3))     # Xavier
    b2 = np.zeros(3)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def log_softmax(Z):
    """Стабільний log-softmax зі зсувом на максимум по рядку."""
    shifted = Z - Z.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def forward(X, p):
    """Повертає логіти та кеш проміжних значень для зворотного проходу."""
    Z1 = X @ p["W1"] + p["b1"]               # (N, 8)
    A1 = np.maximum(Z1, 0.0)                 # (N, 8)
    Z2 = A1 @ p["W2"] + p["b2"]              # (N, 3)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2}
    return Z2, cache


def cross_entropy(Z2, y):
    """Середня крос-ентропія за об'єктами через log-softmax."""
    logp = log_softmax(Z2)
    return -logp[np.arange(len(y)), y].mean()


def loss_fn(X, y, p):
    Z2, _ = forward(X, p)
    return cross_entropy(Z2, y)


def backward(y, p, cache, faulty=False):
    """Градієнти dW1, db1, dW2, db2.

    faulty=True -- навмисна помилка (дослід 4): немає ділення на N
    у градієнті за логітами.
    """
    X, Z1, A1, Z2 = cache["X"], cache["Z1"], cache["A1"], cache["Z2"]
    N = X.shape[0]

    P = np.exp(log_softmax(Z2))              # softmax, (N, 3)
    Y = np.zeros_like(P)
    Y[np.arange(N), y] = 1.0                 # one-hot, (N, 3)

    dZ2 = P - Y                              # (N, 3)
    if not faulty:
        dZ2 = dZ2 / N

    dW2 = A1.T @ dZ2                         # (8, 3)
    db2 = dZ2.sum(axis=0)                    # (3,)
    dA1 = dZ2 @ p["W2"].T                    # (N, 8)
    dZ1 = dA1 * (Z1 > 0)                     # (N, 8)
    dW1 = X.T @ dZ1                          # (4, 8)
    db1 = dZ1.sum(axis=0)                    # (8,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
