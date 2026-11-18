import json
import os
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message=".*zero standard deviation.*")

QUICK = os.environ.get("QUICK") == "1"   # QUICK=1 runs a small smoke test
RESULTS = "results"
SEEDS = [0, 1, 2]                        # at least 3 runs, as the project requires
CPUS = os.cpu_count()

CLASSES = ["Top", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Boot"]
MODELS = ["Baseline", "LogReg", "kNN", "MLP"]    # Baseline = nearest centroid
DETERMINISTIC = {"Baseline", "LogReg", "kNN"}    # same data gives the same model
COLORS = {"Baseline": "#2a78d6", "LogReg": "#eb6834",
          "kNN": "#1baf7a", "MLP": "#eda100"}

# Same tuning budget for every model: 3 configurations, chosen on validation data
SEARCH = {
    "Baseline": [{}],
    "LogReg": [{"C": 0.01}, {"C": 0.1}, {"C": 1.0}],
    "kNN": [{"n_neighbors": 3}, {"n_neighbors": 5}, {"n_neighbors": 9}],
    "MLP": [{"hidden_layer_sizes": (128,)},
            {"hidden_layer_sizes": (256,)},
            {"hidden_layer_sizes": (256, 128)}],
}

TRAIN_SIZES = [500, 1000, 2000, 5000, 10000, 20000]   # full training set is added on top
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.5]                   # std of Gaussian pixel noise
MASK_SIZES = [4, 8, 12, 16]                           # side of a black square, in pixels

FOCUS = {"train_size": 1000, "noise": 0.2, "mask": 12}

if QUICK:
    TRAIN_SIZES = [500, 1000, 2000]
    NOISE_LEVELS = [0.2, 0.5]
    MASK_SIZES = [8, 12]


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# Data
def _download_raw():
    import gzip
    import urllib.request
    base = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"

    def fetch(name, offset):
        log(f"Downloading {name}")
        with urllib.request.urlopen(base + name + ".gz") as r:
            return np.frombuffer(gzip.decompress(r.read()), dtype=np.uint8, offset=offset)

    return (fetch("train-images-idx3-ubyte", 16).reshape(60000, 784),
            fetch("train-labels-idx1-ubyte", 8),
            fetch("t10k-images-idx3-ubyte", 16).reshape(10000, 784),
            fetch("t10k-labels-idx1-ubyte", 8))


def load_raw():
    cache = "fashion_mnist.npz"
    if not os.path.exists(cache):
        try:
            from keras.datasets import fashion_mnist   # same loader as main.ipynb
            (xtr, ytr), (xte, yte) = fashion_mnist.load_data()
            xtr, xte = xtr.reshape(60000, 784), xte.reshape(10000, 784)
        except Exception as err:                       # keras missing or download failed
            log(f"keras loader unavailable ({type(err).__name__}), downloading directly")
            xtr, ytr, xte, yte = _download_raw()
        np.savez_compressed(cache, X_train=xtr, y_train=ytr.astype(np.int64),
                            X_test=xte, y_test=yte.astype(np.int64))
    d = np.load(cache)
    return d["X_train"], d["y_train"], d["X_test"], d["y_test"]


def get_data():
    X_full, y_full, X_test, y_test = load_raw()
    X_full = (X_full / 255.0).astype(np.float32)
    X_test = (X_test / 255.0).astype(np.float32)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=10000, stratify=y_full, random_state=0)

    if QUICK:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=6000, stratify=y_train, random_state=0)
        X_val, _, y_val, _ = train_test_split(
            X_val, y_val, train_size=2000, stratify=y_val, random_state=0)
        X_test, _, y_test, _ = train_test_split(
            X_test, y_test, train_size=2000, stratify=y_test, random_state=0)

    log(f"data: train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")
    return dict(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val,
                X_test=X_test, y_test=y_test)


def subsample(X, y, size, seed):
    if size >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=size, stratify=y, random_state=seed)
    return Xs, ys

def _rng(kind, level, seed):
    return np.random.default_rng([seed, int(round(level * 1000)), {"noise": 1, "mask": 2}[kind]])


def add_noise(X, sigma, seed):
    rng = _rng("noise", sigma, seed)
    noisy = X + rng.normal(0.0, sigma, X.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def add_mask(X, size, seed):
    rng = _rng("mask", size, seed)
    imgs = X.reshape(len(X), 28, 28).copy()
    rows = rng.integers(0, 28 - size + 1, len(X))
    cols = rng.integers(0, 28 - size + 1, len(X))
    for i, (r, c) in enumerate(zip(rows, cols)):
        imgs[i, r:r + size, c:c + size] = 0.0
    return imgs.reshape(len(X), 784)


CORRUPT = {"noise": add_noise, "mask": add_mask}
SETTING = {"noise": NOISE_SETTING, "mask": MASK_SETTING}


# Models
def make_model(name, params, seed):
    if name == "Baseline":
        return NearestCentroid()
    if name == "LogReg":
        return LogisticRegression(max_iter=300, **params)
    if name == "kNN":
        return KNeighborsClassifier(algorithm="brute", n_jobs=CPUS, **params)
    if name == "MLP":
        return MLPClassifier(early_stopping=True, max_iter=20 if QUICK else 50,
                             batch_size=256, random_state=seed, **params)
    raise ValueError(name)


def get_best_params(data):
    os.makedirs(RESULTS, exist_ok=True)
    path = f"{RESULTS}/best_params{'_quick' if QUICK else ''}.json"
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        log(f"loaded tuned hyperparameters from {path}")
        return {k: {p: tuple(v) if isinstance(v, list) else v for p, v in d.items()}
                for k, d in saved["best_params"].items()}

    best, rows = {}, []
    for name in MODELS:
        best_acc = None
        for params in SEARCH[name]:
            m = make_model(name, params, seed=0).fit(data["X_train"], data["y_train"])
            acc = accuracy_score(data["y_val"], m.predict(data["X_val"]))
            rows.append({"model": name, "params": str(params), "val_accuracy": round(acc, 4)})
            log(f"tune {name} {params}: val acc {acc:.4f}")
            if best_acc is None or acc > best_acc:
                best_acc, best[name] = acc, params
    with open(path, "w") as f:
        json.dump({"best_params": best, "tuning": rows}, f, indent=2)
    return best


def train_reference_models(data, best_params, X=None, y=None):
    X = data["X_train"] if X is None else X
    y = data["y_train"] if y is None else y
    models = {}
    for name in MODELS:
        if name in DETERMINISTIC:
            m = make_model(name, best_params[name], 0).fit(X, y)
            models[name] = {s: m for s in SEEDS}
        else:
            models[name] = {s: make_model(name, best_params[name], s).fit(X, y) for s in SEEDS}
    return models


# Metrics
def per_class_recall(y, pred):
    cm = confusion_matrix(y, pred, labels=range(10))
    return cm.diagonal() / cm.sum(axis=1)
