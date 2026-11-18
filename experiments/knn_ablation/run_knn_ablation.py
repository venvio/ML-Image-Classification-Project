"""kNN ablation for Fashion-MNIST: clean training, clean/noisy test inputs.

Downloads the official Fashion-MNIST IDX gzip files into ./data/fashion_mnist,
then writes metrics, settings, and a plot into ./results.
"""

import gzip
import json
import platform
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


SEED = 0
NOISE_SEEDS = [0, 1, 2]
K_VALUES = [1, 3, 5, 7, 9, 15]
NOISE_LEVELS = [0.1, 0.2, 0.3, 0.5]
N_CLASSES = 10
BASE_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/"
    "fashion-mnist/master/data/fashion/"
)
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data" / "fashion_mnist"
RESULTS_DIR = SCRIPT_DIR / "results"

FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def download_dataset():
    """Download the four official Fashion-MNIST data files if not present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = DATA_DIR / filename
        if destination.exists():
            continue
        print(f"Downloading {filename}...")
        request = urllib.request.Request(
            BASE_URL + filename,
            headers={"User-Agent": "Fashion-MNIST course project"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            destination.write_bytes(response.read())


def load_dataset():
    """Read Fashion-MNIST IDX files from this script's local data directory."""
    download_dataset()

    def read_images(filename, count):
        with gzip.open(DATA_DIR / filename, "rb") as stream:
            data = stream.read()
        images = np.frombuffer(data, dtype=np.uint8, offset=16)
        if images.size != count * 28 * 28:
            raise ValueError(f"Unexpected image data size in {filename}")
        return images.reshape(count, 28, 28)

    def read_labels(filename, count):
        with gzip.open(DATA_DIR / filename, "rb") as stream:
            data = stream.read()
        labels = np.frombuffer(data, dtype=np.uint8, offset=8)
        if labels.size != count:
            raise ValueError(f"Unexpected label data size in {filename}")
        return labels

    x_train_full = read_images(FILES[0], 60000)
    y_train_full = read_labels(FILES[1], 60000)
    x_test = read_images(FILES[2], 10000)
    y_test = read_labels(FILES[3], 10000)
    return x_train_full, y_train_full, x_test, y_test


def flatten_scale(images):
    """Flatten 28x28 images and scale pixel intensities into [0, 1]."""
    return images.reshape(len(images), 784).astype(np.float32) / 255.0


def add_test_noise(images, sigma, seed):
    """Add Gaussian noise to test inputs only, clipping pixels to [0, 1]."""
    rng = np.random.default_rng([seed, int(round(sigma * 1000)), 1])
    noisy = images + rng.normal(0.0, sigma, images.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)


def predict_for_all_k(reference_labels, neighbor_indices):
    """Apply uniform voting to the nearest-neighbor list for every k."""
    neighbor_labels = reference_labels[neighbor_indices]
    votes = np.eye(N_CLASSES, dtype=np.int16)[neighbor_labels]
    cumulative_votes = np.cumsum(votes, axis=1)
    return {
        k: np.argmax(cumulative_votes[:, k - 1, :], axis=1)
        for k in K_VALUES
    }


def score_rows(y_true, predictions, condition, sigma, noise_seed):
    return [
        {
            "condition": condition,
            "noise_sigma": sigma,
            "noise_seed": noise_seed,
            "k": k,
            "accuracy": accuracy_score(y_true, prediction),
            "macro_f1": f1_score(y_true, prediction, average="macro"),
        }
        for k, prediction in predictions.items()
    ]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading Fashion-MNIST from the project data folder...")
    x_all_img, y_all, x_test_img, y_test = load_dataset()

    x_all = flatten_scale(x_all_img)
    x_test = flatten_scale(x_test_img)

    # Stratified split of official training data: 50k train / 10k validation.
    # The official 10k test set remains untouched until final evaluation.
    x_train, x_val, y_train, y_val = train_test_split(
        x_all,
        y_all,
        test_size=10000,
        stratify=y_all,
        random_state=SEED,
    )
    print(
        f"Train={len(x_train)}, validation={len(x_val)}, test={len(x_test)}"
    )
    print(f"k values: {K_VALUES}")
    print(f"Test-only noise levels: {NOISE_LEVELS}")

    # Fit the exact Euclidean neighbor index once on clean training data.
    print("Building exact Euclidean kNN index on clean training images...")
    knn = NearestNeighbors(
        n_neighbors=max(K_VALUES),
        algorithm="brute",
        metric="euclidean",
        n_jobs=-1,
    ).fit(x_train)

    # Select k using validation accuracy only.
    print("Selecting k on clean validation data...")
    val_indices = knn.kneighbors(x_val, return_distance=False)
    val_predictions = predict_for_all_k(y_train, val_indices)
    validation_rows = [
        {
            "k": k,
            "validation_accuracy": accuracy_score(y_val, pred),
            "validation_macro_f1": f1_score(y_val, pred, average="macro"),
        }
        for k, pred in val_predictions.items()
    ]
    validation_df = pd.DataFrame(validation_rows).sort_values(
        ["validation_accuracy", "k"], ascending=[False, True]
    )
    selected_k = int(validation_df.iloc[0]["k"])
    print(validation_df.to_string(index=False))
    print(f"Validation-selected k: {selected_k}")

    results = []

    # Clean reference condition; training and test images are both clean.
    print("Evaluating clean test inputs...")
    clean_indices = knn.kneighbors(x_test, return_distance=False)
    clean_predictions = predict_for_all_k(y_train, clean_indices)
    results.extend(score_rows(y_test, clean_predictions, "clean", 0.0, 0))

    # Stress condition: keep training clean and perturb test images only.
    # Three noise seeds quantify variation due to random noise generation.
    for sigma in NOISE_LEVELS:
        for noise_seed in NOISE_SEEDS:
            print(f"Evaluating test noise sigma={sigma}, seed={noise_seed}...")
            x_noisy = add_test_noise(x_test, sigma, noise_seed)
            noisy_indices = knn.kneighbors(x_noisy, return_distance=False)
            noisy_predictions = predict_for_all_k(y_train, noisy_indices)
            results.extend(
                score_rows(
                    y_test, noisy_predictions,
                    "test_only_noise", sigma, noise_seed
                )
            )

    raw = pd.DataFrame(results)
    raw.to_csv(RESULTS_DIR / "knn_ablation_raw.csv", index=False)

    summary = (
        raw.groupby(["condition", "noise_sigma", "k"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .fillna(0.0)
    )
    summary.to_csv(RESULTS_DIR / "knn_ablation_summary.csv", index=False)

    # Curves show how k behaves on clean and increasingly noisy test inputs.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, mean_col, std_col, title in [
        (axes[0], "accuracy_mean", "accuracy_std", "Test accuracy"),
        (axes[1], "macro_f1_mean", "macro_f1_std", "Test macro-F1"),
    ]:
        for sigma in [0.0] + NOISE_LEVELS:
            condition = "clean" if sigma == 0.0 else "test_only_noise"
            subset = summary[
                (summary["condition"] == condition)
                & (summary["noise_sigma"] == sigma)
            ].sort_values("k")
            label = "Clean" if sigma == 0.0 else f"Noise sigma={sigma}"
            axis.errorbar(
                subset["k"], subset[mean_col], yerr=subset[std_col],
                marker="o", capsize=3, label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Number of neighbors (k)")
        axis.set_ylabel("Score")
        axis.set_xticks(K_VALUES)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    fig.suptitle("kNN ablation: clean training, clean/noisy test inputs")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "knn_ablation_curve.png", dpi=160)
    plt.close(fig)

    settings = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "split_seed": SEED,
        "noise_seeds": NOISE_SEEDS,
        "k_values": K_VALUES,
        "noise_levels_sigma_on_0_to_1_pixels": NOISE_LEVELS,
        "noise_setting": "test only; training images stay clean",
        "distance": "Euclidean",
        "voting": "uniform",
        "selection_metric": "validation accuracy",
        "validation_selected_k": selected_k,
        "training_size": len(x_train),
        "validation_size": len(x_val),
        "test_size": len(x_test),
    }
    with open(RESULTS_DIR / "settings.json", "w") as stream:
        json.dump(settings, stream, indent=2)

    print(f"\nFinished. Results saved in {RESULTS_DIR}")
    print("  knn_ablation_raw.csv")
    print("  knn_ablation_summary.csv")
    print("  knn_ablation_curve.png")
    print("  settings.json")


if __name__ == "__main__":
    main()
