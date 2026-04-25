"""
Evaluación en test, métricas y figuras estándar para TechStream.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    FEATURE_COLS,
    SEED,
    TEST_SIZE,
    VAL_SIZE,
    get_dataloaders,
    load_data,
)
from src.model import AnomalyDetector  # noqa: E402

HISTORY_JSON: Path = _PROJECT_ROOT / "models" / "training_history.json"
MODEL_PTH: Path = _PROJECT_ROOT / "models" / "model.pth"
DATA_CSV: Path = _PROJECT_ROOT / "data" / "sensors.csv"

FIGURES_DIR: Path = _PROJECT_ROOT / "figures"

COLOR_NORMAL: str = "#2196F3"
COLOR_FAILURE: str = "#F44336"

_EVAL_STATE: dict[str, object] = {}

BATCH_SIZE_EVAL: int = 64
HIDDEN_DIMS: list[int] = [128, 64, 32]
DROPOUT_RATE: float = 0.3


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_test_probabilities() -> tuple[np.ndarray, np.ndarray]:
    """Inferencia en el split de test (mismas semillas y rutas que en entrenamiento)."""
    device = _device()
    bundle = load_data(DATA_CSV, TEST_SIZE, VAL_SIZE, SEED)
    loaders = get_dataloaders(bundle, BATCH_SIZE_EVAL)

    model = AnomalyDetector(
        input_dim=len(FEATURE_COLS),
        hidden_dims=HIDDEN_DIMS,
        dropout_rate=DROPOUT_RATE,
    )
    try:
        state = torch.load(MODEL_PTH, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(MODEL_PTH, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    probs_list: list[np.ndarray] = []
    true_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loaders["test"]:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(p)
            true_list.append(yb.numpy().ravel())

    y_prob = np.concatenate(probs_list)
    y_true = np.concatenate(true_list).astype(int)
    return y_true, y_prob


def _ensure_predictions() -> None:
    if "y_true" not in _EVAL_STATE or "y_prob" not in _EVAL_STATE:
        yt, yp = predict_test_probabilities()
        _EVAL_STATE["y_true"] = yt
        _EVAL_STATE["y_prob"] = yp


def compute_metrics(threshold: float = 0.5) -> pd.DataFrame:
    """
    Calcula métricas en el conjunto de test y las muestra como tabla.
    Deja predicciones en caché para el resto de gráficos.
    """
    _ensure_predictions()
    y_true = np.asarray(_EVAL_STATE["y_true"], dtype=int)
    y_prob = np.asarray(_EVAL_STATE["y_prob"], dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    _EVAL_STATE["y_pred"] = y_pred
    _EVAL_STATE["threshold"] = float(threshold)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_prob)

    rows = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
    }
    df = pd.DataFrame([rows]).T
    df.columns = ["valor"]
    print(df.to_string())
    return df


def plot_training_history(
    history_path: str | Path | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Pérdidas train vs validation desde ``training_history.json``."""
    path = Path(history_path) if history_path else HISTORY_JSON
    with path.open(encoding="utf-8") as f:
        hist = json.load(f)
    fig, ax = plt.subplots(figsize=figsize)
    epochs = np.arange(1, len(hist["train_loss"]) + 1)
    ax.plot(epochs, hist["train_loss"], color=COLOR_NORMAL, lw=2, label="Train")
    ax.plot(epochs, hist["val_loss"], color=COLOR_FAILURE, lw=2, label="Val")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (BCE logits)")
    ax.set_title("Entrenamiento — pérdida por época")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_threshold_analysis(
    figsize: tuple[float, float] = (12, 6),
) -> plt.Figure:
    """Precision, recall y F1 vs umbral de decisión (probabilidad)."""
    _ensure_predictions()
    y_true = np.asarray(_EVAL_STATE["y_true"], dtype=int)
    y_prob = np.asarray(_EVAL_STATE["y_prob"], dtype=float)

    thresholds = np.linspace(0.01, 0.99, 99)
    precs, recs, f1s = [], [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true, yp, zero_division=0))
        recs.append(recall_score(y_true, yp, zero_division=0))
        f1s.append(f1_score(y_true, yp, zero_division=0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(thresholds, precs, color=COLOR_NORMAL, lw=2, label="Precision")
    ax.plot(thresholds, recs, color=COLOR_FAILURE, lw=2, label="Recall")
    ax.plot(thresholds, f1s, color="#4CAF50", lw=2, label="F1")
    ax.set_xlabel("Umbral de probabilidad")
    ax.set_ylabel("Métrica")
    ax.set_title("Análisis de umbral en test")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_confusion_matrix_fig(
    figsize: tuple[float, float] = (12, 6),
    threshold: float | None = None,
) -> plt.Figure:
    _ensure_predictions()
    y_true = np.asarray(_EVAL_STATE["y_true"], dtype=int)
    y_prob = np.asarray(_EVAL_STATE["y_prob"], dtype=float)
    thr = float(threshold if threshold is not None else _EVAL_STATE.get("threshold", 0.5))
    y_pred = (y_prob >= thr).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticklabels(["Real 0", "Real 1"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", color="black", fontsize=14)
    ax.set_title(f"Matriz de confusión (umbral={thr:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_roc_curve_fig(figsize: tuple[float, float] = (12, 6)) -> plt.Figure:
    _ensure_predictions()
    y_true = np.asarray(_EVAL_STATE["y_true"], dtype=int)
    y_prob = np.asarray(_EVAL_STATE["y_prob"], dtype=float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color=COLOR_NORMAL, lw=2, label=f"ROC AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC (test)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig


def plot_pr_curve_fig(figsize: tuple[float, float] = (12, 6)) -> plt.Figure:
    _ensure_predictions()
    y_true = np.asarray(_EVAL_STATE["y_true"], dtype=int)
    y_prob = np.asarray(_EVAL_STATE["y_prob"], dtype=float)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(rec, prec, color=COLOR_FAILURE, lw=2, label=f"PR AUC = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision–Recall (test)")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def plot_score_distribution_fig(figsize: tuple[float, float] = (12, 6)) -> plt.Figure:
    """Histograma de probabilidades predichas por clase real (4.ª figura estándar)."""
    _ensure_predictions()
    y_true = np.asarray(_EVAL_STATE["y_true"], dtype=int)
    y_prob = np.asarray(_EVAL_STATE["y_prob"], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        y_prob[y_true == 0],
        bins=30,
        alpha=0.65,
        color=COLOR_NORMAL,
        label="failure=0 (normal)",
    )
    ax.hist(
        y_prob[y_true == 1],
        bins=30,
        alpha=0.65,
        color=COLOR_FAILURE,
        label="failure=1",
    )
    ax.set_xlabel("Probabilidad predicha de fallo")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de scores por clase real")
    ax.legend()
    plt.tight_layout()
    return fig


def training_summary_table() -> pd.DataFrame:
    """Épocas, mejor val_loss y época del mejor modelo a partir del JSON de historial."""
    with HISTORY_JSON.open(encoding="utf-8") as f:
        hist = json.load(f)
    val_losses = hist["val_loss"]
    best_i = int(np.argmin(val_losses))
    df = pd.DataFrame(
        {
            "metrica": [
                "epochs_entrenadas",
                "mejor_val_loss",
                "epoch_mejor_modelo",
            ],
            "valor": [
                len(val_losses),
                float(val_losses[best_i]),
                best_i + 1,
            ],
        }
    )
    return df


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    print("Métricas en conjunto de test (umbral 0.5):")
    compute_metrics(0.5)
