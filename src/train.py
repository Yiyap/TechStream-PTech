"""
Bucle de entrenamiento con early stopping para TechStream AnomalyDetector.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset import (  # noqa: E402
    TEST_SIZE,
    VAL_SIZE,
    get_dataloaders,
    get_pos_weight,
    load_data,
)
from src.model import build_model  # noqa: E402

LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 64
MAX_EPOCHS: int = 150
PATIENCE: int = 10
HIDDEN_DIMS: list[int] = [128, 64, 32]
DROPOUT_RATE: float = 0.3
MODEL_PATH: Path = Path("models/model.pth")
HISTORY_PATH: Path = Path("models/training_history.json")
SEED: int = 42
DATA_PATH: Path = Path("data/sensors.csv")


def _resolve_project_path(path: Path | str) -> Path:
    """Rutas relativas al directorio raíz del repo (donde están ``data/`` y ``models/``)."""
    p = Path(path)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def _ensure_utf8_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass


def set_seed(seed: int) -> None:
    """Fija semillas de ``torch``, ``numpy`` y ``random`` para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Detiene el entrenamiento si ``val_loss`` no mejora durante ``patience`` épocas."""

    def __init__(self, patience: int, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float | None = None
        self._counter = 0

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self._counter = 0
            return False
        self._counter += 1
        return self._counter >= self.patience


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Una época de entrenamiento; devuelve pérdida y accuracy medios."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        batch_n = xb.size(0)
        total_loss += loss.item() * batch_n
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        correct += int((pred == yb).sum().item())
        total += batch_n

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }


def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evalúa sin gradientes: pérdida y accuracy medios."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_n = xb.size(0)
            total_loss += loss.item() * batch_n
            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).float()
            correct += int((pred == yb).sum().item())
            total += batch_n

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }


def train(config: dict | None = None) -> dict[str, list[float]]:
    """
    Entrena ``AnomalyDetector`` con BCE logits, Adam, ReduceLROnPlateau y early stopping.

    Guarda el mejor ``state_dict`` (menor ``val_loss``) en ``MODEL_PATH`` y el historial
    en ``HISTORY_PATH``.
    """
    _ensure_utf8_stdout()

    cfg: dict[str, object] = {
        "LEARNING_RATE": LEARNING_RATE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_EPOCHS": MAX_EPOCHS,
        "PATIENCE": PATIENCE,
        "HIDDEN_DIMS": list(HIDDEN_DIMS),
        "DROPOUT_RATE": DROPOUT_RATE,
        "MODEL_PATH": MODEL_PATH,
        "HISTORY_PATH": HISTORY_PATH,
        "SEED": SEED,
        "DATA_PATH": DATA_PATH,
        "TEST_SIZE": TEST_SIZE,
        "VAL_SIZE": VAL_SIZE,
    }
    if config:
        cfg.update(config)

    seed = int(cfg["SEED"])
    set_seed(seed)

    data_path = _resolve_project_path(str(cfg["DATA_PATH"]))
    bundle = load_data(
        data_path,
        float(cfg["TEST_SIZE"]),
        float(cfg["VAL_SIZE"]),
        seed,
    )
    loaders = get_dataloaders(bundle, int(cfg["BATCH_SIZE"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dims_cfg = cfg["HIDDEN_DIMS"]
    if not isinstance(hidden_dims_cfg, list):
        raise TypeError("HIDDEN_DIMS debe ser list[int].")
    hidden_dims_list = [int(h) for h in hidden_dims_cfg]

    model = build_model(
        input_dim=7,
        hidden_dims=hidden_dims_list,
        dropout_rate=float(cfg["DROPOUT_RATE"]),
    ).to(device)

    pos_w = get_pos_weight(bundle.y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["LEARNING_RATE"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    early_stop = EarlyStopping(patience=int(cfg["PATIENCE"]), min_delta=1e-4)
    model_path = _resolve_project_path(str(cfg["MODEL_PATH"]))
    history_path = _resolve_project_path(str(cfg["HISTORY_PATH"]))
    model_path.parent.mkdir(parents=True, exist_ok=True)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_for_save = float("inf")
    max_epochs = int(cfg["MAX_EPOCHS"])

    for epoch in range(1, max_epochs + 1):
        tr = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        va = evaluate_epoch(model, loaders["val"], criterion, device)
        scheduler.step(va["loss"])

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr["accuracy"])
        history["val_acc"].append(va["accuracy"])

        lr = optimizer.param_groups[0]["lr"]
        improved = va["loss"] < best_val_for_save - 1e-4
        if improved:
            best_val_for_save = va["loss"]
            torch.save(model.state_dict(), model_path)

        stop_now = early_stop(va["loss"])

        star = "⭐ " if improved else ""
        stop_mark = "🛑 " if stop_now else ""
        print(
            f"{star}{stop_mark}"
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train Loss: {tr['loss']:.4f} | "
            f"Val Loss: {va['loss']:.4f} | "
            f"Val Acc: {va['accuracy']:.1f}% | "
            f"LR: {lr:.1e}"
        )

        if stop_now:
            break

    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history


if __name__ == "__main__":
    _ensure_utf8_stdout()
    hist = train(config=None)
    n_ep = len(hist["val_loss"])
    best_i = int(np.argmin(hist["val_loss"]))
    best_loss = float(hist["val_loss"][best_i])
    best_ep = best_i + 1
    print()
    print(f"Mejor val_loss: {best_loss:.4f} (época {best_ep})")
    print(f"Épocas totales entrenadas: {n_ep}")
