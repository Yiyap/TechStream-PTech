"""
Dataset PyTorch y utilidades de carga para telemetría TechStream (clasificación binaria).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

FEATURE_COLS: list[str] = [
    "cpu_usage",
    "memory_usage",
    "temperature",
    "network_traffic",
    "disk_io",
    "error_rate",
    "response_time",
]
TARGET_COL: str = "failure"
TEST_SIZE: float = 0.15
VAL_SIZE: float = 0.15
SEED: int = 42

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH: Path = _PROJECT_ROOT / "data" / "sensors.csv"


@dataclass(frozen=True)
class TrainValTestData:
    """Contenedor de matrices escaladas, target, scaler y pesos de clase."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    class_weights: torch.Tensor


class SensorDataset(Dataset):
    """Pares (features, etiqueta) para entrenamiento o evaluación."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32).ravel()

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


def load_data(
    csv_path: Path | str,
    test_size: float,
    val_size: float,
    seed: int,
) -> TrainValTestData:
    """
    Lee el CSV, divide train/val/test de forma estratificada, escala con
    ``StandardScaler`` ajustado solo en train y calcula pesos de clase
    balanceados para la rama de entrenamiento.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float64)
    y = df[TARGET_COL].to_numpy()

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )

    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_fraction,
        stratify=y_train_val,
        random_state=seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    classes = np.array([0, 1], dtype=np.int64)
    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(np.int64),
    )
    class_weights = torch.as_tensor(cw, dtype=torch.float32)

    return TrainValTestData(
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        scaler=scaler,
        class_weights=class_weights,
    )


def get_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    """``n_negatives / n_positives`` para ``BCEWithLogitsLoss(pos_weight=...)``."""
    y = np.asarray(y_train).ravel()
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    if n_pos <= 0:
        raise ValueError("y_train no contiene ejemplos positivos (failure=1).")
    return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def get_dataloaders(data: TrainValTestData, batch_size: int) -> dict[str, DataLoader]:
    """
    ``DataLoader`` de train (con ``WeightedRandomSampler``), val y test.
    Val y test usan el doble de tamaño de lote y sin barajado.
    """
    train_ds = SensorDataset(data.X_train, data.y_train)
    val_ds = SensorDataset(data.X_val, data.y_val)
    test_ds = SensorDataset(data.X_test, data.y_test)

    y_train_int = np.asarray(data.y_train, dtype=np.int64)
    counts = np.bincount(y_train_int, minlength=2)
    inv_freq = 1.0 / np.maximum(counts, 1)
    sample_weights = inv_freq[y_train_int]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
    )

    return {"train": train_loader, "val": val_loader, "test": test_loader}


def _failure_rate(y: np.ndarray) -> float:
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 0.0
    return float(np.mean(y == 1) * 100.0)


def _print_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
    y = np.asarray(y).ravel()
    print(
        f"{name}: X={X.shape}  y={y.shape}  "
        f"Fallos: {_failure_rate(y):.1f}%"
    )


if __name__ == "__main__":
    bundle = load_data(DEFAULT_CSV_PATH, TEST_SIZE, VAL_SIZE, SEED)
    _print_split("Train", bundle.X_train, bundle.y_train)
    _print_split("Val", bundle.X_val, bundle.y_val)
    _print_split("Test", bundle.X_test, bundle.y_test)
    pw = get_pos_weight(bundle.y_train)
    print(f"pos_weight: {pw.item():.4f}")
