"""
Generador sintético de telemetría de servidores para TechStream Anomaly Detection.

Produce un CSV con features continuas y etiqueta binaria `failure` según reglas
físicas coherentes (sobrecarga, térmica, errores, degradación).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- Constantes globales (sin números mágicos fuera de este bloque) ---

SEED: int = 42
N_SAMPLES: int = 5000

_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
OUTPUT_PATH: Path = _PROJECT_ROOT / "data" / "sensors.csv"

NORMAL_BUCKET_FRACTION: float = 0.8

TIMESTAMP_START: str = "2024-01-01"
TIMESTAMP_FREQ: str = "1min"

# Rangos “normales” (80 % de filas)
CPU_NORMAL_LOW: float = 20.0
CPU_NORMAL_HIGH: float = 75.0
MEMORY_NORMAL_LOW: float = 30.0
MEMORY_NORMAL_HIGH: float = 80.0
TEMP_NORMAL_LOW: float = 40.0
TEMP_NORMAL_HIGH: float = 70.0
NETWORK_NORMAL_LOW: float = 10.0
NETWORK_NORMAL_HIGH: float = 500.0
DISK_NORMAL_LOW: float = 5.0
DISK_NORMAL_HIGH: float = 200.0
ERROR_NORMAL_LOW: float = 0.0
ERROR_NORMAL_HIGH: float = 5.0
RESPONSE_NORMAL_LOW: float = 50.0
RESPONSE_NORMAL_HIGH: float = 300.0

# Rangos elevados (20 % de filas): sesgados hacia condiciones de fallo sin forzar 100 %
CPU_ELEVATED_LOW: float = 76.0
CPU_ELEVATED_HIGH: float = 100.0
MEMORY_ELEVATED_LOW: float = 78.0
MEMORY_ELEVATED_HIGH: float = 97.0
TEMP_ELEVATED_LOW: float = 76.0
TEMP_ELEVATED_HIGH: float = 94.0
NETWORK_ELEVATED_LOW: float = 10.0
NETWORK_ELEVATED_HIGH: float = 500.0
DISK_ELEVATED_LOW: float = 5.0
DISK_ELEVATED_HIGH: float = 200.0
ERROR_ELEVATED_LOW: float = 5.0
ERROR_ELEVATED_HIGH: float = 13.0
RESPONSE_ELEVATED_LOW: float = 520.0
RESPONSE_ELEVATED_HIGH: float = 980.0

FEATURE_NOISE_STD: dict[str, float] = {
    "cpu_usage": 3.0,
    "memory_usage": 2.5,
    "temperature": 1.5,
    "network_traffic": 20.0,
    "disk_io": 10.0,
    "error_rate": 0.5,
    "response_time": 15.0,
}

# Umbrales de reglas de fallo
CPU_R1_THRESHOLD: float = 90.0
MEMORY_R1_THRESHOLD: float = 85.0
TEMP_R2_THRESHOLD: float = 85.0
ERROR_R3_THRESHOLD: float = 8.0
RESPONSE_R4_THRESHOLD: float = 800.0
CPU_R4_THRESHOLD: float = 80.0

FEATURE_ORDER: tuple[str, ...] = (
    "cpu_usage",
    "memory_usage",
    "temperature",
    "network_traffic",
    "disk_io",
    "error_rate",
    "response_time",
)


def _sample_feature(
    name: str,
    low: float,
    high: float,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Muestrea uniforme en [low, high], añade ruido gaussiano con desviación
    típica según el nombre del feature y recorta a valores >= 0.
    """
    std = FEATURE_NOISE_STD[name]
    base = rng.uniform(low, high, size=n)
    noise = rng.normal(0.0, std, size=n)
    return np.maximum(base + noise, 0.0)


def _is_failure(row: pd.Series) -> int:
    """
    Aplica las cuatro reglas físicas; devuelve 1 si al menos una se cumple.

    R1: sobrecarga combinada (CPU y memoria altas).
    R2: riesgo térmico.
    R3: inestabilidad por tasa de errores.
    R4: degradación de servicio (latencia y CPU altas).
    """
    r1 = (row["cpu_usage"] > CPU_R1_THRESHOLD) and (
        row["memory_usage"] > MEMORY_R1_THRESHOLD
    )
    r2 = row["temperature"] > TEMP_R2_THRESHOLD
    r3 = row["error_rate"] > ERROR_R3_THRESHOLD
    r4 = (row["response_time"] > RESPONSE_R4_THRESHOLD) and (
        row["cpu_usage"] > CPU_R4_THRESHOLD
    )
    return 1 if (r1 or r2 or r3 or r4) else 0


def _normal_bounds() -> dict[str, tuple[float, float]]:
    return {
        "cpu_usage": (CPU_NORMAL_LOW, CPU_NORMAL_HIGH),
        "memory_usage": (MEMORY_NORMAL_LOW, MEMORY_NORMAL_HIGH),
        "temperature": (TEMP_NORMAL_LOW, TEMP_NORMAL_HIGH),
        "network_traffic": (NETWORK_NORMAL_LOW, NETWORK_NORMAL_HIGH),
        "disk_io": (DISK_NORMAL_LOW, DISK_NORMAL_HIGH),
        "error_rate": (ERROR_NORMAL_LOW, ERROR_NORMAL_HIGH),
        "response_time": (RESPONSE_NORMAL_LOW, RESPONSE_NORMAL_HIGH),
    }


def _elevated_bounds() -> dict[str, tuple[float, float]]:
    return {
        "cpu_usage": (CPU_ELEVATED_LOW, CPU_ELEVATED_HIGH),
        "memory_usage": (MEMORY_ELEVATED_LOW, MEMORY_ELEVATED_HIGH),
        "temperature": (TEMP_ELEVATED_LOW, TEMP_ELEVATED_HIGH),
        "network_traffic": (NETWORK_ELEVATED_LOW, NETWORK_ELEVATED_HIGH),
        "disk_io": (DISK_ELEVATED_LOW, DISK_ELEVATED_HIGH),
        "error_rate": (ERROR_ELEVATED_LOW, ERROR_ELEVATED_HIGH),
        "response_time": (RESPONSE_ELEVATED_LOW, RESPONSE_ELEVATED_HIGH),
    }


def _build_feature_block(
    n_rows: int,
    bounds: dict[str, tuple[float, float]],
    rng: np.random.Generator,
) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for name in FEATURE_ORDER:
        low, high = bounds[name]
        data[name] = _sample_feature(name, low, high, n_rows, rng)
    return pd.DataFrame(data)


def generate_dataset(n_samples: int, seed: int) -> pd.DataFrame:
    """
    Orquesta la generación del dataset.

    - Asigna ``int(n_samples * NORMAL_BUCKET_FRACTION)`` filas a rangos normales
      y el resto a rangos elevados (sesgo a fallo).
    - Calcula ``failure`` con las reglas R1–R4.
    - Mezcla filas con generador fijado por ``seed``.
    - Añade ``timestamp`` desde ``TIMESTAMP_START`` con paso ``TIMESTAMP_FREQ``,
      permutado con el mismo orden que las filas.
    """
    rng = np.random.default_rng(seed)
    n_normal = int(NORMAL_BUCKET_FRACTION * n_samples)
    n_elevated = n_samples - n_normal

    df_normal = _build_feature_block(n_normal, _normal_bounds(), rng)
    df_elevated = _build_feature_block(n_elevated, _elevated_bounds(), rng)
    df = pd.concat([df_normal, df_elevated], ignore_index=True)

    df["failure"] = df.apply(_is_failure, axis=1)

    perm = rng.permutation(n_samples)
    df = df.iloc[perm].reset_index(drop=True)

    base_ts = pd.date_range(
        start=TIMESTAMP_START,
        periods=n_samples,
        freq=TIMESTAMP_FREQ,
    )
    df.insert(0, "timestamp", base_ts[perm].values)

    return df


def save_dataset(df: pd.DataFrame, path: Path) -> None:
    """Guarda el CSV y muestra resumen de filas, fallos, normales y ``describe()``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

    total = len(df)
    n_fail = int(df["failure"].sum())
    n_ok = total - n_fail
    pct_fail = 100.0 * n_fail / total if total else 0.0
    pct_ok = 100.0 * n_ok / total if total else 0.0

    print(f"✅ Dataset guardado en: {path}")
    print(f"   Filas totales : {total:,}")
    print(f"   Fallos (1)    : {n_fail:,}  ({pct_fail:.1f}%)")
    print(f"   Normales (0)  : {n_ok:,}  ({pct_ok:.1f}%)")
    print(df.describe())


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (OSError, ValueError):
            pass
    print("🔧 Generando dataset de telemetría TechStream...")
    dataset = generate_dataset(N_SAMPLES, SEED)
    save_dataset(dataset, OUTPUT_PATH)
