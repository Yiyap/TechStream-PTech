# 🖥️ TechStream Anomaly Detection System

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Descripción

Sistema de detección de anomalías en telemetría de servidores usando una red neuronal **MLP** implementada en **PyTorch**. Resuelve una **clasificación binaria** (operación normal vs. fallo) a partir de siete variables numéricas escaladas. El pipeline incluye generación sintética de datos con reglas físicas, partición estratificada, normalización con `StandardScaler`, manejo del **desbalance** de clases (muestreo ponderado y `pos_weight` en la pérdida) y evaluación con curvas ROC/PR.

## Resultados obtenidos

Ejecuta `python src/evaluate.py` tras entrenar y completa la tabla con los valores impresos (conjunto de **test**, umbral 0,5).

| Métrica   | Valor        |
|-----------|--------------|
| Accuracy  | [RESULTADO]  |
| Precision | [RESULTADO]  |
| Recall    | [RESULTADO]  |
| F1-Score  | [RESULTADO]  |
| ROC-AUC   | [RESULTADO]  |

## Arquitectura del modelo

```
Input (7) → Dense(128)+BN+ReLU+Drop → Dense(64)+BN+ReLU+Drop → Dense(32)+BN+ReLU+Drop → Output(1)
```

La salida es un **logit** (sin sigmoid en el modelo); en entrenamiento se usa `BCEWithLogitsLoss`; en inferencia se aplica `sigmoid` para obtener la probabilidad de fallo.

## Instalación

```bash
git clone https://github.com/Yiyap/TechStream-PTech.git
cd TechStream-PTech
pip install -r requirements.txt
```

## Uso paso a paso

```bash
# 1. Generar datos
python src/data_generator.py

# 2. Entrenar modelo
python src/train.py

# 3. Evaluar modelo
python src/evaluate.py

# 4. Notebook de presentación
jupyter notebook notebooks/presentation.ipynb
```

> Ejecuta los comandos desde la raíz del repositorio (donde están `data/`, `src/`, `models/` y `notebooks/`).

## Estructura del proyecto

```
TechStream-PTech/
├── data/
│   └── sensors.csv              # Telemetría sintética (features + failure)
├── figures/                     # Figuras exportadas (EDA, evaluación, etc.)
├── models/
│   ├── model.pth                # Pesos del mejor checkpoint (menor val_loss)
│   └── training_history.json    # Historial train/val por época
├── notebooks/
│   └── presentation.ipynb       # Presentación técnica (EDA → inferencia)
├── src/
│   ├── data_generator.py        # Genera sensors.csv con reglas de fallo
│   ├── dataset.py               # Dataset PyTorch, splits, scaler, DataLoaders
│   ├── model.py                 # AnomalyDetector (MLP + BN + Dropout)
│   ├── train.py                 # Entrenamiento, early stopping, scheduler
│   └── evaluate.py              # Métricas en test y figuras de evaluación
├── requirements.txt             # Dependencias Python
└── README.md                    # Este archivo
```

## Decisiones de diseño

**MLP frente a LSTM:** cada fila del dataset es un *snapshot* instantáneo de sensores, no una ventana temporal alineada por servidor. Una MLP captura bien las combinaciones no lineales entre variables (CPU, memoria, temperatura, etc.) con menos complejidad y datos. Una LSTM tendría sentido si se alimentaran secuencias temporales explícitas (por ejemplo, ventanas de *k* lecturas por nodo).

**BCEWithLogitsLoss con `pos_weight`:** la clase “fallo” suele ser minoritaria. `pos_weight = n_negativos / n_positivos` compensa el gradiente hacia los positivos sin duplicar filas, y encaja con logits sin sigmoid en la capa final (estabilidad numérica frente a `BCELoss` + `sigmoid`).

**Prioridad del Recall:** en monitorización de infraestructura, un falso negativo (fallo no detectado) suele ser más costoso que un falso positivo (alarma revisable). Por eso se enfatiza un **Recall** alto junto con un análisis de umbral; la precisión se equilibra con el coste operativo de alertas.

**BatchNorm y Dropout juntos:** BatchNorm1d estabiliza la escala de activaciones por capa y acelera la convergencia; Dropout reduce el sobreajuste cuando el modelo memoriza patrones del conjunto mayoritario. Se combinan en cada bloque oculto para regularizar sin renunciar a entrenamiento estable.

## Autor

**Javier León Soro** — [javierleonsoro@gmail.com](mailto:javierleonsoro@gmail.com)
