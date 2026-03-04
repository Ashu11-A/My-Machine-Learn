"""
diabetes_ml/data/dataset.py
----------------------------
Contêiner de dados pré-processados.
Armazena os splits CPU (NumPy) e GPU (CuPy) em um único objeto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cupy as cp
import numpy as np
import pandas as pd


@dataclass
class ProcessedDataset:
    """
    Contêiner com todos os splits já escalonados e prontos para GPU.

    Os arrays GPU são criados automaticamente no __post_init__
    a partir dos arrays NumPy recebidos.
    """

    # Arrays CPU (NumPy)
    features_train: np.ndarray
    features_test: np.ndarray
    target_train: np.ndarray
    target_test: np.ndarray

    # DataFrames originais (para exibição de informações nos tooltips)
    features_train_raw: pd.DataFrame
    features_test_raw: pd.DataFrame

    # Arrays GPU (CuPy) — inicializados automaticamente
    X_train_gpu: cp.ndarray = field(init=False)
    X_test_gpu: cp.ndarray = field(init=False)
    y_train_gpu: cp.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.X_train_gpu = cp.asarray(self.features_train)
        self.X_test_gpu = cp.asarray(self.features_test)
        self.y_train_gpu = cp.asarray(self.target_train)
