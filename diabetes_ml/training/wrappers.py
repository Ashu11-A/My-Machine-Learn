"""
diabetes_ml/training/wrappers.py
----------------------------------
Strategy Pattern para os classificadores GPU.

Para adicionar um novo modelo:
  1. Crie uma subclasse de GPUModelWrapper
  2. Implemente `build()` e `is_param_valid()`
  3. Registre a instância na lista `_default_wrappers()` em pipeline.py
  4. Adicione cor/label em TuningPlotBuilder (visualization/tuning_plot.py)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from cuml.ensemble import RandomForestClassifier
from cuml.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


class GPUModelWrapper(ABC):
    """Interface comum para todos os classificadores GPU."""

    #: Identificador curto usado em logs e estados de Early Stopping
    name: str

    @abstractmethod
    def build(self, param: int) -> Any:
        """Instancia o modelo com o hiperparâmetro ``param``."""

    @abstractmethod
    def is_param_valid(self, param: int, n_samples: int) -> bool:
        """Retorna True se ``param`` é um valor válido para este modelo."""


class KNNWrapper(GPUModelWrapper):
    """KNN via cuML — parâmetro = número de vizinhos (K)."""

    name = "KNN"
    _MAX_K: int = 1024

    def build(self, param: int) -> KNeighborsClassifier:
        return KNeighborsClassifier(n_neighbors=param)

    def is_param_valid(self, param: int, n_samples: int) -> bool:
        return param <= n_samples and param <= self._MAX_K


class RandomForestWrapper(GPUModelWrapper):
    """Random Forest via cuML — parâmetro = número de árvores."""

    name = "RF"

    def build(self, param: int) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=param,
            random_state=42,
            max_depth=5,
        )

    def is_param_valid(self, param: int, n_samples: int) -> bool:
        return True


class GradientBoostingWrapper(GPUModelWrapper):
    """Gradient Boosting via XGBoost (CUDA) — parâmetro = número de estimadores."""

    name = "GB"

    def build(self, param: int) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=param,
            random_state=42,
            max_depth=3,
            tree_method="hist",
            device="cuda",
            verbosity=0,
        )

    def is_param_valid(self, param: int, n_samples: int) -> bool:
        return True
