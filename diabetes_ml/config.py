"""
diabetes_ml/config.py
---------------------
Configuração central e imutável de todo o pipeline.
Altere os valores aqui para ajustar o comportamento sem tocar em
nenhuma outra parte do código.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineConfig:
    """Configuração imutável de todo o pipeline."""

    # Dados
    file_path: Path = Path("./diabetes.csv")
    feature_columns: tuple[str, ...] = ("Insulin", "Glucose", "BMI")
    target_column: str = "Outcome"
    test_size: float = 0.1
    random_state: int = 42
    scaler_range: tuple[float, float] = (-1.0, 1.0)

    # Early Stopping
    patience_limit: int = 150
    min_delta: float = 0.1   # tolerância mínima de melhoria
    initial_param: int = 20

    # Grade de fronteiras de decisão
    grid_resolution: int = 15
    grid_axis_range: tuple[float, float] = (-1.1, 1.1)
