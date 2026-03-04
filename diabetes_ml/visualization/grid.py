"""
diabetes_ml/visualization/grid.py
-----------------------------------
Gera a malha 3D utilizada para desenhar as fronteiras de decisão nos
subplots. O cálculo é feito na CPU e o resultado é enviado para a GPU.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from diabetes_ml.config import PipelineConfig


class DecisionBoundaryGrid:
    """
    Malha 3D uniforme para visualização das fronteiras de decisão.

    Attributes
    ----------
    flat_insulin, flat_glucose, flat_bmi : np.ndarray
        Coordenadas achatadas de cada eixo (usadas pelo matplotlib scatter).
    X_grid_gpu : cp.ndarray
        Matriz (N, 3) na GPU para predição em lote.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._build()

    def _build(self) -> None:
        axis_vals = np.linspace(
            *self.config.grid_axis_range,
            self.config.grid_resolution,
            dtype=np.float32,
        )
        gi, gg, gb = np.meshgrid(axis_vals, axis_vals, axis_vals)
        self.flat_insulin: np.ndarray = gi.ravel()
        self.flat_glucose: np.ndarray = gg.ravel()
        self.flat_bmi: np.ndarray = gb.ravel()

        matrix = np.column_stack(
            (self.flat_insulin, self.flat_glucose, self.flat_bmi)
        ).astype(np.float32)
        self.X_grid_gpu: cp.ndarray = cp.asarray(matrix)
