"""
diabetes_ml/visualization/subplots.py
---------------------------------------
Preenche os ScatterViewState com vispy Markers renderizados na GPU.

Cada ponto do scatter é enviado diretamente para a VRAM via OpenGL,
usando o mesmo buffer CuPy→NumPy já calculado pelo pipeline de treino.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from vispy.scene import visuals

from diabetes_ml.data.dataset import ProcessedDataset
from diabetes_ml.visualization.gpu_canvas import (
    GPUScatterRow,
    ScatterViewState,
    hex_to_rgba,
)


# ── Paleta de cores ───────────────────────────────────────────────────────────

_PALETTE: dict[str, str] = {
    'class_0': '#3d9bff',   # classe 0 correta  → azul
    'class_1': '#ff4d4d',   # classe 1 correta  → vermelho
    'error_0': '#00ffaa',   # classe 0 errada   → verde-menta
    'error_1': '#ffee00',   # classe 1 errada   → amarelo
}


# ── Helpers de cor ────────────────────────────────────────────────────────────

def _label_colors(labels: np.ndarray, kind: str, alpha: float = 1.0) -> np.ndarray:
    """Converte array de rótulos inteiros em (N, 4) RGBA float32."""
    out = np.empty((len(labels), 4), dtype=np.float32)
    for i, lbl in enumerate(labels):
        out[i] = hex_to_rgba(_PALETTE[f'{kind}_{int(lbl)}'], alpha)
    return out


def _mixed_colors(actual: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Cor de acerto para pontos corretos, cor de erro para incorretos."""
    out = np.empty((len(actual), 4), dtype=np.float32)
    for i, (a, p) in enumerate(zip(actual, predicted)):
        kind = 'class' if a == p else 'error'
        out[i] = hex_to_rgba(_PALETTE[f'{kind}_{int(a)}'])
    return out


# ── Construtor de subplots ────────────────────────────────────────────────────

class ModelSubplotBuilder:
    """
    Instancia os visuals vispy (Markers) para cada modelo nas linhas
    de treino e teste de dois GPUScatterRow.

    Os pontos são enviados para a VRAM via OpenGL na primeira chamada
    de set_data — todo o rendering subsequente ocorre na GPU.
    """

    _BG_ALPHA = 0.045  # transparência da fronteira de decisão
    _BG_SIZE  = 4.5    # tamanho dos pontos de background (px)
    _PT_SIZE  = 8.0    # tamanho dos pontos de dados (px)
    _ERR_SIZE = 9.5    # tamanho dos pontos de erro (px)

    def __init__(
        self,
        train_row: GPUScatterRow,
        test_row:  GPUScatterRow,
        dataset:   ProcessedDataset,
    ) -> None:
        self.train_row = train_row
        self.test_row  = test_row
        self.dataset   = dataset

    # ── Público ───────────────────────────────────────────────────────────────

    def build(
        self,
        col:        int,
        model_name: str,
        grid_pos:   np.ndarray,   # (N, 3) float32 — posições da malha
        grid_preds: np.ndarray,   # (N,)   int32   — predições na malha
        pred_train: np.ndarray,
        pred_test:  np.ndarray,
    ) -> tuple[ScatterViewState, ScatterViewState]:
        sv_train = self.train_row.views[col]
        sv_test  = self.test_row.views[col]

        self._fill(sv_train, model_name, 'train', grid_pos, grid_preds, pred_train)
        self._fill(sv_test,  model_name, 'test',  grid_pos, grid_preds, pred_test)

        return sv_train, sv_test

    # ── Privado ───────────────────────────────────────────────────────────────

    def _fill(
        self,
        sv:          ScatterViewState,
        model_name:  str,
        split:       str,
        grid_pos:    np.ndarray,
        grid_preds:  np.ndarray,
        predictions: np.ndarray,
    ) -> None:
        ds       = self.dataset
        is_train = (split == 'train')
        features = ds.features_train if is_train else ds.features_test
        targets  = ds.target_train   if is_train else ds.target_test
        raw_df   = ds.features_train_raw if is_train else ds.features_test_raw

        sv.dataset_type  = split
        sv.model_name    = model_name
        sv.feat_raw_ref  = raw_df
        sv.target_ref    = targets
        sv.pred_ref      = predictions

        error_mask       = targets != predictions
        sv.error_indices = np.where(error_mask)[0]

        # 1. Background: fronteira de decisão (muito transparente)
        bg_colors = _label_colors(grid_preds, 'class', self._BG_ALPHA)
        sv.bg_markers = self._markers(sv.view, grid_pos, bg_colors, self._BG_SIZE)

        # 2. Pontos padrão
        pt_colors = (
            _label_colors(targets, 'class')
            if is_train
            else _mixed_colors(targets, predictions)
        )
        sv.pt_markers = self._markers(sv.view, features, pt_colors, self._PT_SIZE)

        # 3. Pontos de erro (ocultos por padrão)
        err_feats  = features[error_mask]
        err_colors = _label_colors(targets[error_mask], 'error')
        sv.err_markers = self._markers(
            sv.view, err_feats, err_colors, self._ERR_SIZE, visible=False
        )

    @staticmethod
    def _markers(
        view:       Any,
        pos:        np.ndarray,
        face_color: np.ndarray,
        size:       float,
        visible:    bool = True,
    ) -> visuals.Markers:
        """
        Cria um Markers visual e envia os dados para a GPU via OpenGL.
        Arrays vazios recebem um ponto fantasma com alpha=0.
        """
        if pos.shape[0] == 0:
            pos        = np.zeros((1, 3), dtype=np.float32)
            face_color = np.zeros((1, 4), dtype=np.float32)

        m = visuals.Markers(parent=view.scene)
        m.set_data(
            pos.astype(np.float32),
            face_color=face_color,
            edge_width=0,
            size=size,
        )
        m.visible = visible
        return m