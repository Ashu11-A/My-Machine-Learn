"""
diabetes_ml/visualization/interaction.py
------------------------------------------
Janela Qt principal que integra:
  - Dois GPUScatterRow (vispy OpenGL) para os gráficos 3D
  - Um FigureCanvas matplotlib para o gráfico de Fine Tuning
  - Labels Qt para títulos e acurácias
  - Botões de Reset e Modo Erros

Sincronização de câmera
-----------------------
A sincronização é garantida em nível de objeto Python: os dois canvases
vispy recebem a *mesma instância* de TurntableCamera. Qualquer interação
em qualquer canvas propaga imediatamente para todos os outros porque
compartilham o mesmo estado de câmera.

Picking (clique em pontos)
--------------------------
O clique detecado pelo evento mouse_press do canvas vispy projeta os
pontos 3D para coordenadas de tela via a transformada da câmera e
encontra o vizinho mais próximo por distância euclidiana 2D.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# PyQt — compatível com Qt5 e Qt6 via vispy's app abstraction
try:
    from PyQt5.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSizePolicy, QFrame,
    )
    from PyQt5.QtCore import Qt
except ImportError:
    from PyQt6.QtWidgets import (                               # type: ignore[no-redef]
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QLabel, QSizePolicy, QFrame,
    )
    from PyQt6.QtCore import Qt                                 # type: ignore[no-redef]

from diabetes_ml.data.dataset import ProcessedDataset
from diabetes_ml.training.early_stopping import EarlyStoppingState
from diabetes_ml.visualization.gpu_canvas import GPUScatterRow, ScatterViewState
from diabetes_ml.visualization.tuning_plot import TuningPlotBuilder


# ── Estilos ───────────────────────────────────────────────────────────────────

_DARK_BG   = '#0b0b18'
_MID_BG    = '#11112a'
_ACCENT    = '#2a2a5a'
_BORDER    = '#3a3a7a'
_TEXT_CLR  = '#c8c8e8'

_LBL_STYLE = f"""
    QLabel {{
        color: {_TEXT_CLR};
        font-size: 11px;
        font-weight: bold;
        background-color: {_MID_BG};
        padding: 5px 10px;
        border-bottom: 1px solid {_BORDER};
    }}
"""

_BTN_STYLE = f"""
    QPushButton {{
        background-color: {_ACCENT};
        color: {_TEXT_CLR};
        border: 1px solid {_BORDER};
        border-radius: 6px;
        padding: 7px 24px;
        font-size: 13px;
        min-width: 140px;
    }}
    QPushButton:hover   {{ background-color: #3a3a7a; border-color: #6666cc; }}
    QPushButton:pressed {{ background-color: #1a1a3a; }}
"""

_INFO_STYLE = f"""
    QLabel {{
        color: {_TEXT_CLR};
        font-size: 11px;
        background-color: {_MID_BG};
        padding: 4px 12px;
        border-top: 1px solid {_BORDER};
    }}
"""


# ── Janela principal ──────────────────────────────────────────────────────────

class DiabetesMLWindow(QMainWindow):
    """
    Janela Qt que orquestra toda a visualização do pipeline.

    Parameters
    ----------
    train_row, test_row : GPUScatterRow
        Canvases vispy com gráficos 3D já populados.
    all_views : list[ScatterViewState]
        Todas as views (train + test) para controle de modo diff.
    dataset : ProcessedDataset
        Necessário para o tooltip de informação de pontos.
    tuning_states : dict[str, EarlyStoppingState]
        Para o gráfico de Fine Tuning.
    model_names : list[str]
        Nomes curtos dos modelos (ex: ['KNN (param=7)', ...]).
    acc_train, acc_test : list[float]
        Acurácias finais de cada modelo em treino e teste.
    """

    def __init__(
        self,
        train_row:     GPUScatterRow,
        test_row:      GPUScatterRow,
        all_views:     list[ScatterViewState],
        dataset:       ProcessedDataset,
        tuning_states: dict[str, EarlyStoppingState],
        min_delta:     float,
        model_names:   list[str],
        acc_train:     list[float],
        acc_test:      list[float],
    ) -> None:
        super().__init__()
        self.train_row    = train_row
        self.test_row     = test_row
        self.all_views    = all_views
        self.dataset      = dataset
        self.is_diff_mode = False
        self._info_text   = "Clique em um ponto para ver informações"

        self.setWindowTitle('Diabetes ML — GPU Visualization')
        self.setStyleSheet(f'background-color: {_DARK_BG};')
        self.resize(1440, 960)

        self._build_ui(tuning_states, min_delta, model_names, acc_train, acc_test)
        self._connect_picking()

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build_ui(
        self,
        tuning_states: dict[str, EarlyStoppingState],
        min_delta:     float,
        model_names:   list[str],
        acc_train:     list[float],
        acc_test:      list[float],
    ) -> None:
        root = QWidget()
        root.setStyleSheet(f'background-color: {_DARK_BG};')
        self.setCentralWidget(root)

        vlay = QVBoxLayout(root)
        vlay.setContentsMargins(4, 4, 4, 4)
        vlay.setSpacing(2)

        # ── Linha de Treino ───────────────────────────────────────────────
        vlay.addLayout(self._title_row(model_names, acc_train, prefix='Treino'))
        vlay.addWidget(self.train_row.native, stretch=4)

        # ── Separador ─────────────────────────────────────────────────────
        vlay.addWidget(self._separator())

        # ── Linha de Teste ────────────────────────────────────────────────
        vlay.addLayout(self._title_row(model_names, acc_test, prefix='Teste'))
        vlay.addWidget(self.test_row.native, stretch=4)

        # ── Fine Tuning (matplotlib) ──────────────────────────────────────
        vlay.addWidget(self._separator())
        vlay.addWidget(self._tuning_widget(tuning_states, min_delta), stretch=2)

        # ── Barra de informação de ponto ─────────────────────────────────
        self._info_label = QLabel(self._info_text)
        self._info_label.setStyleSheet(_INFO_STYLE)
        self._info_label.setAlignment(Qt.AlignCenter)
        vlay.addWidget(self._info_label)

        # ── Botões ────────────────────────────────────────────────────────
        vlay.addLayout(self._button_row())

    def _title_row(
        self, model_names: list[str], accs: list[float], prefix: str
    ) -> QHBoxLayout:
        hlay = QHBoxLayout()
        hlay.setSpacing(2)
        hlay.setContentsMargins(0, 0, 0, 0)
        for name, acc in zip(model_names, accs):
            lbl = QLabel(f'{prefix}: {name}    Acc: {acc:.4f}')
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(_LBL_STYLE)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            hlay.addWidget(lbl)
        return hlay

    def _tuning_widget(
        self,
        states:    dict[str, EarlyStoppingState],
        min_delta: float,
    ) -> FigureCanvas:
        fig = Figure(figsize=(14, 2.8), facecolor=_DARK_BG)
        ax  = fig.add_subplot(111, facecolor=_MID_BG)
        fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.18)

        # Estilo escuro para o gráfico matplotlib
        for spine in ax.spines.values():
            spine.set_edgecolor(_BORDER)
        ax.tick_params(colors=_TEXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(_TEXT_CLR)
        ax.yaxis.label.set_color(_TEXT_CLR)
        ax.title.set_color(_TEXT_CLR)
        ax.grid(color=_BORDER, linestyle=':', alpha=0.5)

        TuningPlotBuilder().build(ax, states, min_delta)

        canvas = FigureCanvas(fig)
        canvas.setStyleSheet(f'background-color: {_DARK_BG};')
        return canvas

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f'color: {_BORDER}; background-color: {_BORDER};')
        line.setFixedHeight(1)
        return line

    def _button_row(self) -> QHBoxLayout:
        hlay = QHBoxLayout()
        hlay.setContentsMargins(0, 6, 0, 6)

        self._reset_btn = QPushButton('🔄  Resetar Visão')
        self._diff_btn  = QPushButton('🔍  Ver Erros')
        self._legend    = QLabel(
            '  🔵 Classe 0 correta   🔴 Classe 1 correta   '
            '🟢 Classe 0 errada   🟡 Classe 1 errada'
        )
        self._legend.setStyleSheet(f'color: {_TEXT_CLR}; font-size: 10px;')

        for btn in (self._reset_btn, self._diff_btn):
            btn.setStyleSheet(_BTN_STYLE)
            btn.setFixedHeight(38)

        self._reset_btn.clicked.connect(self._on_reset)
        self._diff_btn.clicked.connect(self._on_toggle_diff)

        hlay.addWidget(self._legend)
        hlay.addStretch()
        hlay.addWidget(self._reset_btn)
        hlay.addSpacing(12)
        hlay.addWidget(self._diff_btn)
        hlay.addSpacing(20)
        return hlay

    # ── Picking ───────────────────────────────────────────────────────────────

    def _connect_picking(self) -> None:
        """Conecta o evento de clique nos dois canvases vispy."""
        for row in (self.train_row, self.test_row):
            row.canvas.events.mouse_press.connect(
                lambda e, r=row: self._on_canvas_click(e, r)
            )

    def _on_canvas_click(self, event: Any, row: GPUScatterRow) -> None:
        """
        Detecta o ponto mais próximo ao clique e exibe suas informações.

        Estratégia: projeta as coordenadas 3D de cada ponto para o espaço
        de tela (pixels) usando a transformada da câmera e encontra o
        vizinho mais próximo por distância euclidiana 2D.
        """
        if event.button != 1:   # somente botão esquerdo
            return

        click_pos = np.array(event.pos[:2], dtype=np.float64)

        # Descobre qual ViewBox foi clicado
        target_sv = self._find_view_at(click_pos, row)
        if target_sv is None:
            return

        sv = target_sv
        features = (
            self.dataset.features_train if sv.dataset_type == 'train'
            else self.dataset.features_test
        )

        # Tenta projetar pontos 3D → tela
        try:
            tr = sv.view.scene.transform
            screen_pts = tr.map(features)   # (N, 4) homogêneas
            screen_2d  = screen_pts[:, :2]

            dists = np.linalg.norm(screen_2d - click_pos, axis=1)
            nearest = int(np.argmin(dists))

            if dists[nearest] > 30:   # threshold em pixels
                return

            self._show_point_info(sv, nearest)
        except Exception:
            pass   # picking não-essencial; falha silenciosa

    @staticmethod
    def _find_view_at(
        click_pos: np.ndarray, row: GPUScatterRow
    ) -> ScatterViewState | None:
        """Retorna o ScatterViewState cujo ViewBox contém a posição de clique."""
        for sv in row.views:
            rect = sv.view.rect
            x, y, w, h = rect.left, rect.bottom, rect.width, rect.height
            if x <= click_pos[0] <= x + w and y <= click_pos[1] <= y + h:
                return sv
        return None

    def _show_point_info(self, sv: ScatterViewState, idx: int) -> None:
        """Atualiza a barra de informação com os dados do ponto clicado."""
        real_idx = sv.error_indices[idx] if self.is_diff_mode else idx

        if real_idx >= len(sv.feat_raw_ref):
            return

        row     = sv.feat_raw_ref.iloc[real_idx]
        actual  = sv.target_ref[real_idx]
        pred    = sv.pred_ref[real_idx]
        correct = '✅ Sim' if actual == pred else '❌ Não'
        split   = 'Treino' if sv.dataset_type == 'train' else 'Teste'

        self._info_label.setText(
            f'[{split} · {sv.model_name}]   '
            f'Insulina: {row["Insulin"]:.1f}   '
            f'Glicose: {row["Glucose"]:.1f}   '
            f'IMC: {row["BMI"]:.1f}   '
            f'Classe Real: {actual}   '
            f'Previsão: {pred}   '
            f'Acertou: {correct}'
        )

    # ── Botões ────────────────────────────────────────────────────────────────

    def _on_reset(self) -> None:
        """
        Reseta a câmera compartilhada para a posição inicial.
        Como todos os ViewBoxes compartilham a mesma câmera, basta
        modificar uma única instância.
        """
        cam = self.train_row.views[0].view.camera
        cam.elevation = 25.0
        cam.azimuth   = 45.0
        cam.fov       = 40.0
        cam.distance  = 4.5
        self.train_row.canvas.update()
        self.test_row.canvas.update()

    def _on_toggle_diff(self) -> None:
        """Alterna entre modo padrão e modo de visualização de erros."""
        self.is_diff_mode = not self.is_diff_mode
        self.train_row.set_diff_mode(self.is_diff_mode)
        self.test_row.set_diff_mode(self.is_diff_mode)
        label = '🔙  Visão Padrão' if self.is_diff_mode else '🔍  Ver Erros'
        self._diff_btn.setText(label)
        self._info_label.setText(self._info_text)