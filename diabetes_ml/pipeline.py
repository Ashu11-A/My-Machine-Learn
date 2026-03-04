"""
diabetes_ml/pipeline.py
-------------------------
Orquestrador de alto nível do pipeline de ML.

Ordem de importação obrigatória
--------------------------------
vispy.app.use_app() deve ser chamado antes de qualquer importação Qt.
Por isso, a primeira coisa que este módulo faz é configurar o backend
vispy. O restante das importações segue depois.
"""

from __future__ import annotations

import sys

# ── Backend vispy — DEVE ser definido antes de qualquer import Qt ──────────
from vispy import app as _vispy_app
_vispy_app.use_app('pyqt5')   # troque por 'pyqt6' se o ambiente usar Qt6

# ── Agora é seguro importar Qt e matplotlib ────────────────────────────────
try:
    from PyQt5.QtWidgets import QApplication
except ImportError:
    from PyQt6.QtWidgets import QApplication         # type: ignore[no-redef]

import cupy as cp
import numpy as np
from vispy import scene

from diabetes_ml.config import PipelineConfig
from diabetes_ml.data import DataPipeline, ProcessedDataset
from diabetes_ml.training import (
    EarlyStoppingState,
    GPUModelWrapper,
    GradientBoostingWrapper,
    HyperparameterTuner,
    KNNWrapper,
    RandomForestWrapper,
)
from diabetes_ml.visualization import (
    DecisionBoundaryGrid,
    DiabetesMLWindow,
    GPUScatterRow,
    ModelSubplotBuilder,
    ScatterViewState,
)


class DiabetesMLPipeline:
    """
    Orquestrador de alto nível: conecta todas as camadas do projeto.

    Fluxo: dados → tuning (GPU) → melhores modelos → visualização (OpenGL GPU)

    Parameters
    ----------
    config : PipelineConfig, optional
        Configuração customizada. Usa valores padrão se omitida.
    """

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()

    # ── Público ───────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Executa o pipeline completo e abre a janela de visualização."""

        # 1. Dados
        dataset = DataPipeline(self.config).build()

        # 2. Busca de hiperparâmetros com Early Stopping
        wrappers = self._default_wrappers()
        states   = HyperparameterTuner(wrappers, dataset, self.config).run()
        best_models = self._build_best_models(wrappers, states)

        # 3. Grade de fronteira de decisão
        grid     = DecisionBoundaryGrid(self.config)
        grid_pos = np.column_stack((
            grid.flat_insulin, grid.flat_glucose, grid.flat_bmi
        )).astype(np.float32)

        # 4. Câmera compartilhada — única instância = sincronização perfeita
        shared_camera = scene.cameras.TurntableCamera(
            fov=40.0, elevation=25.0, azimuth=45.0, distance=4.5
        )

        n_models  = len(best_models)
        train_row = GPUScatterRow(n_models, shared_camera)
        test_row  = GPUScatterRow(n_models, shared_camera)

        # 5. Construção dos scatter plots (envia dados para a VRAM via OpenGL)
        builder     = ModelSubplotBuilder(train_row, test_row, dataset)
        all_views:  list[ScatterViewState] = []
        model_names: list[str] = []
        accs_train:  list[float] = []
        accs_test:   list[float] = []

        for col, (model_name, model) in enumerate(best_models.items()):
            model.fit(dataset.X_train_gpu, dataset.y_train_gpu)

            pred_train = cp.asnumpy(model.predict(dataset.X_train_gpu)).astype(np.int32)
            pred_test  = cp.asnumpy(model.predict(dataset.X_test_gpu)).astype(np.int32)
            grid_preds = cp.asnumpy(model.predict(grid.X_grid_gpu)).astype(np.int32)

            acc_train = float(np.mean(pred_train == dataset.target_train))
            acc_test  = float(np.mean(pred_test  == dataset.target_test))

            sv_train, sv_test = builder.build(
                col, model_name, grid_pos, grid_preds, pred_train, pred_test
            )
            all_views.extend([sv_train, sv_test])
            model_names.append(model_name)
            accs_train.append(acc_train)
            accs_test.append(acc_test)

        # 6. Qt Application + janela principal
        qt_app = QApplication.instance() or QApplication(sys.argv)

        window = DiabetesMLWindow(
            train_row     = train_row,
            test_row      = test_row,
            all_views     = all_views,
            dataset       = dataset,
            tuning_states = states,
            min_delta     = self.config.min_delta,
            model_names   = model_names,
            acc_train     = accs_train,
            acc_test      = accs_test,
        )
        window.show()
        qt_app.exec()

    # ── Privado ───────────────────────────────────────────────────────────────

    @staticmethod
    def _default_wrappers() -> list[GPUModelWrapper]:
        return [KNNWrapper(), RandomForestWrapper(), GradientBoostingWrapper()]

    @staticmethod
    def _build_best_models(
        wrappers: list[GPUModelWrapper],
        states:   dict[str, EarlyStoppingState],
    ) -> dict[str, object]:
        mapping = {w.name: w for w in wrappers}
        return {
            f"{name} (param={state.best_param})": mapping[name].build(state.best_param)
            for name, state in states.items()
        }