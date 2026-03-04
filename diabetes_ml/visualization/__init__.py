"""
diabetes_ml.visualization
--------------------------
Camada de visualização GPU do pipeline.

Exporta os componentes principais para uso em pipeline.py.
"""

from diabetes_ml.visualization.gpu_canvas import GPUScatterRow, ScatterViewState
from diabetes_ml.visualization.interaction import DiabetesMLWindow
from diabetes_ml.visualization.subplots import ModelSubplotBuilder
from diabetes_ml.visualization.tuning_plot import TuningPlotBuilder
from diabetes_ml.visualization.grid import DecisionBoundaryGrid

__all__ = [
    "DecisionBoundaryGrid",
    "DiabetesMLWindow",
    "GPUScatterRow",
    "ModelSubplotBuilder",
    "ScatterViewState",
    "TuningPlotBuilder",
]