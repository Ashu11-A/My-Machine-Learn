# Diabetes ML Pipeline — Folder Structure Guide

```
diabetes_ml/
│
│   # ── Entrypoints ──────────────────────────────────────────────────────
├── main.py                      # $ python main.py  ← único ponto de entrada
├── config.py                    # PipelineConfig  (dataclass frozen, imutável)
├── pipeline.py                  # DiabetesMLPipeline  (orquestrador de alto nível)
│
│   # ── Camada de Dados ───────────────────────────────────────────────────
├── data/
│   ├── __init__.py              # expõe: ProcessedDataset, DataPipeline
│   ├── dataset.py               # ProcessedDataset  (contêiner CPU+GPU)
│   └── pipeline.py              # DataPipeline  (load → split → scale)
│
│   # ── Camada de Treinamento ─────────────────────────────────────────────
├── training/
│   ├── __init__.py              # expõe: EarlyStopping, EarlyStoppingState,
│   │                            #         GPUModelWrapper, KNNWrapper,
│   │                            #         RandomForestWrapper, GradientBoostingWrapper,
│   │                            #         HyperparameterTuner
│   ├── early_stopping.py        # EarlyStoppingState + EarlyStopping (min_delta)
│   ├── wrappers.py              # Strategy: GPUModelWrapper + 3 implementações
│   └── tuner.py                 # HyperparameterTuner  (loop de busca)
│
│   # ── Camada de Visualização ────────────────────────────────────────────
└── visualization/
    ├── __init__.py              # expõe: DecisionBoundaryGrid, SubplotArtists,
    │                            #         ModelSubplotBuilder, TuningPlotBuilder,
    │                            #         PlotInteractionController
    ├── grid.py                  # DecisionBoundaryGrid  (malha 3D na GPU)
    ├── subplots.py              # SubplotArtists + ModelSubplotBuilder
    ├── tuning_plot.py           # TuningPlotBuilder  (acurácia vs parâmetro)
    └── interaction.py           # PlotInteractionController  (eventos matplotlib)
```

## Dependency Graph

```
main.py
 └── pipeline.py  (DiabetesMLPipeline)
      ├── config.py              (PipelineConfig)
      ├── data/
      │    ├── dataset.py        (ProcessedDataset)
      │    └── pipeline.py       (DataPipeline)
      ├── training/
      │    ├── early_stopping.py (EarlyStoppingState, EarlyStopping)
      │    ├── wrappers.py       (GPUModelWrapper, KNNWrapper, RFWrapper, GBWrapper)
      │    └── tuner.py          (HyperparameterTuner)
      └── visualization/
           ├── grid.py           (DecisionBoundaryGrid)
           ├── subplots.py       (SubplotArtists, ModelSubplotBuilder)
           ├── tuning_plot.py    (TuningPlotBuilder)
           └── interaction.py    (PlotInteractionController)
```

## Adding a new classifier

1. Create a new subclass of `GPUModelWrapper` in `training/wrappers.py`
2. Register it in the `wrappers` list inside `DiabetesMLPipeline.run()` (`pipeline.py`)
3. Add its display color/label in `TuningPlotBuilder` (`visualization/tuning_plot.py`)

No other files need to change.
