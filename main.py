"""
main.py
-------
Ponto de entrada único do projeto.

Uso
---
    python main.py
    uv run main.py

Para customizar a configuração sem editar o código:

    from diabetes_ml.config import PipelineConfig
    from diabetes_ml.pipeline import DiabetesMLPipeline

    cfg = PipelineConfig(patience_limit=200, min_delta=0.005)
    DiabetesMLPipeline(cfg).run()
"""

from diabetes_ml.pipeline import DiabetesMLPipeline

if __name__ == "__main__":
    DiabetesMLPipeline().run()
