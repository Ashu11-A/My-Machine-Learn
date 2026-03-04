"""
diabetes_ml/training/early_stopping.py
----------------------------------------
Implementação do Early Stopping com tolerância mínima de melhoria.

Classes
-------
EarlyStoppingState
    Estado mutável de um único modelo durante a busca.
EarlyStopping
    Lógica stateless que atualiza EarlyStoppingState a cada step.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStoppingState:
    """Estado do Early Stopping para um único modelo."""

    best_acc: float = 0.0
    best_param: int = 1
    patience: int = 0
    active: bool = True
    test_acc: list[float] = field(default_factory=list)
    params: list[int] = field(default_factory=list)


class EarlyStopping:
    """
    Controla a parada antecipada com tolerância mínima de melhoria.

    Parameters
    ----------
    patience_limit : int
        Número máximo de steps consecutivos sem melhoria antes de parar.
    min_delta : float
        Melhoria mínima considerada significativa.
        Se a diferença ``acc - best_acc <= min_delta``, o passo é
        tratado como estagnação e a paciência é incrementada.
    """

    def __init__(self, patience_limit: int, min_delta: float = 0.01) -> None:
        self.patience_limit = patience_limit
        self.min_delta = min_delta

    def step(self, state: EarlyStoppingState, acc: float, param: int) -> None:
        """Registra o resultado do step e atualiza o estado."""
        state.test_acc.append(acc)
        state.params.append(param)

        improvement = acc - state.best_acc
        if improvement > self.min_delta:
            state.best_acc = acc
            state.best_param = param
            state.patience = 0
        else:
            state.patience += 1
            if state.patience >= self.patience_limit:
                state.active = False

    def is_active(self, state: EarlyStoppingState) -> bool:
        """Retorna True se o modelo ainda deve continuar sendo treinado."""
        return state.active
