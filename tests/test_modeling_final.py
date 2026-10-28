"""Pruebas del sistema híbrido de predicción."""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.modeling.forecast import (
    entrenar_y_predecir_horizonte,
)
from src.modeling.models import (
    ESTRATEGIA_POR_HORIZONTE,
    construir_rf_delta_flexible,
)


def test_estrategias_por_horizonte():
    """Comprueba las estrategias seleccionadas."""

    assert ESTRATEGIA_POR_HORIZONTE == {
        1: "RandomForest_delta_flexible",
        2: "Persistencia_estacional",
        3: "Persistencia_estacional",
    }


def test_hiperparametros_random_forest():
    """Comprueba que se aplica la optimización elegida."""

    modelo = construir_rf_delta_flexible(["morosidad_pp", "roa"])

    estimadores = [
        paso
        for paso in modelo.named_steps.values()
        if isinstance(paso, RandomForestRegressor)
    ]

    assert len(estimadores) == 1

    rf = estimadores[0]

    assert rf.n_estimators == 150
    assert rf.max_depth == 14
    assert rf.min_samples_leaf == 3
    assert rf.max_features == 0.7
    assert rf.random_state == 42


def test_horizonte_no_soportado():
    """El pronosticador debe rechazar horizontes inválidos."""

    with pytest.raises(
        ValueError,
        match="Horizonte no soportado",
    ):
        entrenar_y_predecir_horizonte(
            horizonte=4,
            entrenamiento=pd.DataFrame(),
            validacion=pd.DataFrame(),
            target="target",
            fecha_target="fecha_target",
            numericas=[],
            predictoras=[],
        )
