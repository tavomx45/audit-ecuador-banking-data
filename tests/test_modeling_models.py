"""Pruebas unitarias de construcción de modelos."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

from src.modeling.models import construir_modelo


@pytest.mark.parametrize(
    "nombre,tipo_esperado",
    [
        ("Ridge", Ridge),
        ("RandomForest", RandomForestRegressor),
        ("HistGradientBoosting", HistGradientBoostingRegressor),
    ],
)
def test_fabrica_construye_modelos_correctos(
    nombre,
    tipo_esperado,
):
    """Cada nombre produce el estimador esperado."""

    modelo = construir_modelo(
        nombre,
        ["morosidad_pp", "roa"],
    )

    assert isinstance(modelo, Pipeline)

    estimador = modelo.named_steps["modelo"]

    assert isinstance(estimador, tipo_esperado)

    preprocesador = modelo.named_steps["preprocesador"]

    numericas = preprocesador.transformers[0][1]

    nombres_pasos = list(numericas.named_steps)

    # Todos utilizan imputación por mediana.
    assert "imputador" in nombres_pasos

    # Únicamente Ridge utiliza RobustScaler.
    if nombre == "Ridge":
        assert "escalador" in nombres_pasos
        assert estimador.alpha == 10.0
    else:
        assert "escalador" not in nombres_pasos

    # Configuraciones académicas originales.
    if nombre == "RandomForest":
        assert estimador.n_estimators == 100
        assert estimador.max_depth == 10
        assert estimador.min_samples_leaf == 5
        assert estimador.random_state == 42

    if nombre == "HistGradientBoosting":
        assert estimador.max_iter == 150
        assert estimador.learning_rate == 0.05
        assert estimador.max_leaf_nodes == 15
        assert estimador.random_state == 42


def test_imputacion_se_ajusta_solo_con_entrenamiento():
    """La mediana debe aprenderse durante fit()."""

    X_train = pd.DataFrame(
        {
            "x": [1.0, np.nan, 5.0],
            "banco": ["A", "A", "B"],
        }
    )

    y_train = np.array([1.0, 2.0, 3.0])

    modelo = construir_modelo(
        "Ridge",
        ["x"],
    )

    modelo.fit(X_train, y_train)

    preprocesador = modelo.named_steps["preprocesador"]

    imputador = preprocesador.named_transformers_["numericas"].named_steps["imputador"]

    # Mediana de los valores observados: (1 + 5) / 2.
    assert imputador.statistics_[0] == pytest.approx(3.0)

    # El banco C no estaba presente en entrenamiento.
    X_nuevo = pd.DataFrame(
        {
            "x": [np.nan],
            "banco": ["C"],
        }
    )

    prediccion = modelo.predict(X_nuevo)

    assert prediccion.shape == (1,)
    assert np.isfinite(prediccion).all()


def test_modelo_desconocido_es_rechazado():
    """No debemos aceptar nombres de modelos inválidos."""

    with pytest.raises(
        ValueError,
        match="Modelo desconocido",
    ):
        construir_modelo(
            "ModeloInexistente",
            ["morosidad_pp"],
        )
