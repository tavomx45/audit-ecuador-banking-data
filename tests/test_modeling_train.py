"""Pruebas del entrenamiento supervisado temporal."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from src.modeling.models import construir_modelo
from src.modeling.train import evaluar_modelos_cv


@pytest.fixture
def experimento_sintetico():
    """Crea una partición temporal pequeña y controlada."""

    fechas = pd.to_datetime(
        [
            "2020-06-01",
            "2020-07-01",
            "2020-08-01",
            "2020-09-01",
            "2020-10-01",
            "2021-01-01",
            "2021-02-01",
        ]
    )

    datos = pd.DataFrame(
        {
            "banco": ["A", "A", "B", "A", "B", "A", "B"],
            "fecha": fechas,
            "fecha_objetivo_t1": fechas + pd.DateOffset(months=1),
            "morosidad_pp": [2.0, 2.2, 3.0, 2.3, 3.2, 2.5, 3.4],
            "roa": [0.01, np.nan, 0.02, 0.01, 0.03, 0.02, 0.01],
            "target_morosidad_t1_pp": [2.2, 2.3, 3.1, 2.5, 3.4, 2.7, 3.6],
        }
    )

    datasets = {
        1: {
            "desarrollo": datos,
            "target": "target_morosidad_t1_pp",
            "fecha_target": "fecha_objetivo_t1",
            "numericas": ["morosidad_pp", "roa"],
            "predictoras": ["morosidad_pp", "roa", "banco"],
        }
    }

    folds = {
        1: [
            {
                "fold": 2021,
                "train_idx": np.arange(5),
                "val_idx": np.array([5, 6]),
            }
        ]
    }

    return datasets, folds


def test_entrenamiento_cv_reproduce_ridge(
    experimento_sintetico,
):
    """Comprueba entrenamiento, predicción y MAE."""

    datasets, folds = experimento_sintetico

    original = datasets[1]["desarrollo"].copy(deep=True)

    detalle, resumen = evaluar_modelos_cv(
        datasets,
        folds,
        nombres_modelos=("Ridge",),
    )

    assert len(detalle) == 1
    assert len(resumen) == 1

    assert detalle.loc[0, "horizonte"] == 1
    assert detalle.loc[0, "fold"] == 2021
    assert detalle.loc[0, "modelo"] == "Ridge"
    assert detalle.loc[0, "observaciones"] == 2
    assert resumen.loc[0, "folds"] == 1

    # Comparación con un ajuste independiente.
    datos = datasets[1]["desarrollo"]

    columnas = datasets[1]["predictoras"]

    modelo = construir_modelo(
        "Ridge",
        datasets[1]["numericas"],
    )

    modelo.fit(
        datos.loc[:4, columnas],
        datos.loc[:4, "target_morosidad_t1_pp"],
    )

    pred = modelo.predict(datos.loc[5:6, columnas])

    mae_esperado = mean_absolute_error(
        datos.loc[5:6, "target_morosidad_t1_pp"],
        pred,
    )

    assert detalle.loc[0, "mae_pp"] == pytest.approx(mae_esperado)

    # El entrenamiento no modifica los datos.
    pd.testing.assert_frame_equal(
        datos,
        original,
    )


def test_entrenamiento_rechaza_fuga_temporal(
    experimento_sintetico,
):
    """Las etiquetas de entrenamiento no pueden llegar al fold."""

    datasets, folds = experimento_sintetico

    datos = datasets[1]["desarrollo"].copy()

    # Introducir deliberadamente una etiqueta
    # del período de validación en entrenamiento.
    datos.loc[4, "fecha_objetivo_t1"] = pd.Timestamp("2021-01-01")

    datasets[1]["desarrollo"] = datos

    with pytest.raises(
        ValueError,
        match="Fuga temporal",
    ):
        evaluar_modelos_cv(
            datasets,
            folds,
            nombres_modelos=("Ridge",),
        )
