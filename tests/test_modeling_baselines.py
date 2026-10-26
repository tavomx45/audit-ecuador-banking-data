"""Pruebas unitarias de los modelos baseline."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.baselines import (
    aprender_ajustes_estacionales,
    predecir_persistencia,
    predecir_persistencia_estacional,
)


def test_persistencia_utiliza_morosidad_actual():
    """La persistencia devuelve los valores actuales."""

    datos = pd.DataFrame(
        {
            "morosidad_pp": [2.0, 3.5, 5.0],
        }
    )

    pred = predecir_persistencia(datos)

    np.testing.assert_array_equal(
        pred,
        np.array([2.0, 3.5, 5.0]),
    )


def test_ajustes_estacionales_utilizan_mediana():
    """Comprueba la mediana de cambios por mes objetivo."""

    train = pd.DataFrame(
        {
            "morosidad_pp": [4.0, 5.0, 6.0],
            "target_morosidad_t1_pp": [5.0, 6.0, 8.0],
            "fecha_objetivo_t1": pd.to_datetime(
                [
                    "2020-01-01",
                    "2021-01-01",
                    "2022-01-01",
                ]
            ),
        }
    )

    ajustes = aprender_ajustes_estacionales(
        train,
        "target_morosidad_t1_pp",
        "fecha_objetivo_t1",
    )

    # Cambios observados: 1, 1, 2.
    # Mediana: 1.
    assert ajustes.loc[1] == pytest.approx(1.0)


def test_estacionalidad_no_utiliza_targets_validacion():
    """Las etiquetas de validación no pueden ajustar el baseline."""

    train = pd.DataFrame(
        {
            "morosidad_pp": [4.0, 5.0, 6.0],
            "target_morosidad_t1_pp": [5.0, 6.0, 8.0],
            "fecha_objetivo_t1": pd.to_datetime(
                [
                    "2020-01-01",
                    "2021-01-01",
                    "2022-01-01",
                ]
            ),
        }
    )

    val = pd.DataFrame(
        {
            "morosidad_pp": [6.0, 100.0, 1.0],
            "target_morosidad_t1_pp": [50.0, 0.0, 99.0],
            "fecha_objetivo_t1": pd.to_datetime(
                [
                    "2023-01-01",
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
        }
    )

    pred = predecir_persistencia_estacional(
        train,
        val,
        "target_morosidad_t1_pp",
        "fecha_objetivo_t1",
    )

    # Enero: corrección aprendida igual a 1.
    assert pred[0] == pytest.approx(7.0)

    # La predicción se limita al 100 %.
    assert pred[1] == pytest.approx(100.0)

    # Febrero no tiene corrección aprendida.
    assert pred[2] == pytest.approx(1.0)

    # Modificar las etiquetas de validación
    # no debe modificar las predicciones.
    val_modificada = val.copy()

    val_modificada["target_morosidad_t1_pp"] = [0.0, 100.0, 0.0]

    pred_modificada = predecir_persistencia_estacional(
        train,
        val_modificada,
        "target_morosidad_t1_pp",
        "fecha_objetivo_t1",
    )

    np.testing.assert_array_equal(
        pred,
        pred_modificada,
    )
