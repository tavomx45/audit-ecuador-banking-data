"""Pruebas de modelado de variaciones de morosidad."""

import numpy as np
import pandas as pd
import pytest

from src.modeling.delta import (
    calcular_objetivo_delta,
    entrenar_predecir_delta,
    reconstruir_morosidad,
)


def test_calculo_del_objetivo_delta():
    """Comprueba el cálculo de cambios futuros."""

    datos = pd.DataFrame(
        {
            "morosidad_pp": [4.0, 3.5, 5.0],
            "target_morosidad_t3_pp": [4.2, 3.0, 5.0],
        }
    )

    original = datos.copy(deep=True)

    delta = calcular_objetivo_delta(
        datos,
        "target_morosidad_t3_pp",
    )

    np.testing.assert_allclose(
        delta.to_numpy(),
        [0.2, -0.5, 0.0],
    )

    # No modifica los datos originales.
    pd.testing.assert_frame_equal(
        datos,
        original,
    )


def test_reconstruccion_de_morosidad():
    """La morosidad futura se obtiene sumando el cambio."""

    actual = np.array([4.0, 3.5, 5.0])
    delta = np.array([0.2, -0.5, 0.0])

    resultado = reconstruir_morosidad(
        actual,
        delta,
    )

    np.testing.assert_allclose(
        resultado,
        [4.2, 3.0, 5.0],
    )

    # El recorte es opcional.
    resultado_recortado = reconstruir_morosidad(
        np.array([99.0, 1.0]),
        np.array([5.0, -3.0]),
        recortar=True,
    )

    np.testing.assert_allclose(
        resultado_recortado,
        [100.0, 0.0],
    )


def test_entrenamiento_delta_no_utiliza_etiquetas_validacion():
    """Modificar las etiquetas de validación no altera pronósticos."""

    entrenamiento = pd.DataFrame(
        {
            "banco": ["A", "A", "B", "A", "B"],
            "morosidad_pp": [2.0, 2.3, 3.0, 2.6, 3.2],
            "roa": [0.01, np.nan, 0.02, 0.02, 0.03],
            "target_morosidad_t1_pp": [2.3, 2.6, 3.2, 2.8, 3.5],
        }
    )

    validacion = pd.DataFrame(
        {
            "banco": ["A", "C"],
            "morosidad_pp": [2.8, 3.5],
            "roa": [0.02, 0.01],
            "target_morosidad_t1_pp": [3.0, 3.7],
        }
    )

    argumentos = {
        "nombre": "Ridge",
        "entrenamiento": entrenamiento,
        "target": "target_morosidad_t1_pp",
        "numericas": ["morosidad_pp", "roa"],
        "predictoras": ["morosidad_pp", "roa", "banco"],
    }

    pred_original = entrenar_predecir_delta(
        validacion=validacion,
        **argumentos,
    )

    # Cambiar las etiquetas de validación.
    validacion_modificada = validacion.copy()
    validacion_modificada["target_morosidad_t1_pp"] = [
        90.0,
        1.0,
    ]

    pred_modificada = entrenar_predecir_delta(
        validacion=validacion_modificada,
        **argumentos,
    )

    np.testing.assert_allclose(
        pred_original,
        pred_modificada,
    )

    # También debe funcionar si no existen las etiquetas de validación.
    sin_etiquetas = validacion.drop(columns=["target_morosidad_t1_pp"])

    pred_sin_etiquetas = entrenar_predecir_delta(
        validacion=sin_etiquetas,
        **argumentos,
    )

    np.testing.assert_allclose(
        pred_original,
        pred_sin_etiquetas,
    )

    assert np.isfinite(pred_original).all()


def test_delta_rechaza_objetivos_invalidos():
    """Los targets faltantes no deben utilizarse en entrenamiento."""

    datos = pd.DataFrame(
        {
            "morosidad_pp": [3.0, 4.0],
            "target_morosidad_t1_pp": [3.2, np.nan],
        }
    )

    with pytest.raises(
        ValueError,
        match="objetivo contiene valores inválidos",
    ):
        calcular_objetivo_delta(
            datos,
            "target_morosidad_t1_pp",
        )
