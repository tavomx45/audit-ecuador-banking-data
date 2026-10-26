"""Pruebas de comparación de modelos de morosidad."""

import pandas as pd
import pytest

from src.modeling.evaluate import comparar_modelos_cv


@pytest.fixture
def resultados_sinteticos():

    baselines = pd.DataFrame(
        [
            (1, 2021, "Persistencia", 10, 1.0),
            (1, 2022, "Persistencia", 10, 2.0),
            (1, 2021, "Persistencia_estacional", 10, 0.8),
            (1, 2022, "Persistencia_estacional", 10, 1.2),
        ],
        columns=[
            "horizonte",
            "fold",
            "modelo",
            "observaciones",
            "mae_pp",
        ],
    )

    supervisados = pd.DataFrame(
        [
            (1, 2021, "Ridge", 10, 1.0),
            (1, 2022, "Ridge", 10, 1.5),
        ],
        columns=baselines.columns,
    )

    return baselines, supervisados


def test_comparacion_identifica_mejor_baseline(
    resultados_sinteticos,
):
    baselines, supervisados = resultados_sinteticos

    detalle, resumen = comparar_modelos_cv(
        baselines,
        supervisados,
    )

    assert len(detalle) == 6
    assert len(resumen) == 3

    # Persistencia estacional tiene MAE promedio 1.0.
    assert resumen.iloc[0]["modelo"] == "Persistencia_estacional"

    ridge = resumen.loc[resumen["modelo"] == "Ridge"].iloc[0]

    assert ridge["mae_medio"] == pytest.approx(1.25)
    assert ridge["mae_baseline"] == pytest.approx(1.0)

    # Ridge tiene un 25 % más de error que el baseline.
    assert ridge["diferencia_mae_pp"] == pytest.approx(0.25)
    assert ridge["mejora_pct"] == pytest.approx(-25.0)


def test_comparacion_rechaza_folds_incompletos(
    resultados_sinteticos,
):
    baselines, supervisados = resultados_sinteticos

    supervisados = supervisados.iloc[:1].copy()

    with pytest.raises(
        ValueError,
        match="mismos folds",
    ):
        comparar_modelos_cv(baselines, supervisados)


def test_comparacion_rechaza_tamaños_inconsistentes(
    resultados_sinteticos,
):
    baselines, supervisados = resultados_sinteticos

    supervisados = supervisados.copy()
    supervisados.loc[0, "observaciones"] = 11

    with pytest.raises(
        ValueError,
        match="Tamaños de validación inconsistentes",
    ):
        comparar_modelos_cv(baselines, supervisados)


def test_comparacion_rechaza_evaluaciones_duplicadas(
    resultados_sinteticos,
):
    baselines, supervisados = resultados_sinteticos

    duplicados = pd.concat(
        [supervisados, supervisados.iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicadas",
    ):
        comparar_modelos_cv(baselines, duplicados)
