"""Pruebas de preparación de datos para Machine Learning."""

import numpy as np
import pandas as pd
import pytest

from src.features import construir_dataset_multihorizonte
from src.modeling.data import TEST_INICIO, preparar_datasets


@pytest.fixture
def features_sinteticas():
    """Construye datos de prueba sin depender de archivos reales."""

    registros = []

    fechas = pd.date_range(
        "2025-01-01",
        periods=18,
        freq="MS",
    )

    for banco in ("BANCO A", "BANCO B"):
        for i, fecha in enumerate(fechas):
            cartera = 1000.0 + i * 50

            morosidad = 0.03 + i * 0.0001

            registros.append(
                {
                    "banco": banco,
                    "fecha": fecha,
                    "solvencia": 0.12,
                    "liquidez": 0.25,
                    "roa": 0.01,
                    "roe": 0.08,
                    "cartera_depositos": 0.85,
                    "cartera_bruta": cartera,
                    "cartera_improductiva": cartera * morosidad,
                    "morosidad_total": morosidad,
                    "cobertura": 2.5,
                }
            )

    panel = pd.DataFrame(registros)

    return construir_dataset_multihorizonte(panel)


def test_preparacion_respeta_separacion_temporal(
    features_sinteticas,
):
    """Desarrollo nunca utiliza etiquetas del período Test."""

    datasets, resumen = preparar_datasets(features_sinteticas)

    assert set(datasets) == {1, 2, 3}
    assert len(resumen) == 3

    assert (resumen["variables_numericas"] == 21).all()
    assert (resumen["variables_categoricas"] == 1).all()

    for configuracion in datasets.values():
        desarrollo = configuracion["desarrollo"]
        test = configuracion["test"]
        fecha_target = configuracion["fecha_target"]

        assert not desarrollo.empty
        assert not test.empty

        assert desarrollo[fecha_target].max() < TEST_INICIO
        assert test[fecha_target].min() >= TEST_INICIO

        assert len(configuracion["predictoras"]) == 22

        assert not any(
            columna.startswith("target_") for columna in configuracion["predictoras"]
        )


def test_preparacion_no_modifica_entrada(
    features_sinteticas,
):
    """Los valores infinitos se limpian sin modificar el original."""

    datos = features_sinteticas.copy()

    datos.loc[0, "liquidez_log"] = np.inf

    original = datos.copy(deep=True)

    datasets, _ = preparar_datasets(datos)

    # El DataFrame recibido permanece intacto.
    pd.testing.assert_frame_equal(datos, original)

    # En desarrollo, el valor infinito se transforma en NaN.
    desarrollo = datasets[1]["desarrollo"]

    fila = desarrollo.loc[
        (desarrollo["banco"] == "BANCO A")
        & (desarrollo["fecha"] == pd.Timestamp("2025-01-01"))
    ].iloc[0]

    assert pd.isna(fila["liquidez_log"])
