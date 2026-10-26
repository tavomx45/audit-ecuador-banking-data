"""Pruebas de validación cruzada temporal."""

import pandas as pd
import pytest

from src.modeling.splits import construir_folds_temporales


def test_entrenamiento_usa_fecha_objetivo():
    """No basta con que el origen sea anterior al fold."""

    fechas = pd.to_datetime(
        [
            "2020-09-01",
            "2020-10-01",
            "2020-11-01",
            "2020-12-01",
            "2021-01-01",
            "2021-02-01",
        ]
    )

    datos = pd.DataFrame(
        {
            "banco": ["BANCO A"] * len(fechas),
            "fecha": fechas,
            "fecha_objetivo_t2": fechas + pd.DateOffset(months=2),
        }
    )

    datasets = {
        2: {
            "desarrollo": datos,
            "fecha_target": "fecha_objetivo_t2",
        }
    }

    folds, resumen = construir_folds_temporales(
        datasets,
        anios_validacion=(2021,),
    )

    fold = folds[2][0]

    # Septiembre y octubre tienen etiquetas de 2020.
    assert fold["train_idx"].tolist() == [0, 1]

    # Noviembre y diciembre no pueden entrar en train:
    # sus etiquetas corresponden a enero y febrero de 2021.
    assert 2 not in fold["train_idx"]
    assert 3 not in fold["train_idx"]

    # La validación utiliza orígenes de 2021.
    assert fold["val_idx"].tolist() == [4, 5]

    assert resumen.loc[0, "train"] == 2
    assert resumen.loc[0, "validacion"] == 2


def test_validacion_rechaza_objetivos_de_test():
    """Ninguna etiqueta de Test debe entrar en validación."""

    datos = pd.DataFrame(
        {
            "banco": ["BANCO A", "BANCO A"],
            "fecha": pd.to_datetime(
                [
                    "2024-01-01",
                    "2025-08-01",
                ]
            ),
            "fecha_objetivo_t2": pd.to_datetime(
                [
                    "2024-03-01",
                    "2025-10-01",
                ]
            ),
        }
    )

    datasets = {
        2: {
            "desarrollo": datos,
            "fecha_target": "fecha_objetivo_t2",
        }
    }

    with pytest.raises(
        ValueError,
        match="objetivos de Test",
    ):
        construir_folds_temporales(
            datasets,
            anios_validacion=(2025,),
        )
