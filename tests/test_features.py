"""Pruebas unitarias de ingeniería de características."""

import pandas as pd
import pytest

from src.features import (
    construir_targets,
    construir_estadisticas_moviles,
)


def test_targets_respetan_meses_calendario():
    """Un target futuro debe corresponder al mes exacto."""

    panel = pd.DataFrame({
        "banco": ["BANCO A", "BANCO A", "BANCO B"],
        "fecha": pd.to_datetime([
            "2020-01-01",
            "2020-03-01",
            "2020-02-01",
        ]),
        "morosidad_total": [0.03, 0.05, 0.08],
    })

    resultado = construir_targets(panel)

    enero_a = resultado.loc[
        (resultado["banco"] == "BANCO A")
        & (resultado["fecha"] == "2020-01-01")
    ].iloc[0]

    # Febrero no existe para el Banco A.
    assert pd.isna(
        enero_a["target_morosidad_t1_pp"]
    )

    # Marzo sí existe: morosidad = 5 %.
    assert enero_a["target_morosidad_t2_pp"] == pytest.approx(5.0)

    # No se deben utilizar observaciones de otro banco.
    febrero_b = resultado.loc[
        resultado["banco"] == "BANCO B"
    ].iloc[0]

    assert pd.isna(
        febrero_b["target_morosidad_t1_pp"]
    )


def test_ventanas_moviles_no_omiten_meses():
    """Las ventanas deben contener meses consecutivos."""

    datos = pd.DataFrame({
        "banco": [
            "BANCO A", "BANCO A",
            "BANCO A", "BANCO A",
            "BANCO B",
        ],
        "fecha": pd.to_datetime([
            "2020-01-01",
            "2020-03-01",
            "2020-04-01",
            "2020-05-01",
            "2020-03-01",
        ]),
        "morosidad_pp": [
            3.0, 5.0, 7.0, 9.0, 100.0
        ],
    })

    resultado = construir_estadisticas_moviles(datos)

    marzo_a = resultado.loc[
        (resultado["banco"] == "BANCO A")
        & (resultado["fecha"] == "2020-03-01")
    ].iloc[0]

    mayo_a = resultado.loc[
        (resultado["banco"] == "BANCO A")
        & (resultado["fecha"] == "2020-05-01")
    ].iloc[0]

    # No tenemos febrero: ventana incompleta.
    assert pd.isna(
        marzo_a["morosidad_media_3m_pp"]
    )

    # Marzo, abril y mayo están completos.
    assert mayo_a["morosidad_media_3m_pp"] == pytest.approx(7.0)

    # El dato del Banco B no debe afectar al Banco A.
    assert mayo_a["morosidad_media_3m_pp"] != 100.0

