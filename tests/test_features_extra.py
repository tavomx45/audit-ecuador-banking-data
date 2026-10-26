"""Pruebas adicionales de ingeniería de características."""

import numpy as np
import pandas as pd
import pytest

from src.features import (
    construir_targets,
    construir_rezagos_y_variaciones,
    construir_estadisticas_moviles,
    construir_caracteristicas_estacionales,
    construir_crecimiento_cartera,
    construir_transformaciones_financieras,
)



def seleccionar_fila(datos, banco, fecha=None):
    """Selecciona exactamente una observación de una institución."""

    mascara = datos["banco"].eq(banco)

    if fecha is not None:
        mascara = mascara & datos["fecha"].eq(
            pd.Timestamp(fecha)
        )

    seleccion = datos.loc[mascara]

    assert len(seleccion) == 1, (
        f"Se esperaba una fila para banco={banco}, "
        f"fecha={fecha}, pero se encontraron {len(seleccion)}."
    )

    return seleccion.iloc[0]


def test_rezagos_y_variaciones_respetan_calendario():
    """Los rezagos representan meses, no posiciones de filas."""

    datos = pd.DataFrame({
        "banco": ["A", "A", "A", "B"],
        "fecha": pd.to_datetime([
            "2020-01-01",
            "2020-03-01",
            "2020-04-01",
            "2020-02-01",
        ]),
        "morosidad_pp": [3.0, 5.0, 8.0, 50.0],
    })

    resultado = construir_rezagos_y_variaciones(datos)

    marzo = seleccionar_fila(resultado, "A", "2020-03-01")

    abril = seleccionar_fila(resultado, "A", "2020-04-01")

    febrero_b = seleccionar_fila(resultado, "B")

    # Febrero falta para el banco A
    assert pd.isna(marzo["morosidad_lag_1_pp"])
    assert marzo["morosidad_lag_2_pp"] == pytest.approx(3.0)

    # Abril tiene marzo como lag 1 y enero como lag 3
    assert abril["morosidad_lag_1_pp"] == pytest.approx(5.0)
    assert abril["morosidad_lag_3_pp"] == pytest.approx(3.0)

    # Variaciones históricas
    assert abril["delta_morosidad_1m_pp"] == pytest.approx(3.0)
    assert abril["delta_morosidad_3m_pp"] == pytest.approx(5.0)

    # No se mezclan instituciones
    assert pd.isna(febrero_b["morosidad_lag_1_pp"])


def test_desviacion_movil_usa_ddof_cero():
    """Verifica la desviación estándar poblacional."""

    datos = pd.DataFrame({
        "banco": ["A", "A", "A"],
        "fecha": pd.date_range(
            "2020-01-01", periods=3, freq="MS"
        ),
        "morosidad_pp": [2.0, 4.0, 6.0],
    })

    resultado = construir_estadisticas_moviles(
        datos, ventanas=(3,)
    )

    marzo = resultado.iloc[-1]

    media_esperada = 4.0
    std_esperada = np.sqrt(8 / 3)

    assert marzo["morosidad_media_3m_pp"] == pytest.approx(
        media_esperada
    )

    assert marzo["morosidad_std_3m_pp"] == pytest.approx(
        std_esperada
    )


def test_estacionalidad_depende_del_mes_objetivo():
    """Cada horizonte utiliza su propio mes futuro."""

    panel = pd.DataFrame({
        "banco": ["A"],
        "fecha": pd.to_datetime(["2025-09-01"]),
        "morosidad_total": [0.04],
    })

    datos = construir_targets(panel)

    resultado = construir_caracteristicas_estacionales(datos)

    fila = resultado.iloc[0]

    for h, mes_esperado in [(1, 10), (2, 11), (3, 12)]:

        assert fila[f"mes_objetivo_t{h}"] == mes_esperado

        seno = fila[f"mes_sin_t{h}"]
        coseno = fila[f"mes_cos_t{h}"]

        assert seno**2 + coseno**2 == pytest.approx(1.0)

        angulo = 2 * np.pi * mes_esperado / 12

        assert seno == pytest.approx(np.sin(angulo))
        assert coseno == pytest.approx(np.cos(angulo))

    # La estacionalidad se conoce aunque el target no exista
    assert pd.isna(fila["target_morosidad_t1_pp"])
    assert pd.notna(fila["mes_sin_t1"])


def test_crecimiento_cartera_controla_meses_y_ceros():
    """Comprueba los crecimientos y los saldos inválidos."""

    datos = pd.DataFrame({
        "banco": ["A", "A", "A", "B", "C", "C"],
        "fecha": pd.to_datetime([
            "2020-01-01",
            "2020-03-01",
            "2020-04-01",
            "2020-02-01",
            "2020-01-01",
            "2020-02-01",
        ]),
        "cartera_bruta": [
            100.0, 121.0, 133.1, 500.0, 0.0, 100.0
        ],
    })

    resultado = construir_crecimiento_cartera(datos)

    marzo_a = seleccionar_fila(resultado, "A", "2020-03-01")

    abril_a = seleccionar_fila(resultado, "A", "2020-04-01")

    febrero_c = seleccionar_fila(resultado, "C", "2020-02-01")

    assert pd.isna(marzo_a["crecimiento_cartera_1m_pct"])

    assert abril_a["crecimiento_cartera_1m_pct"] == pytest.approx(
        10.0
    )

    assert abril_a["crecimiento_cartera_3m_pct"] == pytest.approx(
        33.1
    )

    # Cartera anterior igual a cero
    assert pd.isna(febrero_c["crecimiento_cartera_1m_pct"])


def test_transformaciones_financieras_controlan_invalidos():
    """Valida logaritmos, ceros, negativos e infinitos."""

    datos = pd.DataFrame({
        "cartera_bruta": [110.0, 0.0, 100.0],
        "cartera_bruta_lag_1": [100.0, 100.0, 0.0],
        "cartera_bruta_lag_3": [100.0, 100.0, 50.0],
        "liquidez": [0.0, -1.0, 10.0],
        "cartera_depositos": [1.0, 0.0, np.inf],
    })

    columnas_originales = datos.columns.tolist()

    resultado = construir_transformaciones_financieras(datos)

    # Crecimiento válido
    assert resultado.loc[
        0, "crecimiento_log_cartera_1m"
    ] == pytest.approx(np.log(1.1))

    # Cartera actual o anterior igual a cero
    assert pd.isna(
        resultado.loc[1, "crecimiento_log_cartera_1m"]
    )

    assert pd.isna(
        resultado.loc[2, "crecimiento_log_cartera_1m"]
    )

    # Ceros válidos y liquidez negativa
    assert resultado.loc[0, "liquidez_log"] == 0.0
    assert pd.isna(resultado.loc[1, "liquidez_log"])

    # Valores infinitos
    assert pd.isna(
        resultado.loc[2, "cartera_depositos_log"]
    )

    # La función no modifica las columnas del DataFrame recibido
    assert datos.columns.tolist() == columnas_originales


def test_targets_rechazan_duplicados():
    """Una combinación banco-mes debe ser única."""

    panel = pd.DataFrame({
        "banco": ["A", "A"],
        "fecha": pd.to_datetime([
            "2020-01-01",
            "2020-01-01",
        ]),
        "morosidad_total": [0.03, 0.04],
    })

    with pytest.raises(ValueError, match="duplicados"):
        construir_targets(panel)

