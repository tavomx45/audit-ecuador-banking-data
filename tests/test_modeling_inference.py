"""Pruebas de inferencia multihorizonte de morosidad bancaria."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.modeling import inference

# ======================================================
# 1. CONFIGURACIÓN PARA LAS PRUEBAS
# ======================================================

NUMERICAS = [
    "morosidad_pp",
    "morosidad_lag_1_pp",
    "morosidad_lag_2_pp",
    "morosidad_lag_3_pp",
    "morosidad_lag_6_pp",
    "morosidad_lag_12_pp",
    "delta_morosidad_1m_pp",
    "delta_morosidad_3m_pp",
    "morosidad_media_3m_pp",
    "morosidad_media_6m_pp",
    "morosidad_std_3m_pp",
    "morosidad_std_6m_pp",
    "solvencia",
    "roa",
    "roe",
    "liquidez_log",
    "cartera_depositos_log",
    "crecimiento_log_cartera_1m",
    "crecimiento_log_cartera_3m",
    "mes_sin_t1",
    "mes_cos_t1",
]

PREDICTORAS = NUMERICAS + ["banco"]

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "modeling"

assert len(PREDICTORAS) == 22


class RandomForestSimulado:
    """Modelo simulado para probar la interfaz sin MLflow Server."""

    feature_names_in_ = np.asarray(PREDICTORAS)

    def predict(self, X):
        assert list(X.columns) == PREDICTORAS
        return np.full(len(X), 0.25)


@pytest.fixture
def entrada_rf():
    """Observación de prueba con las 22 características."""

    valores = {columna: 1.0 for columna in NUMERICAS}

    valores.update(
        {
            "banco": "BP PRUEBA",
            "fecha": "2025-09-30",
            "morosidad_pp": 3.0,
        }
    )

    return pd.DataFrame([valores])


def leer_artefacto(horizonte):
    """Carga los ajustes estacionales versionados."""

    ruta = CONFIG_DIR / f"seasonal_h{horizonte}.json"

    return json.loads(ruta.read_text(encoding="utf-8"))


# ======================================================
# 2. PRUEBA DE RANDOM FOREST
# ======================================================


def test_inferencia_rf_h1(monkeypatch, entrada_rf):
    """Verifica reconstrucción y horizonte de un mes."""

    uris_cargadas = []

    def cargar_modelo(uri):
        uris_cargadas.append(uri)
        return RandomForestSimulado()

    monkeypatch.setattr(
        inference.mlflow.sklearn,
        "load_model",
        cargar_modelo,
    )

    monkeypatch.setattr(
        inference.mlflow,
        "set_tracking_uri",
        lambda uri: None,
    )

    resultado = inference.pronosticar_morosidad(
        entrada_rf,
        horizonte=1,
    )

    assert len(resultado) == 1
    assert uris_cargadas == [inference.MODELO_RF_H1]

    assert resultado["horizonte"].iloc[0] == 1

    assert resultado["fecha_objetivo"].iloc[0] == pd.Timestamp("2025-10-31")

    np.testing.assert_allclose(
        resultado["delta_predicha_pp"],
        [0.25],
    )

    np.testing.assert_allclose(
        resultado["morosidad_predicha_pp"],
        [3.25],
    )


# ======================================================
# 3. PRUEBAS DE PERSISTENCIA ESTACIONAL
# ======================================================


@pytest.mark.parametrize("horizonte", [2, 3])
def test_inferencia_estacional_sin_mlflow(
    monkeypatch,
    horizonte,
):
    """La persistencia estacional no debe cargar MLflow."""

    def prohibir_carga(*args, **kwargs):
        pytest.fail("La persistencia estacional no debe cargar MLflow.")

    monkeypatch.setattr(
        inference.mlflow.sklearn,
        "load_model",
        prohibir_carga,
    )

    artefacto = leer_artefacto(horizonte)

    corte = pd.Period(
        artefacto["fecha_objetivo_max_train"],
        freq="M",
    )

    periodo_origen = corte + 1

    entrada = pd.DataFrame(
        {
            "banco": ["BP PRUEBA"],
            "fecha": [periodo_origen.to_timestamp(how="end").normalize()],
            "morosidad_pp": [3.0],
        }
    )

    resultado = inference.pronosticar_morosidad(
        entrada,
        horizonte=horizonte,
    )

    periodo_objetivo = periodo_origen + horizonte

    ajuste = artefacto["ajustes"].get(
        str(periodo_objetivo.month),
        0.0,
    )

    esperado = np.clip(
        3.0 + float(ajuste),
        0,
        100,
    )

    assert len(resultado) == 1

    assert resultado["modelo"].iloc[0] == ("Persistencia_estacional")

    assert resultado["fecha_objetivo"].iloc[0].to_period("M") == periodo_objetivo

    np.testing.assert_allclose(
        resultado["morosidad_predicha_pp"],
        [esperado],
        atol=1e-12,
    )


# ======================================================
# 4. VALIDACIONES DE SEGURIDAD
# ======================================================


@pytest.mark.parametrize("horizonte", [1, 2, 3])
def test_rechaza_etiquetas_futuras(
    entrada_rf,
    horizonte,
):
    """Una etiqueta futura nunca debe entrar al predictor."""

    entrada = entrada_rf.copy()

    entrada["target_morosidad_t1_pp"] = 5.0

    with pytest.raises(
        ValueError,
        match="etiquetas futuras",
    ):
        inference.pronosticar_morosidad(
            entrada,
            horizonte=horizonte,
        )


def test_rechaza_horizonte_no_soportado(entrada_rf):
    """Solamente se permiten horizontes de 1 a 3 meses."""

    with pytest.raises(
        ValueError,
        match="Horizonte no soportado",
    ):
        inference.pronosticar_morosidad(
            entrada_rf,
            horizonte=4,
        )


def test_rechaza_predictores_faltantes(
    monkeypatch,
    entrada_rf,
):
    """No se permite omitir características del modelo."""

    monkeypatch.setattr(
        inference.mlflow.sklearn,
        "load_model",
        lambda uri: RandomForestSimulado(),
    )

    entrada = entrada_rf.drop(columns=["roa"])

    with pytest.raises(
        ValueError,
        match="Faltan predictores",
    ):
        inference.pronosticar_morosidad(
            entrada,
            horizonte=1,
        )


def test_rechaza_duplicados(monkeypatch, entrada_rf):
    """Un mismo banco y mes no puede repetirse."""

    monkeypatch.setattr(
        inference.mlflow.sklearn,
        "load_model",
        lambda uri: RandomForestSimulado(),
    )

    entrada = pd.concat(
        [entrada_rf, entrada_rf],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="duplicadas",
    ):
        inference.pronosticar_morosidad(
            entrada,
            horizonte=1,
        )


@pytest.mark.parametrize("horizonte", [2, 3])
def test_estacional_rechaza_origen_antes_del_corte(
    horizonte,
):
    """No permite pronósticos históricos previos al corte."""

    artefacto = leer_artefacto(horizonte)

    entrada = pd.DataFrame(
        {
            "banco": ["BP PRUEBA"],
            "fecha": [
                pd.Period(
                    artefacto["fecha_objetivo_max_train"],
                    freq="M",
                )
                .to_timestamp(how="end")
                .normalize()
            ],
            "morosidad_pp": [3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="no es posterior",
    ):
        inference.pronosticar_morosidad(
            entrada,
            horizonte=horizonte,
        )
