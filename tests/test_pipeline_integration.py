"""Pruebas de integración del pipeline financiero."""

import pandas as pd
import pytest

from src import pipeline
from src.features import construir_dataset_multihorizonte

# ==================================================
# 1. FIXTURE: PANEL BANCARIO ARTIFICIAL
# ==================================================


@pytest.fixture
def panel_sintetico():
    """Genera un panel de dos bancos con un mes ausente."""

    registros = []

    calendarios = {
        "BANCO A": [1, 3, 4, 5, 6, 7],
        "BANCO B": [1, 2, 3, 4, 5, 6, 7],
    }

    for banco, meses in calendarios.items():
        for posicion, mes in enumerate(meses, start=1):
            if banco == "BANCO A":
                morosidad = 0.02 + 0.001 * posicion
                cartera = 1000.0 + 100.0 * posicion
            else:
                morosidad = 0.05 + 0.002 * posicion
                cartera = 5000.0 + 150.0 * posicion

            registros.append(
                {
                    "banco": banco,
                    "fecha": pd.Timestamp(2025, mes, 1),
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

    return pd.DataFrame(registros)


# ==================================================
# 2. FIXTURE: INGENIERÍA DE CARACTERÍSTICAS
# ==================================================


@pytest.fixture
def datos_integrados(panel_sintetico):
    """Ejecuta las seis transformaciones consecutivamente."""

    datos = construir_dataset_multihorizonte(panel_sintetico)

    return panel_sintetico, datos


# ==================================================
# 3. PRUEBA DE INTEGRACIÓN DE CARACTERÍSTICAS
# ==================================================


def test_integracion_conserva_observaciones(datos_integrados):
    """El procesamiento conserva bancos, meses y valores."""

    panel, features = datos_integrados

    assert panel.shape == (13, 11)
    assert features.shape == (13, 46)

    assert features["banco"].nunique() == 2

    assert not features.duplicated(["banco", "fecha"]).any()

    # Verificar conservación de las claves banco-mes
    claves = ["banco", "fecha"]

    original = panel[claves].sort_values(claves).reset_index(drop=True)

    reconstruido = features[claves].sort_values(claves).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original,
        reconstruido,
    )

    # Enero del Banco A no tiene febrero.
    enero_a = features.loc[
        (features["banco"] == "BANCO A")
        & (features["fecha"] == pd.Timestamp("2025-01-01"))
    ].iloc[0]

    assert pd.isna(enero_a["target_morosidad_t1_pp"])

    # El target t+2 de enero corresponde a marzo.
    marzo_a = panel.loc[
        (panel["banco"] == "BANCO A") & (panel["fecha"] == pd.Timestamp("2025-03-01"))
    ].iloc[0]

    assert enero_a["target_morosidad_t2_pp"] == pytest.approx(
        marzo_a["morosidad_total"] * 100
    )

    # Marzo del Banco A no tiene rezago de un mes.
    marzo_features = features.loc[
        (features["banco"] == "BANCO A")
        & (features["fecha"] == pd.Timestamp("2025-03-01"))
    ].iloc[0]

    assert pd.isna(marzo_features["morosidad_lag_1_pp"])

    assert pd.isna(marzo_features["morosidad_media_3m_pp"])


# ==================================================
# 4. PRUEBA DE INTEGRACIÓN DEL ALMACENAMIENTO
# ==================================================


def test_almacenamiento_en_directorio_temporal(
    datos_integrados,
    tmp_path,
):
    """Guardar los resultados sin tocar archivos reales."""

    panel, features = datos_integrados

    reporte = pd.DataFrame(
        {
            "banco": ["BANCO A", "BANCO B"],
            "observaciones": [6, 7],
        }
    )

    # La raíz temporal se inyecta al guardar.

    pipeline.guardar_resultados(
        panel,
        reporte,
        features,
        output_root=tmp_path,
    )

    ruta_panel = tmp_path / "data/interim/rebuild/panel_bancario_2026_09.parquet"

    ruta_reporte = tmp_path / "data/interim/rebuild/reporte_calidad_2026_09.csv"

    ruta_features = (
        tmp_path / "data/processed/rebuild/panel_features_multihorizonte.parquet"
    )

    # Comprobar que los tres archivos fueron creados.
    assert ruta_panel.is_file()
    assert ruta_reporte.is_file()
    assert ruta_features.is_file()

    # Recuperar los archivos.
    panel_guardado = pd.read_parquet(ruta_panel)

    features_guardadas = pd.read_parquet(ruta_features)

    reporte_guardado = pd.read_csv(ruta_reporte)

    # Comprobar dimensiones.
    assert panel_guardado.shape == (13, 11)

    assert features_guardadas.shape == (13, 46)

    assert reporte_guardado.shape == (2, 2)

    # Verificar equivalencia del contenido.
    pd.testing.assert_frame_equal(
        panel.reset_index(drop=True),
        panel_guardado.reset_index(drop=True),
        check_dtype=False,
    )

    pd.testing.assert_frame_equal(
        features.reset_index(drop=True),
        features_guardadas.reset_index(drop=True),
        check_dtype=False,
    )

    assert reporte_guardado["observaciones"].sum() == 13


def test_orquestador_es_determinista_y_no_modifica_entrada(
    panel_sintetico,
):
    """El resultado no depende del orden de entrada ni modifica el panel."""

    panel_original = panel_sintetico.copy(deep=True)

    # Primera ejecución
    resultado_a = construir_dataset_multihorizonte(panel_sintetico)

    # Segunda ejecución con las filas desordenadas
    panel_desordenado = panel_sintetico.sample(
        frac=1,
        random_state=42,
    )

    resultado_b = construir_dataset_multihorizonte(panel_desordenado)

    # La función no debe modificar el DataFrame recibido
    pd.testing.assert_frame_equal(
        panel_sintetico,
        panel_original,
    )

    # Las dimensiones deben conservarse
    assert resultado_a.shape == (13, 46)
    assert resultado_b.shape == (13, 46)

    # Ambos resultados deben ser equivalentes
    claves = ["banco", "fecha"]

    def ordenar(datos):
        return datos.sort_values(claves).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        ordenar(resultado_a),
        ordenar(resultado_b),
    )
