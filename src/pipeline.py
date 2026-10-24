"""Pipeline reproducible del panel bancario ecuatoriano."""

import argparse
import os
from pathlib import Path

import pandas as pd

from src.boletines import construir_panel
from src.data_manifest import PROJECT_ROOT, verificar_manifiesto


CORTE = "2026_09"

SERIES_DIR = PROJECT_ROOT / "data/raw/series_entidad"

REFERENCIA = (
    PROJECT_ROOT / f"data/interim/panel_bancario_{CORTE}.parquet"
)

SALIDA_DIR = PROJECT_ROOT / "data/interim/rebuild"


def validar_panel(panel: pd.DataFrame) -> None:
    """Comprueba las propiedades estructurales del panel."""

    columnas_requeridas = {
        "fecha", "banco", "morosidad_total",
        "cartera_bruta", "cartera_improductiva",
    }

    if not columnas_requeridas.issubset(panel.columns):
        raise ValueError("Faltan columnas requeridas.")

    if panel.empty:
        raise ValueError("El panel está vacío.")

    if panel[["banco", "fecha"]].isna().any().any():
        raise ValueError("Existen identificadores faltantes.")

    if panel.duplicated(["banco", "fecha"]).any():
        raise ValueError("Existen duplicados banco-mes.")

    print("Estructura del panel: VALIDADA")


def comparar_referencia(panel: pd.DataFrame) -> None:
    """Verifica equivalencia con la referencia académica."""

    if not REFERENCIA.is_file():
        raise FileNotFoundError(
            f"No se encontró la referencia: {REFERENCIA}"
        )

    original = pd.read_parquet(REFERENCIA)

    if set(original.columns) != set(panel.columns):
        raise ValueError("Las columnas no coinciden.")

    columnas = original.columns.tolist()

    def ordenar(df):
        return (
            df[columnas]
            .sort_values(["banco", "fecha"])
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(
        ordenar(original),
        ordenar(panel),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-8,
    )

    print("Comparación con referencia: VALIDADA")


def guardar_resultados(panel, reporte, features) -> None:
    """Guarda resultados sin sobrescribir la referencia."""

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    salidas = [
        (
            SALIDA_DIR / f"panel_bancario_{CORTE}.parquet",
            lambda ruta: panel.to_parquet(ruta, index=False),
        ),
        (
            SALIDA_DIR / f"reporte_calidad_{CORTE}.csv",
            lambda ruta: pd.DataFrame(reporte).to_csv(
                ruta, index=False
            ),
        ),
    ]


    salidas.append(
        (
            PROJECT_ROOT
            / "data/processed/rebuild/panel_features_multihorizonte.parquet",
            lambda ruta: features.to_parquet(ruta, index=False),
        )
    )

    for destino, escribir in salidas:
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_name(destino.name + ".tmp")

        try:
            escribir(temporal)
            os.replace(temporal, destino)
        finally:
            temporal.unlink(missing_ok=True)

        print("Guardado:", destino.relative_to(PROJECT_ROOT))


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Guardar el panel reconstruido.",
    )
    args = parser.parse_args()

    print("\n1. VERIFICACIÓN DE ARCHIVOS")
    verificar_manifiesto()

    print("\n2. CONSTRUCCIÓN DEL PANEL")

    panel, reporte = construir_panel(
        directorio=SERIES_DIR,
        patron=f"*{CORTE}.xlsx",
    )

    print("Dimensiones:", panel.shape)
    print("Instituciones:", panel["banco"].nunique())

    print("\n3. VALIDACIÓN DEL PANEL")
    validar_panel(panel)

    print("\n4. COMPARACIÓN ACADÉMICA")
    comparar_referencia(panel)


    print("\n5. INGENIERÍA DE CARACTERÍSTICAS")

    from src.features import (
        construir_targets,
        construir_rezagos_y_variaciones,
        construir_estadisticas_moviles,
        construir_caracteristicas_estacionales,
        construir_crecimiento_cartera,
        construir_transformaciones_financieras,
    )

    features = construir_targets(panel)
    features = construir_rezagos_y_variaciones(features)
    features = construir_estadisticas_moviles(features)
    features = construir_caracteristicas_estacionales(features)
    features = construir_crecimiento_cartera(features)
    features = construir_transformaciones_financieras(features)

    print("Dimensiones:", features.shape)
    print("Instituciones:", features["banco"].nunique())

    if features.shape != (5972, 46):
        raise ValueError(
            "Las dimensiones del dataset no coinciden."
        )

    # Comparación con el dataset académico
    referencia_features = (
        PROJECT_ROOT
        / "data/processed/panel_features_multihorizonte.parquet"
    )

    if not referencia_features.is_file():
        raise FileNotFoundError(
            f"No se encontró la referencia: {referencia_features}"
        )

    original_features = pd.read_parquet(referencia_features)

    if set(original_features.columns) != set(features.columns):
        raise ValueError(
            "Las columnas de características no coinciden."
        )

    columnas = original_features.columns.tolist()

    def ordenar_features(datos):
        return (
            datos[columnas]
            .sort_values(["banco", "fecha"])
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(
        ordenar_features(original_features),
        ordenar_features(features),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-8,
    )

    print("Comparación de las 46 columnas: VALIDADA")

    if args.write:
        print("\n6. ALMACENAMIENTO")
        guardar_resultados(panel, reporte, features)
    else:
        print("\nModo verificación: no se escribieron archivos.")

    print("\nPIPELINE FINALIZADO CORRECTAMENTE")


if __name__ == "__main__":
    main()
