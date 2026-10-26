"""Pipeline reproducible del panel bancario ecuatoriano."""

import argparse
import os
from pathlib import Path

import pandas as pd

from src.boletines import construir_panel
from src.data_manifest import PROJECT_ROOT, verificar_manifiesto

CORTE = "2026_09"

SERIES_DIR = PROJECT_ROOT / "data/raw/series_entidad"

REFERENCIA = PROJECT_ROOT / f"data/interim/panel_bancario_{CORTE}.parquet"

SALIDA_DIR = PROJECT_ROOT / "data/interim/rebuild"


def validar_panel(panel: pd.DataFrame) -> None:
    """Comprueba las propiedades estructurales del panel."""

    columnas_requeridas = {
        "fecha",
        "banco",
        "morosidad_total",
        "cartera_bruta",
        "cartera_improductiva",
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


def comparar_referencia(panel: pd.DataFrame, ruta: Path = REFERENCIA) -> None:
    """Verifica equivalencia con la referencia académica."""

    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró la referencia: {ruta}")

    original = pd.read_parquet(ruta)

    if set(original.columns) != set(panel.columns):
        raise ValueError("Las columnas no coinciden.")

    columnas = original.columns.tolist()

    def ordenar(df):
        return df[columnas].sort_values(["banco", "fecha"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        ordenar(original),
        ordenar(panel),
        check_dtype=False,
        check_exact=False,
        rtol=1e-10,
        atol=1e-8,
    )

    print("Comparación con referencia: VALIDADA")


def guardar_resultados(
    panel,
    reporte,
    features,
    *,
    output_root: Path | None = None,
) -> None:
    """Guarda resultados sin sobrescribir la referencia."""

    SALIDA_DIR.mkdir(parents=True, exist_ok=True)

    salidas = [
        (
            SALIDA_DIR / f"panel_bancario_{CORTE}.parquet",
            lambda ruta: panel.to_parquet(ruta, index=False),
        ),
        (
            SALIDA_DIR / f"reporte_calidad_{CORTE}.csv",
            lambda ruta: pd.DataFrame(reporte).to_csv(ruta, index=False),
        ),
    ]

    salidas.append(
        (
            PROJECT_ROOT
            / "data/processed/rebuild/panel_features_multihorizonte.parquet",
            lambda ruta: features.to_parquet(ruta, index=False),
        )
    )

    # Directorio real por defecto; temporal durante las pruebas.
    raiz_salida = PROJECT_ROOT if output_root is None else Path(output_root).resolve()

    for destino, escribir in salidas:
        # Las rutas existentes pertenecen al proyecto.
        # Recuperamos su parte relativa y cambiamos la raíz.
        ruta_relativa = Path(destino).relative_to(PROJECT_ROOT)

        destino = raiz_salida / ruta_relativa
        destino.parent.mkdir(parents=True, exist_ok=True)
        temporal = destino.with_name(destino.name + ".tmp")

        try:
            escribir(temporal)
            os.replace(temporal, destino)
        finally:
            temporal.unlink(missing_ok=True)

        print("Guardado:", destino.relative_to(raiz_salida))


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Guardar el panel reconstruido.",
    )
    parser.add_argument(
        "--verify-reference",
        action="store_true",
        help="Comparar resultados con los Parquet académicos.",
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

    print("\n4. INGENIERÍA DE CARACTERÍSTICAS")

    from src.features import construir_dataset_multihorizonte

    features = construir_dataset_multihorizonte(panel)

    print("Dimensiones:", features.shape)
    print("Instituciones:", features["banco"].nunique())

    # Validaciones independientes de los datos académicos
    if len(features) != len(panel):
        raise ValueError("Se modificó la cantidad de observaciones.")

    if features.shape[1] != 46:
        raise ValueError("El esquema de características no coincide.")

    if features.duplicated(["banco", "fecha"]).any():
        raise ValueError("Existen duplicados banco-mes.")

    # Comprobar que se conservaron los identificadores
    def ordenar_claves(df):
        return (
            df[["banco", "fecha"]]
            .sort_values(["banco", "fecha"])
            .reset_index(drop=True)
        )

    pd.testing.assert_frame_equal(
        ordenar_claves(panel),
        ordenar_claves(features),
    )

    # Comprobar las fechas objetivo de cada horizonte
    for h in (1, 2, 3):
        fecha_esperada = features["fecha"] + pd.DateOffset(months=h)

        if not features[f"fecha_objetivo_t{h}"].equals(fecha_esperada):
            raise ValueError(f"Fechas objetivo incorrectas: horizonte {h}.")

    print("Validación de características: CORRECTA")

    print("\n5. VERIFICACIÓN ACADÉMICA")

    if args.verify_reference:
        comparar_referencia(panel)

        referencia_features = (
            PROJECT_ROOT / "data/processed/panel_features_multihorizonte.parquet"
        )

        comparar_referencia(features, referencia_features)

        print("Ambas referencias académicas: VALIDADAS")
    else:
        print("Verificación académica omitida.")

    if args.write:
        print("\n6. ALMACENAMIENTO")
        guardar_resultados(panel, reporte, features)
    else:
        print("\nModo verificación: no se escribieron archivos.")

    print("\nPIPELINE FINALIZADO CORRECTAMENTE")


if __name__ == "__main__":
    main()
