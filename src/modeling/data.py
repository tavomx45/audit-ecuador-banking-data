"""Preparación de datasets para predicción de morosidad bancaria."""

import numpy as np
import pandas as pd

HORIZONTES = (1, 2, 3)

TEST_INICIO = pd.Timestamp("2025-10-01")
TEST_FIN = pd.Timestamp("2026-09-01")
CORTE_ORIGEN = pd.Timestamp("2025-09-01")


# ==================================================
# VARIABLES DEL EXPERIMENTO ACADÉMICO
# ==================================================

VARIABLES_BASE = [
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
]

VARIABLES_CATEGORICAS = ["banco"]


# ==================================================
# PREPARACIÓN MULTIHORIZONTE
# ==================================================


def preparar_datasets(
    features: pd.DataFrame,
    horizontes=HORIZONTES,
    test_inicio=TEST_INICIO,
    test_fin=TEST_FIN,
    corte_origen=CORTE_ORIGEN,
):
    """Prepara los conjuntos de desarrollo y Test por horizonte.

    Mantiene los criterios temporales del experimento académico.
    No imputa valores ni ajusta transformadores estadísticos.
    """

    df = features.copy()

    # Validar columnas identificadoras.
    requeridas = {"banco", "fecha", "morosidad_pp"}

    if not requeridas.issubset(df.columns):
        raise ValueError("Faltan columnas identificadoras.")

    df["fecha"] = pd.to_datetime(df["fecha"])

    if df.duplicated(["banco", "fecha"]).any():
        raise ValueError("Existen registros banco-mes duplicados.")

    # Normalizar límites temporales.
    test_inicio = pd.Timestamp(test_inicio)
    test_fin = pd.Timestamp(test_fin)
    corte_origen = pd.Timestamp(corte_origen)

    if test_inicio > test_fin:
        raise ValueError("El inicio de Test supera su fecha final.")

    datasets = {}
    resumen = []

    for h in horizontes:
        if not isinstance(h, int) or h < 1:
            raise ValueError("Horizonte inválido.")

        # --------------------------------------
        # 1. DEFINIR TARGET Y FECHA OBJETIVO
        # --------------------------------------

        target = f"target_morosidad_t{h}_pp"
        fecha_target = f"fecha_objetivo_t{h}"

        # --------------------------------------
        # 2. DEFINIR PREDICTORES
        # --------------------------------------

        estacionales = [
            f"mes_sin_t{h}",
            f"mes_cos_t{h}",
        ]

        numericas = VARIABLES_BASE + estacionales

        predictoras = numericas + VARIABLES_CATEGORICAS

        columnas_necesarias = set(predictoras) | {target, fecha_target}

        if not columnas_necesarias.issubset(df.columns):
            faltantes = sorted(columnas_necesarias - set(df.columns))

            raise ValueError(f"Faltan columnas para horizonte {h}: {faltantes}")

        # Evitar incluir targets entre los predictores.
        if any(columna.startswith("target_") for columna in predictoras):
            raise ValueError("Se detectaron targets entre los predictores.")

        # --------------------------------------
        # 3. CONSERVAR OBSERVACIONES VÁLIDAS
        # --------------------------------------

        datos = df.loc[df[target].notna() & df["morosidad_pp"].notna()].copy()

        datos[fecha_target] = pd.to_datetime(datos[fecha_target])

        # Los infinitos no son predictores válidos.
        # La imputación se ajustará más adelante,
        # exclusivamente sobre el entrenamiento.

        datos[numericas] = datos[numericas].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        # --------------------------------------
        # 4. SEPARACIÓN TEMPORAL
        # --------------------------------------

        desarrollo = datos.loc[datos[fecha_target] < test_inicio].copy()

        test = datos.loc[
            datos[fecha_target].between(
                test_inicio,
                test_fin,
            )
            & (datos["fecha"] >= corte_origen)
        ].copy()

        # --------------------------------------
        # 5. ORDENAR OBSERVACIONES
        # --------------------------------------

        desarrollo = desarrollo.sort_values(["fecha", "banco"]).reset_index(drop=True)

        test = test.sort_values(["fecha", "banco"]).reset_index(drop=True)

        # --------------------------------------
        # 6. VALIDACIONES TEMPORALES
        # --------------------------------------

        if not desarrollo.empty and desarrollo[fecha_target].max() >= test_inicio:
            raise ValueError(f"Fuga temporal en desarrollo, horizonte {h}.")

        if not test.empty:
            if test[fecha_target].min() < test_inicio:
                raise ValueError(f"Fecha objetivo incorrecta, horizonte {h}.")

            if test[fecha_target].max() > test_fin:
                raise ValueError(f"Test fuera de período, horizonte {h}.")

            if test["fecha"].min() < corte_origen:
                raise ValueError(f"Origen de pronóstico inválido, horizonte {h}.")

        # --------------------------------------
        # 7. ALMACENAR CONFIGURACIÓN
        # --------------------------------------

        datasets[h] = {
            "desarrollo": desarrollo,
            "test": test,
            "target": target,
            "fecha_target": fecha_target,
            "numericas": numericas,
            "predictoras": predictoras,
        }

        resumen.append(
            {
                "horizonte": h,
                "desarrollo": len(desarrollo),
                "test": len(test),
                "variables_numericas": len(numericas),
                "variables_categoricas": len(VARIABLES_CATEGORICAS),
            }
        )

    return datasets, pd.DataFrame(resumen)
