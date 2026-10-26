"""Construcción de folds temporales con ventana expansiva."""

import pandas as pd

from src.modeling.data import TEST_INICIO

ANIOS_VALIDACION = (2021, 2022, 2023, 2024, 2025)


def construir_folds_temporales(
    datasets: dict,
    anios_validacion=ANIOS_VALIDACION,
    test_inicio=TEST_INICIO,
):
    """Construye folds temporales por horizonte predictivo.

    Entrenamiento:
        Etiquetas anteriores al inicio del fold.

    Validación:
        Pronósticos originados durante el año del fold.

    Los índices corresponden a las filas de los DataFrames
    de desarrollo preparados previamente.
    """

    folds_cv = {}
    resumen = []

    test_inicio = pd.Timestamp(test_inicio)

    for h, configuracion in datasets.items():
        datos = configuracion["desarrollo"]

        fecha_target = configuracion["fecha_target"]

        if not datos.index.is_unique:
            raise ValueError(f"Índices duplicados en desarrollo, horizonte {h}.")

        folds_cv[h] = []

        for anio in anios_validacion:
            # Límites del período de validación
            inicio = pd.Timestamp(f"{anio}-01-01")
            fin = pd.Timestamp(f"{anio + 1}-01-01")

            # ==================================
            # ENTRENAMIENTO
            # ==================================

            train = datos.loc[datos[fecha_target] < inicio]

            # ==================================
            # VALIDACIÓN
            # ==================================

            val = datos.loc[
                datos["fecha"].between(
                    inicio,
                    fin,
                    inclusive="left",
                )
            ]

            # ==================================
            # VALIDACIONES
            # ==================================

            if train.empty:
                raise ValueError(f"Entrenamiento vacío: h={h}, año={anio}.")

            if val.empty:
                raise ValueError(f"Validación vacía: h={h}, año={anio}.")

            if train[fecha_target].max() >= inicio:
                raise ValueError(f"Fuga temporal en entrenamiento: h={h}.")

            if val["fecha"].min() < inicio:
                raise ValueError(f"Origen de validación incorrecto: h={h}.")

            if val["fecha"].max() >= fin:
                raise ValueError(f"Validación fuera del año: h={h}.")

            if (
                val[fecha_target].isna().any()
                or (val[fecha_target] >= test_inicio).any()
            ):
                raise ValueError(f"Validación contiene objetivos de Test: h={h}.")

            # ==================================
            # GUARDAR ÍNDICES
            # ==================================

            folds_cv[h].append(
                {
                    "fold": anio,
                    "train_idx": train.index.to_numpy(copy=True),
                    "val_idx": val.index.to_numpy(copy=True),
                }
            )

            # ==================================
            # REGISTRAR RESUMEN
            # ==================================

            resumen.append(
                {
                    "horizonte": h,
                    "fold": anio,
                    "train": len(train),
                    "validacion": len(val),
                    "ultima_etiqueta_train": train[fecha_target].max(),
                    "primera_fecha_val": val["fecha"].min(),
                }
            )

    tabla = pd.DataFrame(resumen)

    return folds_cv, tabla
