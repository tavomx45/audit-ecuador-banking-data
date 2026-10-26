"""Modelos baseline de persistencia para morosidad bancaria."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def predecir_persistencia(datos: pd.DataFrame) -> np.ndarray:
    """Predice la morosidad futura igual a la morosidad actual."""

    if "morosidad_pp" not in datos.columns:
        raise ValueError("Falta la columna morosidad_pp.")

    return datos["morosidad_pp"].to_numpy(
        dtype=float,
        copy=True,
    )


def aprender_ajustes_estacionales(
    entrenamiento: pd.DataFrame,
    target: str,
    fecha_target: str,
) -> pd.Series:
    """Aprende medianas mensuales del cambio futuro de morosidad."""

    requeridas = {
        "morosidad_pp",
        target,
        fecha_target,
    }

    faltantes = requeridas - set(entrenamiento.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas de entrenamiento: {sorted(faltantes)}")

    if entrenamiento.empty:
        raise ValueError("El entrenamiento está vacío.")

    # Variación futura observada en entrenamiento.
    cambios = entrenamiento[target] - entrenamiento["morosidad_pp"]

    # Mes al que corresponde el objetivo.
    mes_objetivo = pd.to_datetime(entrenamiento[fecha_target]).dt.month

    # Mediana histórica por mes objetivo.
    ajustes = cambios.groupby(mes_objetivo).median()

    return ajustes


def predecir_persistencia_estacional(
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    target: str,
    fecha_target: str,
) -> np.ndarray:
    """Predice persistencia ajustada por estacionalidad.

    Las correcciones se aprenden exclusivamente del entrenamiento.
    """

    requeridas_val = {"morosidad_pp", fecha_target}

    faltantes = requeridas_val - set(validacion.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas de validación: {sorted(faltantes)}")

    # Ajustar exclusivamente con entrenamiento.
    ajustes = aprender_ajustes_estacionales(
        entrenamiento,
        target,
        fecha_target,
    )

    # Identificar el mes objetivo de validación.
    meses_val = pd.to_datetime(validacion[fecha_target]).dt.month

    # Meses sin ajuste aprendido: corrección cero.
    correcciones = meses_val.map(ajustes).fillna(0)

    predicciones = validacion["morosidad_pp"].to_numpy(
        dtype=float
    ) + correcciones.to_numpy(dtype=float)

    # Límites utilizados en el notebook académico.
    return np.clip(predicciones, 0, 100)


def evaluar_baselines_cv(
    datasets: dict,
    folds_cv: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evalúa los dos baselines sobre los folds temporales."""

    resultados = []

    for h, configuracion in datasets.items():
        datos = configuracion["desarrollo"]

        target = configuracion["target"]
        fecha_target = configuracion["fecha_target"]

        for fold in folds_cv[h]:
            entrenamiento = datos.loc[fold["train_idx"]]

            validacion = datos.loc[fold["val_idx"]]

            y_real = validacion[target]

            # ==================================
            # BASELINE 1: PERSISTENCIA
            # ==================================

            pred_persistencia = predecir_persistencia(validacion)

            # ==================================
            # BASELINE 2: ESTACIONALIDAD
            # ==================================

            pred_estacional = predecir_persistencia_estacional(
                entrenamiento,
                validacion,
                target,
                fecha_target,
            )

            # ==================================
            # EVALUACIÓN
            # ==================================

            predicciones = {
                "Persistencia": pred_persistencia,
                "Persistencia_estacional": pred_estacional,
            }

            for nombre, pred in predicciones.items():
                mae = mean_absolute_error(
                    y_real,
                    pred,
                )

                resultados.append(
                    {
                        "horizonte": h,
                        "fold": fold["fold"],
                        "modelo": nombre,
                        "observaciones": len(validacion),
                        "mae_pp": mae,
                    }
                )

    detalle = pd.DataFrame(resultados)

    resumen = (
        detalle.groupby(["horizonte", "modelo"])
        .agg(
            mae_medio=("mae_pp", "mean"),
            mae_std=("mae_pp", "std"),
            folds=("fold", "count"),
        )
        .reset_index()
        .sort_values(["horizonte", "mae_medio"])
        .reset_index(drop=True)
    )

    return detalle, resumen
