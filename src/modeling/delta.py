"""Predicción de morosidad mediante variaciones futuras."""

import numpy as np
import pandas as pd

from src.modeling.models import construir_modelo


def calcular_objetivo_delta(
    datos: pd.DataFrame,
    target: str,
) -> pd.Series:
    """Calcula el cambio futuro de morosidad en puntos porcentuales.

    delta = morosidad_futura - morosidad_actual
    """

    requeridas = {"morosidad_pp", target}
    faltantes = requeridas - set(datos.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas para calcular delta: {sorted(faltantes)}")

    if datos.empty:
        raise ValueError("No existen observaciones.")

    actual = datos["morosidad_pp"].to_numpy(dtype=float)
    futuro = datos[target].to_numpy(dtype=float)

    if not np.isfinite(actual).all():
        raise ValueError("La morosidad actual contiene valores inválidos.")

    if not np.isfinite(futuro).all():
        raise ValueError("El objetivo contiene valores inválidos.")

    delta = futuro - actual

    return pd.Series(
        delta,
        index=datos.index,
        name=f"delta_{target}",
    )


def reconstruir_morosidad(
    morosidad_actual,
    delta_predicha,
    *,
    recortar: bool = False,
) -> np.ndarray:
    """Reconstruye niveles futuros a partir de cambios predichos."""

    actual = np.asarray(morosidad_actual, dtype=float)
    delta = np.asarray(delta_predicha, dtype=float)

    if actual.ndim != 1 or delta.ndim != 1:
        raise ValueError("Las entradas deben ser vectores unidimensionales.")

    if len(actual) != len(delta):
        raise ValueError("Las entradas deben tener la misma longitud.")

    if not np.isfinite(actual).all():
        raise ValueError("Existen valores inválidos de morosidad actual.")

    if not np.isfinite(delta).all():
        raise ValueError("Existen variaciones predichas inválidas.")

    prediccion = actual + delta

    if recortar:
        prediccion = np.clip(prediccion, 0, 100)

    return prediccion


def entrenar_predecir_delta(
    nombre: str,
    entrenamiento: pd.DataFrame,
    validacion: pd.DataFrame,
    target: str,
    numericas: list[str],
    predictoras: list[str],
    *,
    recortar: bool = False,
) -> np.ndarray:
    """Entrena sobre cambios y devuelve niveles de morosidad.

    Las etiquetas de validación no intervienen en fit() ni predict().
    """

    if entrenamiento.empty or validacion.empty:
        raise ValueError("Entrenamiento y validación deben contener datos.")

    if target in predictoras:
        raise ValueError("El target no puede utilizarse como predictor.")

    if any(columna.startswith("target_") for columna in predictoras):
        raise ValueError("Se detectaron targets entre los predictores.")

    requeridas_train = set(predictoras) | {
        "morosidad_pp",
        target,
    }

    requeridas_val = set(predictoras) | {
        "morosidad_pp",
    }

    faltantes_train = requeridas_train - set(entrenamiento.columns)
    faltantes_val = requeridas_val - set(validacion.columns)

    if faltantes_train:
        raise ValueError(f"Faltan columnas de entrenamiento: {sorted(faltantes_train)}")

    if faltantes_val:
        raise ValueError(f"Faltan columnas de validación: {sorted(faltantes_val)}")

    # ==================================================
    # 1. TRANSFORMAR EL OBJETIVO DE ENTRENAMIENTO
    # ==================================================

    y_train_delta = calcular_objetivo_delta(
        entrenamiento,
        target,
    )

    # ==================================================
    # 2. PREPARAR PREDICTORES
    # ==================================================

    X_train = entrenamiento[predictoras].copy()
    X_val = validacion[predictoras].copy()

    X_train = X_train.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    X_val = X_val.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # ==================================================
    # 3. CREAR Y ENTRENAR UN MODELO NUEVO
    # ==================================================

    modelo = construir_modelo(
        nombre,
        numericas,
    )

    modelo.fit(
        X_train,
        y_train_delta,
    )

    # ==================================================
    # 4. PREDECIR LAS VARIACIONES
    # ==================================================

    delta_predicha = modelo.predict(X_val)

    # ==================================================
    # 5. RECONSTRUIR LA MOROSIDAD FUTURA
    # ==================================================

    prediccion = reconstruir_morosidad(
        validacion["morosidad_pp"].to_numpy(dtype=float),
        delta_predicha,
        recortar=recortar,
    )

    return prediccion
