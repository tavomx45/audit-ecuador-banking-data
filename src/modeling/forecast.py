"""Predicción multihorizonte de morosidad bancaria."""

import numpy as np

from src.modeling.baselines import (
    predecir_persistencia_estacional,
)
from src.modeling.models import (
    ESTRATEGIA_POR_HORIZONTE,
    construir_rf_delta_flexible,
)


def entrenar_y_predecir_horizonte(
    horizonte,
    entrenamiento,
    validacion,
    target,
    fecha_target,
    numericas,
    predictoras,
):
    """Entrena la estrategia elegida y predice morosidad futura."""

    if horizonte not in ESTRATEGIA_POR_HORIZONTE:
        raise ValueError(f"Horizonte no soportado: {horizonte}")

    # ==========================================
    # HORIZONTE 1: RANDOM FOREST DELTA
    # ==========================================

    if horizonte == 1:
        X_train = entrenamiento[predictoras].replace([np.inf, -np.inf], np.nan)

        X_pred = validacion[predictoras].replace([np.inf, -np.inf], np.nan)

        y_train_delta = entrenamiento[target] - entrenamiento["morosidad_pp"]

        if not np.isfinite(y_train_delta.to_numpy(dtype=float)).all():
            raise ValueError(
                "El objetivo de entrenamiento contiene valores no finitos."
            )

        modelo = construir_rf_delta_flexible(numericas)

        modelo.fit(
            X_train,
            y_train_delta,
        )

        delta_predicha = modelo.predict(X_pred)

        predicciones = validacion["morosidad_pp"].to_numpy(dtype=float) + delta_predicha

    # ==========================================
    # HORIZONTES 2 Y 3: BASELINE ESTACIONAL
    # ==========================================

    else:
        predicciones = predecir_persistencia_estacional(
            entrenamiento,
            validacion,
            target,
            fecha_target,
        )

    predicciones = np.asarray(
        predicciones,
        dtype=float,
    ).reshape(-1)

    if len(predicciones) != len(validacion):
        raise ValueError(
            "La cantidad de predicciones no coincide con las observaciones."
        )

    if not np.isfinite(predicciones).all():
        raise ValueError("Se generaron predicciones no finitas.")

    return predicciones
