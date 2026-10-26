"""Entrenamiento supervisado con validación cruzada temporal."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.modeling.models import NOMBRES_MODELOS, construir_modelo


def evaluar_modelos_cv(
    datasets: dict,
    folds_cv: dict,
    nombres_modelos=NOMBRES_MODELOS,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entrena y evalúa modelos sobre ventanas temporales expansivas.

    Cada combinación de horizonte, fold y algoritmo crea un
    pipeline independiente.

    Devuelve:
        detalle: MAE de cada entrenamiento.
        resumen: estadísticas por horizonte y algoritmo.
    """

    nombres_modelos = tuple(nombres_modelos)

    if not nombres_modelos:
        raise ValueError("Debe especificarse al menos un modelo.")

    resultados = []

    # ==========================================
    # 1. RECORRER HORIZONTES
    # ==========================================

    for h, configuracion in datasets.items():
        if h not in folds_cv:
            raise ValueError(f"No existen folds para horizonte {h}.")

        datos = configuracion["desarrollo"]

        target = configuracion["target"]
        fecha_target = configuracion["fecha_target"]

        numericas = configuracion["numericas"]
        predictoras = configuracion["predictoras"]

        if verbose:
            print(
                f"\n{'=' * 50}\nHORIZONTE: {h} MES(ES)\n{'=' * 50}",
                flush=True,
            )

        # ======================================
        # 2. RECORRER FOLDS
        # ======================================

        for fold in folds_cv[h]:
            anio = fold["fold"]

            train = datos.loc[fold["train_idx"]]
            val = datos.loc[fold["val_idx"]]

            if train.empty or val.empty:
                raise ValueError(f"Partición vacía: h={h}, fold={anio}.")

            # Comprobar que todas las etiquetas de
            # entrenamiento sean anteriores a la
            # primera fecha de validación.
            if train[fecha_target].max() >= val["fecha"].min():
                raise ValueError(f"Fuga temporal: h={h}, fold={anio}.")

            # ==================================
            # 3. PREPARAR X E y
            # ==================================

            X_train = train[predictoras].copy()
            y_train = train[target]

            X_val = val[predictoras].copy()
            y_val = val[target]

            if y_train.isna().any() or y_val.isna().any():
                raise ValueError(f"Targets faltantes: h={h}, fold={anio}.")

            # Reproducir el tratamiento original
            # de valores infinitos.
            X_train = X_train.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            X_val = X_val.replace(
                [np.inf, -np.inf],
                np.nan,
            )

            # ==================================
            # 4. ENTRENAR CADA ALGORITMO
            # ==================================

            for nombre in nombres_modelos:
                # Nueva instancia para cada fold.
                modelo = construir_modelo(
                    nombre,
                    numericas,
                )

                # El preprocesamiento se ajusta
                # exclusivamente con X_train.
                modelo.fit(X_train, y_train)

                predicciones = modelo.predict(X_val)

                if not np.isfinite(predicciones).all():
                    raise ValueError(
                        f"Predicciones no finitas: h={h}, fold={anio}, modelo={nombre}."
                    )

                # ==================================
                # 5. EVALUAR
                # ==================================

                mae = mean_absolute_error(
                    y_val,
                    predicciones,
                )

                resultados.append(
                    {
                        "horizonte": h,
                        "fold": anio,
                        "modelo": nombre,
                        "observaciones": len(val),
                        "mae_pp": mae,
                    }
                )

                if verbose:
                    print(
                        f"Fold {anio} | {nombre:<21} | MAE: {mae:.4f} pp",
                        flush=True,
                    )

    # ==========================================
    # 6. CONSTRUIR RESULTADOS
    # ==========================================

    detalle = pd.DataFrame(resultados)

    if detalle.empty:
        raise ValueError("No se realizaron entrenamientos.")

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
