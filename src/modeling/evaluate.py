"""Evaluación comparativa de modelos predictivos de morosidad."""

import numpy as np
import pandas as pd

MODELOS_BASELINE = (
    "Persistencia",
    "Persistencia_estacional",
)

COLUMNAS_REQUERIDAS = {
    "horizonte",
    "fold",
    "modelo",
    "observaciones",
    "mae_pp",
}


def comparar_modelos_cv(
    detalle_baselines: pd.DataFrame,
    detalle_supervisados: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compara modelos mediante sus resultados de validación temporal.

    Cada modelo debe utilizar los mismos folds y tamaños de
    validación dentro de un horizonte.

    El MAE promedio es la media simple de los folds,
    siguiendo la metodología académica original.
    """

    # ====================================================
    # 1. VALIDAR LOS RESULTADOS RECIBIDOS
    # ====================================================

    for nombre, datos in (
        ("baselines", detalle_baselines),
        ("supervisados", detalle_supervisados),
    ):
        faltantes = COLUMNAS_REQUERIDAS - set(datos.columns)

        if faltantes:
            raise ValueError(f"Faltan columnas en {nombre}: {sorted(faltantes)}")

        if datos.empty:
            raise ValueError(f"Resultados vacíos: {nombre}.")

        if datos[list(COLUMNAS_REQUERIDAS)].isna().any().any():
            raise ValueError(f"Existen valores faltantes en {nombre}.")

    if not set(detalle_baselines["modelo"]).issubset(MODELOS_BASELINE):
        raise ValueError("Se encontraron baselines desconocidos.")

    if set(detalle_supervisados["modelo"]).intersection(MODELOS_BASELINE):
        raise ValueError("Un baseline aparece entre los modelos supervisados.")

    # ====================================================
    # 2. UNIFICAR RESULTADOS
    # ====================================================

    detalle = pd.concat(
        [
            detalle_baselines.assign(tipo="Baseline"),
            detalle_supervisados.assign(tipo="Supervisado"),
        ],
        ignore_index=True,
    )

    claves = ["horizonte", "fold", "modelo"]

    if detalle.duplicated(claves).any():
        raise ValueError(
            "Existen evaluaciones duplicadas por horizonte, fold y modelo."
        )

    if not np.isfinite(detalle["mae_pp"].to_numpy(dtype=float)).all():
        raise ValueError("Existen valores MAE no finitos.")

    if (detalle["mae_pp"] < 0).any():
        raise ValueError("El MAE no puede ser negativo.")

    if (detalle["observaciones"] <= 0).any():
        raise ValueError("Los tamaños de validación deben ser positivos.")

    # ====================================================
    # 3. COMPROBAR COMPARABILIDAD DE FOLDS
    # ====================================================

    for horizonte, datos_h in detalle.groupby("horizonte"):
        modelos = set(datos_h["modelo"])

        if not set(MODELOS_BASELINE).issubset(modelos):
            raise ValueError(f"Faltan modelos baseline en horizonte {horizonte}.")

        conjuntos_folds = [set(grupo["fold"]) for _, grupo in datos_h.groupby("modelo")]

        referencia_folds = conjuntos_folds[0]

        if any(folds != referencia_folds for folds in conjuntos_folds):
            raise ValueError(
                f"Los modelos no comparten los mismos folds: h={horizonte}."
            )

        tamaños = datos_h.groupby("fold")["observaciones"].nunique()

        if (tamaños != 1).any():
            raise ValueError(f"Tamaños de validación inconsistentes: h={horizonte}.")

    # ====================================================
    # 4. CALCULAR MÉTRICAS AGREGADAS
    # ====================================================

    resumen = (
        detalle.groupby(["horizonte", "modelo", "tipo"])
        .agg(
            mae_medio=("mae_pp", "mean"),
            mae_std=("mae_pp", "std"),
            folds=("fold", "count"),
        )
        .reset_index()
    )

    # ====================================================
    # 5. IDENTIFICAR EL MEJOR BASELINE
    # ====================================================

    referencias = (
        resumen.loc[resumen["modelo"].isin(MODELOS_BASELINE)]
        .sort_values(["horizonte", "mae_medio", "modelo"])
        .drop_duplicates("horizonte")[["horizonte", "modelo", "mae_medio"]]
        .rename(
            columns={
                "modelo": "baseline_referencia",
                "mae_medio": "mae_baseline",
            }
        )
    )

    resumen = resumen.merge(
        referencias,
        on="horizonte",
        how="left",
        validate="many_to_one",
    )

    # ====================================================
    # 6. COMPARAR FRENTE AL BASELINE
    # ====================================================

    resumen["diferencia_mae_pp"] = resumen["mae_medio"] - resumen["mae_baseline"]

    # Una referencia con MAE cero no permite
    # calcular una mejora porcentual definida.
    denominador = resumen["mae_baseline"].where(resumen["mae_baseline"] > 0)

    resumen["mejora_pct"] = (
        (resumen["mae_baseline"] - resumen["mae_medio"]).div(denominador).mul(100)
    )

    resumen = resumen.sort_values(["horizonte", "mae_medio", "modelo"]).reset_index(
        drop=True
    )

    resumen["posicion"] = resumen.groupby("horizonte").cumcount() + 1

    return detalle, resumen


def evaluar_sistema_final_cv(detalle_final, detalle_baselines):
    """Compara el sistema híbrido con persistencia estacional.

    Ambos métodos deben haberse evaluado sobre
    los mismos horizontes y folds temporales.
    """

    claves = ["horizonte", "fold"]

    # ==================================================
    # 1. PREPARAR BASELINE ESTACIONAL
    # ==================================================

    baseline = detalle_baselines.loc[
        detalle_baselines["modelo"] == "Persistencia_estacional",
        [
            "horizonte",
            "fold",
            "observaciones",
            "mae_pp",
        ],
    ].copy()

    baseline = baseline.rename(
        columns={
            "observaciones": "observaciones_baseline",
            "mae_pp": "mae_estacional",
        }
    )

    if baseline.duplicated(claves).any():
        raise ValueError("El baseline contiene folds duplicados.")

    if detalle_final.duplicated(claves).any():
        raise ValueError("El sistema final contiene folds duplicados.")

    # ==================================================
    # 2. ALINEAR AMBAS EVALUACIONES
    # ==================================================

    comparacion = detalle_final.merge(
        baseline,
        on=claves,
        how="inner",
        validate="one_to_one",
    )

    if len(comparacion) != len(detalle_final):
        raise ValueError("Faltan folds del baseline estacional.")

    if not (
        comparacion["observaciones"] == comparacion["observaciones_baseline"]
    ).all():
        raise ValueError(
            "Los modelos no tienen el mismo número de observaciones por fold."
        )

    # ==================================================
    # 3. DIFERENCIAS PAREADAS
    # ==================================================

    comparacion["ventaja_pp"] = comparacion["mae_estacional"] - comparacion["mae_pp"]

    comparacion["mejora_pct"] = (
        100 * comparacion["ventaja_pp"] / comparacion["mae_estacional"]
    )

    tolerancia = 1e-10

    comparacion["gana"] = comparacion["ventaja_pp"] > tolerancia

    comparacion["empata"] = comparacion["ventaja_pp"].abs() <= tolerancia

    comparacion["pierde"] = comparacion["ventaja_pp"] < -tolerancia

    # ==================================================
    # 4. RESUMEN POR HORIZONTE
    # ==================================================

    resumen = comparacion.groupby("horizonte", as_index=False).agg(
        estrategia=("estrategia", "first"),
        mae_medio=("mae_pp", "mean"),
        mae_std=("mae_pp", "std"),
        rmse_medio=("rmse_pp", "mean"),
        sesgo_medio=("sesgo_pp", "mean"),
        mae_estacional=("mae_estacional", "mean"),
        folds=("fold", "count"),
        folds_ganados=("gana", "sum"),
        folds_empatados=("empata", "sum"),
        folds_perdidos=("pierde", "sum"),
    )

    resumen["ventaja_media_pp"] = resumen["mae_estacional"] - resumen["mae_medio"]

    resumen["mejora_vs_estacional_pct"] = (
        100 * resumen["ventaja_media_pp"] / resumen["mae_estacional"]
    )

    if len(resumen) != 3:
        raise ValueError("Se esperaban tres horizontes.")

    if not (resumen["folds"] == 5).all():
        raise ValueError("Cada horizonte debe tener cinco folds.")

    return resumen, comparacion
