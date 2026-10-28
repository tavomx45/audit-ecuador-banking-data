"""Construcción de pipelines de Machine Learning."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

NOMBRES_MODELOS = (
    "Ridge",
    "RandomForest",
    "HistGradientBoosting",
)


def construir_modelo(
    nombre: str,
    columnas_numericas: list[str],
) -> Pipeline:
    """Construye un modelo con su preprocesamiento.

    Reproduce la configuración inicial del notebook académico.

    El preprocesamiento se ajusta únicamente al ejecutar fit().
    """

    if nombre not in NOMBRES_MODELOS:
        raise ValueError(f"Modelo desconocido: {nombre}")

    if not columnas_numericas:
        raise ValueError("Se requieren variables numéricas.")

    if len(set(columnas_numericas)) != len(columnas_numericas):
        raise ValueError("Existen variables numéricas duplicadas.")

    if "banco" in columnas_numericas:
        raise ValueError("La variable banco es categórica.")

    # ======================================
    # 1. PREPROCESAMIENTO NUMÉRICO
    # ======================================

    pasos_numericos = [
        (
            "imputador",
            SimpleImputer(strategy="median"),
        ),
    ]

    # Ridge requiere escalado.
    if nombre == "Ridge":
        pasos_numericos.append(
            (
                "escalador",
                RobustScaler(),
            )
        )

    pipeline_numerico = Pipeline(steps=pasos_numericos)

    # ======================================
    # 2. PREPROCESAMIENTO COMPLETO
    # ======================================

    preprocesador = ColumnTransformer(
        transformers=[
            (
                "numericas",
                pipeline_numerico,
                columnas_numericas,
            ),
            (
                "banco",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["banco"],
            ),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    # ======================================
    # 3. SELECCIONAR ESTIMADOR
    # ======================================

    if nombre == "Ridge":
        estimador = Ridge(
            alpha=10.0,
        )

    elif nombre == "RandomForest":
        estimador = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )

    else:
        estimador = HistGradientBoostingRegressor(
            max_iter=150,
            learning_rate=0.05,
            max_leaf_nodes=15,
            random_state=42,
        )

    # ======================================
    # 4. PIPELINE COMPLETO
    # ======================================

    return Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("modelo", estimador),
        ]
    )


# ======================================================
# CONFIGURACIÓN FINAL CANDIDATA
# Seleccionada mediante CV temporal de desarrollo
# ======================================================


ESTRATEGIA_POR_HORIZONTE = {
    1: "RandomForest_delta_flexible",
    2: "Persistencia_estacional",
    3: "Persistencia_estacional",
}


PARAMETROS_RF_FLEXIBLE = {
    "n_estimators": 150,
    "max_depth": 14,
    "min_samples_leaf": 3,
    "max_features": 0.7,
}


def construir_rf_delta_flexible(numericas):
    """Construye el RF flexible usando el pipeline existente."""

    modelo = construir_modelo(
        "RandomForest",
        numericas,
    )

    pasos_rf = [
        nombre
        for nombre, estimador in modelo.named_steps.items()
        if isinstance(estimador, RandomForestRegressor)
    ]

    if len(pasos_rf) != 1:
        raise ValueError("Se esperaba un único RandomForestRegressor.")

    nombre_rf = pasos_rf[0]

    parametros = {
        f"{nombre_rf}__{clave}": valor
        for clave, valor in PARAMETROS_RF_FLEXIBLE.items()
    }

    modelo.set_params(**parametros)

    return modelo
