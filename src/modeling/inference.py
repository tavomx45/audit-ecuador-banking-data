"""Inferencia de morosidad bancaria con modelos registrados en MLflow."""

import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

MODELO_RF_H1 = "models:/MorosidadRFDeltaH1/2"


def _pronosticar_estacional(
    datos: pd.DataFrame,
    horizonte: int,
) -> pd.DataFrame:
    """Predice mediante ajustes estacionales previamente aprendidos."""

    ruta = (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "modeling"
        / f"seasonal_h{horizonte}.json"
    )

    if not ruta.is_file():
        raise FileNotFoundError(f"No existe el artefacto estacional: {ruta}")

    artefacto = json.loads(ruta.read_text(encoding="utf-8"))

    if artefacto["horizonte"] != horizonte:
        raise ValueError("El artefacto no corresponde al horizonte solicitado.")

    ajustes = {int(mes): float(valor) for mes, valor in artefacto["ajustes"].items()}

    if not all(1 <= mes <= 12 and np.isfinite(valor) for mes, valor in ajustes.items()):
        raise ValueError("El artefacto contiene ajustes inválidos.")

    # Validar identificadores y valores de origen.
    if datos["banco"].isna().any():
        raise ValueError("Hay bancos sin identificación.")

    bancos = datos["banco"].astype(str).str.strip()

    if bancos.eq("").any():
        raise ValueError("Hay nombres de bancos vacíos.")

    fechas = pd.to_datetime(
        datos["fecha"],
        errors="coerce",
    )

    if fechas.isna().any():
        raise ValueError("Hay fechas de origen inválidas.")

    periodos_origen = fechas.dt.to_period("M")

    identificadores = pd.DataFrame(
        {
            "banco": bancos.to_numpy(),
            "periodo": periodos_origen.to_numpy(),
        }
    )

    if identificadores.duplicated().any():
        raise ValueError("Existen registros banco-mes duplicados.")

    # Evitar usar un artefacto entrenado con datos posteriores
    # para simular pronósticos históricos anteriores al corte.
    ultimo_mes_entrenamiento = pd.Period(
        artefacto["fecha_objetivo_max_train"],
        freq="M",
    )

    if (periodos_origen <= ultimo_mes_entrenamiento).any():
        raise ValueError(
            "El origen del pronóstico no es posterior al corte del entrenamiento."
        )

    actual = pd.to_numeric(
        datos["morosidad_pp"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(actual).all():
        raise ValueError("La morosidad actual debe ser finita.")

    # Fecha objetivo: t + horizonte.
    periodos_objetivo = periodos_origen + horizonte

    fecha_objetivo = periodos_objetivo.dt.to_timestamp(how="end").dt.normalize()

    # Aplicar exactamente la lógica del baseline:
    # meses sin ajuste aprendido reciben corrección cero.
    correcciones = (
        periodos_objetivo.dt.month.map(ajustes).fillna(0).to_numpy(dtype=float)
    )

    y_pred = np.clip(
        actual + correcciones,
        0,
        100,
    )

    return pd.DataFrame(
        {
            "banco": bancos.to_numpy(),
            "fecha_origen": fechas.to_numpy(),
            "fecha_objetivo": fecha_objetivo.to_numpy(),
            "horizonte": horizonte,
            "morosidad_actual_pp": actual,
            "delta_predicha_pp": y_pred - actual,
            "morosidad_predicha_pp": y_pred,
            "modelo": "Persistencia_estacional",
            "model_uri": None,
        }
    )


def pronosticar_morosidad(
    datos: pd.DataFrame,
    horizonte: int = 1,
    model_uri: str = MODELO_RF_H1,
    tracking_uri: str | None = None,
) -> pd.DataFrame:
    """Genera pronósticos con un modelo entrenado y registrado.

    Parameters
    ----------
    datos : pd.DataFrame
        Observaciones con banco, fecha y predictores del modelo.
    horizonte : int
        Horizonte de predicción. Actualmente se implementa h=1.
    model_uri : str
        Identificador del modelo en MLflow.
    tracking_uri : str, optional
        Dirección del servidor MLflow.

    Returns
    -------
    pd.DataFrame
        Predicciones de morosidad en puntos porcentuales.
    """

    # ==================================================
    # 1. VALIDAR HORIZONTE
    # ==================================================

    if horizonte not in (1, 2, 3):
        raise ValueError(f"Horizonte no soportado: {horizonte}")

    # ==================================================
    # 2. VALIDAR ENTRADA
    # ==================================================

    if not isinstance(datos, pd.DataFrame) or datos.empty:
        raise ValueError("Se requiere un DataFrame con observaciones.")

    requeridas = {"banco", "fecha", "morosidad_pp"}

    faltantes = requeridas - set(datos.columns)

    if faltantes:
        raise ValueError(f"Faltan columnas: {sorted(faltantes)}")

    # ==================================================
    # 3. PROHIBIR ETIQUETAS FUTURAS EN TODOS LOS HORIZONTES
    # ==================================================

    columnas_objetivo = [
        columna for columna in datos.columns if columna.startswith("target_")
    ]

    if columnas_objetivo:
        raise ValueError(f"La entrada contiene etiquetas futuras: {columnas_objetivo}")

    # ==================================================
    # 4. PERSISTENCIA ESTACIONAL: HORIZONTES 2 Y 3
    # ==================================================

    if horizonte in (2, 3):
        return _pronosticar_estacional(
            datos=datos,
            horizonte=horizonte,
        )

    # ==================================================
    # 5. RANDOM FOREST: HORIZONTE 1

    uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(uri)

    modelo = mlflow.sklearn.load_model(model_uri)

    if not hasattr(modelo, "feature_names_in_"):
        raise ValueError("El modelo no contiene nombres de predictores.")

    predictoras = list(modelo.feature_names_in_)

    if len(predictoras) != 22:
        raise ValueError("El modelo no tiene las 22 características esperadas.")

    faltantes = set(predictoras) - set(datos.columns)

    if faltantes:
        raise ValueError(f"Faltan predictores: {sorted(faltantes)}")

    # ==================================================
    # 4. VALIDAR IDENTIFICADORES Y FECHAS
    # ==================================================

    if datos["banco"].isna().any():
        raise ValueError("Existen bancos sin identificación.")

    fechas = pd.to_datetime(
        datos["fecha"],
        errors="coerce",
    )

    if fechas.isna().any():
        raise ValueError("Existen fechas inválidas.")

    bancos = datos["banco"].astype(str).str.strip()

    if bancos.eq("").any():
        raise ValueError("Existen nombres de bancos vacíos.")

    identificadores = pd.DataFrame(
        {
            "banco": bancos,
            "periodo": fechas.dt.to_period("M"),
        }
    )

    if identificadores.duplicated().any():
        raise ValueError("Existen observaciones banco-mes duplicadas.")

    morosidad_actual = pd.to_numeric(
        datos["morosidad_pp"],
        errors="raise",
    ).to_numpy(dtype=float)

    if not np.isfinite(morosidad_actual).all():
        raise ValueError("La morosidad actual debe ser finita.")

    # ==================================================
    # 5. PREPARAR LOS PREDICTORES
    # ==================================================

    X = datos[predictoras].copy()

    X["banco"] = bancos

    for columna in predictoras:
        if columna == "banco":
            continue

        X[columna] = pd.to_numeric(
            X[columna],
            errors="raise",
        )

    # La imputación ya pertenece al pipeline.
    X = X.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # ==================================================
    # 6. PREDECIR EL CAMBIO DE MOROSIDAD
    # ==================================================

    delta_predicha = np.asarray(
        modelo.predict(X),
        dtype=float,
    ).reshape(-1)

    if len(delta_predicha) != len(datos) or not np.isfinite(delta_predicha).all():
        raise ValueError("El modelo produjo predicciones inválidas.")

    # ==================================================
    # 7. RECONSTRUIR EL NIVEL FUTURO
    # ==================================================

    morosidad_futura = morosidad_actual + delta_predicha

    fecha_objetivo = (
        (fechas.dt.to_period("M") + horizonte).dt.to_timestamp(how="end").dt.normalize()
    )

    # ==================================================
    # 8. RESULTADOS
    # ==================================================

    resultado = pd.DataFrame(
        {
            "banco": bancos.to_numpy(),
            "fecha_origen": fechas.to_numpy(),
            "fecha_objetivo": fecha_objetivo.to_numpy(),
            "horizonte": horizonte,
            "morosidad_actual_pp": morosidad_actual,
            "delta_predicha_pp": delta_predicha,
            "morosidad_predicha_pp": morosidad_futura,
            "modelo": "RandomForest_delta_flexible",
            "model_uri": model_uri,
        }
    )

    return resultado
