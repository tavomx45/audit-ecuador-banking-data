"""Ingeniería de características del panel bancario."""

import pandas as pd

HORIZONTES = (1, 2, 3)


def construir_targets(
    panel: pd.DataFrame,
    horizontes=HORIZONTES,
) -> pd.DataFrame:
    """Construye objetivos de morosidad futura por banco y mes."""

    df = panel.copy()

    df["fecha"] = pd.to_datetime(df["fecha"])

    df = df.sort_values(["banco", "fecha"]).reset_index(drop=True)

    # Validaciones previas
    if df.duplicated(["banco", "fecha"]).any():
        raise ValueError("Existen registros banco-mes duplicados.")

    if df["fecha"].isna().any():
        raise ValueError("Existen fechas faltantes.")

    if not df["fecha"].dt.is_month_start.all():
        raise ValueError("Las fechas deben iniciar el mes.")

    # Morosidad expresada en puntos porcentuales
    df["morosidad_pp"] = df["morosidad_total"] * 100

    for h in horizontes:
        if not isinstance(h, int) or h < 1:
            raise ValueError("Los horizontes deben ser enteros positivos.")

        nombre_target = f"target_morosidad_t{h}_pp"

        # Crear tabla auxiliar con información futura
        futuro = df[["banco", "fecha", "morosidad_pp"]].copy()

        futuro["fecha"] = futuro["fecha"] - pd.DateOffset(months=h)

        futuro = futuro.rename(columns={"morosidad_pp": nombre_target})

        # Unir por banco y mes calendario
        df = df.merge(
            futuro,
            on=["banco", "fecha"],
            how="left",
            validate="one_to_one",
        )

        # Registrar la fecha del objetivo
        df[f"fecha_objetivo_t{h}"] = df["fecha"] + pd.DateOffset(months=h)

    return df


REZAGOS = (1, 2, 3, 6, 12)


def construir_rezagos_y_variaciones(
    datos: pd.DataFrame,
    rezagos=REZAGOS,
) -> pd.DataFrame:
    """Genera rezagos y variaciones usando meses calendario."""

    df = datos.copy()

    columnas_requeridas = {
        "banco",
        "fecha",
        "morosidad_pp",
    }

    if not columnas_requeridas.issubset(df.columns):
        raise ValueError("Faltan columnas requeridas.")

    if df.duplicated(["banco", "fecha"]).any():
        raise ValueError("Existen registros banco-mes duplicados.")

    if not df["fecha"].dt.is_month_start.all():
        raise ValueError("Las fechas deben iniciar el mes.")

    # Construir rezagos históricos
    for k in rezagos:
        if not isinstance(k, int) or k < 1:
            raise ValueError("Los rezagos deben ser enteros positivos.")

        nombre_lag = f"morosidad_lag_{k}_pp"

        if nombre_lag in df.columns:
            raise ValueError(f"La columna {nombre_lag} ya existe.")

        historico = datos[["banco", "fecha", "morosidad_pp"]].copy()

        # Desplazar la fecha histórica hacia adelante
        historico["fecha"] = historico["fecha"] + pd.DateOffset(months=k)

        historico = historico.rename(columns={"morosidad_pp": nombre_lag})

        df = df.merge(
            historico,
            on=["banco", "fecha"],
            how="left",
            validate="one_to_one",
        )

    # Variación mensual
    df["delta_morosidad_1m_pp"] = df["morosidad_pp"] - df["morosidad_lag_1_pp"]

    # Variación acumulada de tres meses
    df["delta_morosidad_3m_pp"] = df["morosidad_pp"] - df["morosidad_lag_3_pp"]

    return df


VENTANAS = (3, 6)


def construir_estadisticas_moviles(
    datos: pd.DataFrame,
    ventanas=VENTANAS,
) -> pd.DataFrame:
    """Calcula estadísticas móviles por banco y mes calendario."""

    df = datos.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    requeridas = {"banco", "fecha", "morosidad_pp"}

    if not requeridas.issubset(df.columns):
        raise ValueError("Faltan columnas requeridas.")

    if df.empty:
        raise ValueError("El DataFrame está vacío.")

    if df.duplicated(["banco", "fecha"]).any():
        raise ValueError("Existen registros banco-mes duplicados.")

    if df["fecha"].isna().any():
        raise ValueError("Existen fechas faltantes.")

    if not df["fecha"].dt.is_month_start.all():
        raise ValueError("Las fechas deben iniciar el mes.")

    if any(not isinstance(w, int) or w < 1 for w in ventanas):
        raise ValueError("Las ventanas deben ser enteros positivos.")

    resultados = []

    for banco, grupo in df.groupby("banco"):
        # Completar el índice mensual, incluso si faltan meses
        serie = grupo.set_index("fecha")["morosidad_pp"].sort_index().asfreq("MS")

        auxiliar = pd.DataFrame(
            {
                "banco": banco,
                "fecha": serie.index,
            }
        )

        for w in ventanas:
            movil = serie.rolling(
                window=w,
                min_periods=w,
            )

            auxiliar[f"morosidad_media_{w}m_pp"] = movil.mean().to_numpy()

            auxiliar[f"morosidad_std_{w}m_pp"] = movil.std(ddof=0).to_numpy()

        resultados.append(auxiliar)

    moviles = pd.concat(
        resultados,
        ignore_index=True,
    )

    return df.merge(
        moviles,
        on=["banco", "fecha"],
        how="left",
        validate="one_to_one",
    )


def construir_caracteristicas_estacionales(
    datos: pd.DataFrame,
    horizontes=HORIZONTES,
) -> pd.DataFrame:
    """Genera características cíclicas del mes objetivo."""

    import numpy as np

    df = datos.copy()

    for h in horizontes:
        if not isinstance(h, int) or h < 1:
            raise ValueError("Los horizontes deben ser enteros positivos.")

        columna_fecha = f"fecha_objetivo_t{h}"

        if columna_fecha not in df.columns:
            raise ValueError(f"No existe la columna {columna_fecha}")

        # Obtener el mes correspondiente al objetivo
        mes = pd.to_datetime(df[columna_fecha]).dt.month

        # Registrar el mes objetivo
        df[f"mes_objetivo_t{h}"] = mes

        # Codificación cíclica
        angulo = 2 * np.pi * mes / 12

        df[f"mes_sin_t{h}"] = np.sin(angulo)

        df[f"mes_cos_t{h}"] = np.cos(angulo)

    return df


REZAGOS_CARTERA = (1, 3)


def construir_crecimiento_cartera(
    datos: pd.DataFrame,
    rezagos=REZAGOS_CARTERA,
) -> pd.DataFrame:
    """Calcula crecimientos de cartera por banco y mes calendario."""

    df = datos.copy()

    df["fecha"] = pd.to_datetime(df["fecha"])

    requeridas = {"banco", "fecha", "cartera_bruta"}

    if not requeridas.issubset(df.columns):
        raise ValueError("Faltan columnas requeridas.")

    if df.duplicated(["banco", "fecha"]).any():
        raise ValueError("Existen registros banco-mes duplicados.")

    if df["fecha"].isna().any():
        raise ValueError("Existen fechas faltantes.")

    if not df["fecha"].dt.is_month_start.all():
        raise ValueError("Las fechas deben iniciar el mes.")

    for k in rezagos:
        if not isinstance(k, int) or k < 1:
            raise ValueError("Los rezagos deben ser enteros positivos.")

        nombre_lag = f"cartera_bruta_lag_{k}"
        nombre_crecimiento = f"crecimiento_cartera_{k}m_pct"

        if nombre_lag in df.columns or nombre_crecimiento in df.columns:
            raise ValueError("Las características ya existen.")

        # Recuperar los saldos históricos mediante fechas calendario
        historico = datos[["banco", "fecha", "cartera_bruta"]].copy()

        historico["fecha"] = historico["fecha"] + pd.DateOffset(months=k)

        historico = historico.rename(columns={"cartera_bruta": nombre_lag})

        df = df.merge(
            historico,
            on=["banco", "fecha"],
            how="left",
            validate="one_to_one",
        )

        actual = df["cartera_bruta"]
        anterior = df[nombre_lag]

        # La tasa requiere ambos saldos estrictamente positivos
        validos = actual.gt(0) & anterior.gt(0)

        df[nombre_crecimiento] = ((actual / anterior - 1) * 100).where(validos)

    return df


def construir_transformaciones_financieras(
    datos: pd.DataFrame,
) -> pd.DataFrame:
    """Genera las transformaciones logarítmicas financieras."""

    import numpy as np

    df = datos.copy()

    requeridas = {
        "cartera_bruta",
        "cartera_bruta_lag_1",
        "cartera_bruta_lag_3",
        "liquidez",
        "cartera_depositos",
    }

    if not requeridas.issubset(df.columns):
        raise ValueError("Faltan columnas financieras requeridas.")

    nuevas = [
        "crecimiento_log_cartera_1m",
        "crecimiento_log_cartera_3m",
        "liquidez_log",
        "cartera_depositos_log",
    ]

    if any(col in df.columns for col in nuevas):
        raise ValueError("Las transformaciones ya existen.")

    # ======================================
    # A. CRECIMIENTOS LOGARÍTMICOS
    # ======================================

    for k in (1, 3):
        actual = df["cartera_bruta"]
        anterior = df[f"cartera_bruta_lag_{k}"]

        validos = (
            actual.gt(0) & anterior.gt(0) & np.isfinite(actual) & np.isfinite(anterior)
        )

        df[f"crecimiento_log_cartera_{k}m"] = np.log(
            actual.where(validos) / anterior.where(validos)
        )

    # ======================================
    # B. RAZONES FINANCIERAS
    # ======================================

    transformaciones = {
        "liquidez": "liquidez_log",
        "cartera_depositos": "cartera_depositos_log",
    }

    for original, nueva in transformaciones.items():
        valores = df[original]

        validos = valores.ge(0) & np.isfinite(valores)

        df[nueva] = np.log1p(valores.where(validos))

    return df


def construir_dataset_multihorizonte(
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Construye las características multihorizonte del panel bancario."""

    datos = construir_targets(panel)
    datos = construir_rezagos_y_variaciones(datos)
    datos = construir_estadisticas_moviles(datos)
    datos = construir_caracteristicas_estacionales(datos)
    datos = construir_crecimiento_cartera(datos)
    datos = construir_transformaciones_financieras(datos)

    return datos
