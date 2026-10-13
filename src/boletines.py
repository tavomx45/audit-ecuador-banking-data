"""Extracción reproducible de boletines históricos de bancos privados (corte 2026-09).

Las proporciones (morosidad, solvencia, liquidez, etc.) se conservan entre 0 y 1.
Los saldos monetarios se conservan en miles de USD, según el boletín.
Saldos inválidos se marcan como NaN y quedan registrados en advertencias.
No se imputan valores faltantes ni se construye todavía el target t+1.
"""

from __future__ import annotations

from datetime import date, datetime
from numbers import Real
from pathlib import Path
import re
import unicodedata
import warnings

import numpy as np
import pandas as pd

# Posiciones verificadas en el formato histórico 2026-09.
# Cada registro: nombre_final -> (fila, fragmentos esperados en su etiqueta).
_INDICADORES_FINANCIEROS = {
    "solvencia": (8, ("SOLVENCIA",)),
    "liquidez": (15, ("FONDOS DISPONIBLES", "DEPOSITOS")),
    "roa": (19, ("ROA",)),
    "roe": (21, ("ROE",)),
    "cartera_depositos": (33, ("CARTERA", "DEPOSITOS")),
}

_INDICADORES_CARTERA = {
    "cartera_bruta": (18, ("CARTERA BRUTA",)),
    "cartera_improductiva": (40, ("CARTERA IMPRODUCTIVA",)),
    "morosidad_total": (92, ("MOROSIDAD TOTAL",)),
    "cobertura": (105, ("COBERTURA", "CARTERA")),
}


def _texto_normalizado(valor: object) -> str:
    """Normaliza espacios y acentos *para comparar*, sin cambiar el dato original."""
    texto = unicodedata.normalize("NFKD", str(valor))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto.replace("\xa0", " ").upper()).strip()


def _fecha_excel(valor: object):
    """Interpreta fechas Excel (datetime, texto o número serial)."""
    if pd.isna(valor):
        return pd.NaT
    if isinstance(valor, (pd.Timestamp, datetime, date)):
        resultado = pd.Timestamp(valor)
    elif isinstance(valor, Real) and not isinstance(valor, bool):
        resultado = pd.to_datetime(
            valor, unit="D", origin="1899-12-30", errors="coerce"
        )
    else:
        resultado = pd.to_datetime(str(valor).strip(), errors="coerce")
    if pd.isna(resultado) or not (1990 <= resultado.year <= 2100):
        return pd.NaT
    return resultado


def _extraer_hoja(
    hoja: pd.DataFrame,
    *,
    nombre_hoja: str,
    fila_fechas: int,
    primera_columna: int,
    columna_etiqueta: int,
    indicadores: dict[str, tuple[int, tuple[str, ...]]],
) -> pd.DataFrame:
    """Extrae una tabla mes x indicador verificando las etiquetas esperadas."""
    fechas_crudas = hoja.iloc[fila_fechas, primera_columna:]
    fechas = fechas_crudas.map(_fecha_excel).dropna()
    if fechas.empty:
        raise ValueError(f"{nombre_hoja}: no se encontraron fechas válidas")

    periodos = pd.DatetimeIndex(fechas.to_numpy()).to_period("M")
    if periodos.has_duplicates:
        raise ValueError(f"{nombre_hoja}: meses duplicados en el encabezado")

    resultado = pd.DataFrame({"fecha": periodos.to_timestamp()})
    for variable, (fila, fragmentos) in indicadores.items():
        etiqueta = _texto_normalizado(hoja.iat[fila, columna_etiqueta])
        if not all(fragmento in etiqueta for fragmento in fragmentos):
            raise ValueError(
                f"{nombre_hoja}, fila {fila}, variable {variable}: "
                f"etiqueta inesperada {etiqueta!r} (se esperaba {fragmentos})"
            )
        valores = pd.to_numeric(hoja.loc[fila, fechas.index], errors="coerce")
        resultado[variable] = valores.to_numpy(dtype=float)

    return resultado.sort_values("fecha").reset_index(drop=True)


def _limpiar_saldos_invalidos(serie: pd.DataFrame) -> pd.DataFrame:
    """Marca saldos no válidos como faltantes y conserva los meses.

    Los datos fuente permanecen intactos en data/raw/. Se deja constancia
    de los meses afectados en una advertencia que incluye sus fechas.
    """
    serie = serie.copy()
    bruta_no_positiva = serie["cartera_bruta"].le(0)
    improductiva_negativa = serie["cartera_improductiva"].lt(0)
    afectados = bruta_no_positiva | improductiva_negativa

    if afectados.any():
        banco = serie["banco"].iat[0]
        detalle = ", ".join(
            serie.loc[afectados, "fecha"].dt.strftime("%Y-%m").tolist()[:12]
        )
        if afectados.sum() > 12:
            detalle += ", ..."
        warnings.warn(
            f"{banco}: {int(bruta_no_positiva.sum())} mes(es) con cartera "
            f"bruta <= 0 y {int(improductiva_negativa.sum())} mes(es) con "
            f"cartera improductiva < 0. Períodos afectados: {detalle}. "
            "Se marcan como NaN los saldos inválidos y la morosidad de "
            "esos meses; deben revisarse contra el archivo original.",
            RuntimeWarning,
            stacklevel=2,
        )
        serie.loc[bruta_no_positiva, "cartera_bruta"] = np.nan
        serie.loc[improductiva_negativa, "cartera_improductiva"] = np.nan
        # No conservar como target un ratio con denominador o numerador inválido.
        serie.loc[afectados, "morosidad_total"] = np.nan

                # Identificar meses sin cartera improductiva
        sin_cartera_improductiva = (
            serie["cartera_improductiva"] == 0
        )

        # La cobertura no está definida cuando su denominador es cero
        serie.loc[
            sin_cartera_improductiva,
            "cobertura"
        ] = np.nan

    return serie


def validar_serie(serie: pd.DataFrame, *, tolerancia: float = 1e-8) -> dict:
    """Valida claves y continuidad; comprueba la morosidad cuando sea calculable.

    No imputa datos: permite NaN en indicadores para conservar la cronología.
    """
    if serie.empty:
        raise ValueError("Serie vacía")
    if serie[["fecha", "banco"]].isna().any().any():
        raise ValueError("Hay identificadores banco/fecha vacíos")
    if serie.duplicated(["banco", "fecha"]).any():
        raise ValueError("Existen registros banco-mes duplicados")
    if serie["banco"].nunique() != 1:
        raise ValueError("Una serie individual no puede contener varios bancos")

    periodos = pd.DatetimeIndex(serie["fecha"]).to_period("M")
    esperados = pd.period_range(periodos.min(), periodos.max(), freq="M")
    faltantes = esperados.difference(periodos)
    if len(faltantes):
        raise ValueError(f"Faltan meses dentro de la serie: {list(faltantes[:10])}")

    # Después de la limpieza, pueden existir NaN, pero nunca valores imposibles.
    if (serie["cartera_bruta"].dropna() <= 0).any():
        raise ValueError("Persisten saldos de cartera bruta no positivos")
    if (serie["cartera_improductiva"].dropna() < 0).any():
        raise ValueError("Persisten saldos de cartera improductiva negativos")

    necesarias = ["cartera_bruta", "cartera_improductiva", "morosidad_total"]
    comprobables = serie[necesarias].notna().all(axis=1)
    if not comprobables.any():
        raise ValueError("No hay meses con información suficiente para validar morosidad")

    calculada = (
        serie.loc[comprobables, "cartera_improductiva"]
        / serie.loc[comprobables, "cartera_bruta"]
    )
    diferencias = (serie.loc[comprobables, "morosidad_total"] - calculada).abs()
    maxima = float(diferencias.max())
    if not np.isfinite(maxima) or maxima > tolerancia:
        raise ValueError(
            f"Inconsistencia financiera: diferencia máxima {maxima:.3e} "
            f"(tolerancia {tolerancia:.1e})"
        )

    return {
        "banco": str(serie["banco"].iat[0]),
        "inicio": serie["fecha"].min(),
        "fin": serie["fecha"].max(),
        "meses": len(serie),
        "faltantes_solvencia": int(serie["solvencia"].isna().sum()),
        "faltantes_cartera_bruta": int(serie["cartera_bruta"].isna().sum()),
        "faltantes_cartera_improductiva": int(serie["cartera_improductiva"].isna().sum()),
        "faltantes_morosidad": int(serie["morosidad_total"].isna().sum()),
        "meses_morosidad_verificables": int(comprobables.sum()),
        "meses_morosidad_no_verificables": int((~comprobables).sum()),
        "diferencia_maxima_morosidad": maxima,
        "faltantes_cobertura": int(
        serie["cobertura"].isna().sum()),
        "meses_cartera_improductiva_cero": int(
        (serie["cartera_improductiva"] == 0).sum()),
    }


def extraer_serie_banco(ruta_archivo: str | Path) -> pd.DataFrame:
    """Extrae y valida la serie mensual de un boletín histórico por entidad."""
    ruta = Path(ruta_archivo)
    if not ruta.is_file():
        raise FileNotFoundError(ruta)

    hojas = pd.read_excel(
        ruta, sheet_name=["INDICAD", "INDIC CARTERA"], header=None
    )
    indicad = hojas["INDICAD"]
    cartera = hojas["INDIC CARTERA"]

    banco_indicad = _texto_normalizado(indicad.iat[1, 0])
    banco_cartera = _texto_normalizado(cartera.iat[1, 0])
    if banco_indicad != banco_cartera or not banco_indicad.startswith(("BP ", "BANCO ")):
        raise ValueError(
            f"Institución no identificada de forma consistente en {ruta.name}: "
            f"{banco_indicad!r} / {banco_cartera!r}"
        )

    financiero = _extraer_hoja(
        indicad,
        nombre_hoja="INDICAD",
        fila_fechas=6,
        primera_columna=1,
        columna_etiqueta=0,
        indicadores=_INDICADORES_FINANCIEROS,
    )
    crediticio = _extraer_hoja(
        cartera,
        nombre_hoja="INDIC CARTERA",
        fila_fechas=4,
        primera_columna=3,
        columna_etiqueta=2,
        indicadores=_INDICADORES_CARTERA,
    )

    if set(financiero["fecha"]) != set(crediticio["fecha"]):
        raise ValueError(f"Las fechas de las dos hojas no coinciden: {ruta.name}")

    serie = financiero.merge(
        crediticio, on="fecha", how="inner", validate="one_to_one"
    )
    serie.insert(1, "banco", banco_indicad)
    serie = serie.sort_values("fecha").reset_index(drop=True)
    serie = _limpiar_saldos_invalidos(serie)
    validar_serie(serie)
    return serie


def construir_panel(
    directorio: str | Path, *, patron: str = "*2026_09.xlsx"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Procesa archivos de un mismo corte y devuelve (panel, reporte de calidad)."""
    archivos = sorted(
        p for p in Path(directorio).glob(patron)
        if p.is_file() and not p.name.startswith("~$")
    )
    if not archivos:
        raise FileNotFoundError(f"No hay archivos con patrón {patron!r} en {directorio}")

    series, reportes = [], []
    for archivo in archivos:
        try:
            serie = extraer_serie_banco(archivo)
            reporte = validar_serie(serie)
        except Exception as exc:
            raise RuntimeError(f"Error al procesar {archivo.name}: {exc}") from exc
        series.append(serie)
        reportes.append({"archivo": archivo.name, **reporte})

    panel = pd.concat(series, ignore_index=True)
    if panel.duplicated(["banco", "fecha"]).any():
        raise ValueError("Hay bancos y meses duplicados entre los archivos")
    panel = panel.sort_values(["banco", "fecha"]).reset_index(drop=True)
    return panel, pd.DataFrame(reportes).sort_values("banco").reset_index(drop=True)
