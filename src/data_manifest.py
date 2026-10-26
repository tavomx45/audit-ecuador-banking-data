"""Generación y verificación del manifiesto de datos originales."""

import argparse
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"

MANIFEST_PATH = PROJECT_ROOT / "data" / "metadata" / "raw_manifest.json"


def calcular_sha256(archivo: Path) -> str:
    """Calcula SHA-256 leyendo el archivo por bloques."""

    hash_obj = hashlib.sha256()

    with archivo.open("rb") as f:
        for bloque in iter(lambda: f.read(1024 * 1024), b""):
            hash_obj.update(bloque)

    return hash_obj.hexdigest()


def inventariar_archivos() -> dict:
    """Construye el inventario de archivos originales."""

    archivos = sorted(archivo for archivo in RAW_DIR.rglob("*") if archivo.is_file())

    if not archivos:
        raise RuntimeError("No se encontraron archivos originales.")

    inventario = {}

    for archivo in archivos:
        ruta_relativa = archivo.relative_to(PROJECT_ROOT).as_posix()

        inventario[ruta_relativa] = {
            "size_bytes": archivo.stat().st_size,
            "sha256": calcular_sha256(archivo),
        }

    return inventario


def crear_manifiesto() -> None:
    """Registra el estado inicial de los datos originales."""

    if MANIFEST_PATH.exists():
        raise FileExistsError("El manifiesto ya existe. No se sobrescribirá.")

    contenido = {
        "schema_version": 1,
        "files": inventariar_archivos(),
    }

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    MANIFEST_PATH.write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print("Manifiesto creado:", MANIFEST_PATH)
    print("Archivos registrados:", len(contenido["files"]))


def verificar_manifiesto() -> None:
    """Compara los archivos actuales con el manifiesto."""

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError("No existe el manifiesto.")

    contenido = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    esperado = contenido["files"]
    actual = inventariar_archivos()

    faltantes = sorted(set(esperado) - set(actual))
    adicionales = sorted(set(actual) - set(esperado))

    modificados = sorted(
        ruta for ruta in set(esperado) & set(actual) if esperado[ruta] != actual[ruta]
    )

    print("\nVERIFICACIÓN DE INTEGRIDAD")
    print("-------------------------")
    print("Archivos esperados:", len(esperado))
    print("Archivos actuales:", len(actual))
    print("Faltantes:", len(faltantes))
    print("Adicionales:", len(adicionales))
    print("Modificados:", len(modificados))

    if faltantes or adicionales or modificados:
        print("\nFaltantes:", faltantes)
        print("Adicionales:", adicionales)
        print("Modificados:", modificados)
        raise SystemExit("ERROR: integridad no verificada.")

    print("\nINTEGRIDAD VERIFICADA CORRECTAMENTE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestión del manifiesto de datos.")

    parser.add_argument("accion", choices=["crear", "verificar"])

    args = parser.parse_args()

    if args.accion == "crear":
        crear_manifiesto()

    elif args.accion == "verificar":
        verificar_manifiesto()
