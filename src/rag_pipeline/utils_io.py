import hashlib
import json
import os
from typing import Dict, Any


def verificar_tesseract() -> str | None:
    caminhos = [
        r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        r"C:\\Tesseract-OCR\\tesseract.exe",
    ]
    for caminho in caminhos:
        if os.path.exists(caminho):
            return caminho
    return None


def verificar_poppler() -> str | None:
    env_path = os.getenv("POPPLER_PATH")
    if env_path and os.path.isdir(env_path):
        return env_path

    candidatos = [
        r"C:\\Program Files\\poppler\\Library\\bin",
        r"C:\\Program Files\\poppler-24.08.0\\Library\\bin",
        r"C:\\Program Files\\poppler-24.07.0\\Library\\bin",
        r"C:\\Program Files\\poppler-24.02.0\\Library\\bin",
        r"C:\\poppler\\Library\\bin",
        r"C:\\poppler\\bin",
    ]
    for caminho in candidatos:
        if os.path.isdir(caminho):
            return caminho
    return None


def calcular_hash_arquivo(filepath: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def carregar_registro_processados(registro_path: str = "./arquivos_processados.json") -> Dict[str, Any]:
    if os.path.exists(registro_path):
        with open(registro_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def salvar_registro_processados(registro: Dict[str, Any], registro_path: str = "./arquivos_processados.json") -> None:
    with open(registro_path, "w", encoding="utf-8") as f:
        json.dump(registro, f, indent=2, ensure_ascii=False)

