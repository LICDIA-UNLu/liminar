#!/usr/bin/env python3
"""Sube un modelo Liminar a Hugging Face: crea el repositorio de la organización si no existe, sube la carpeta del modelo
con su ficha como README.md y marca la versión con una etiqueta. Sin --confirmar solo muestra lo que haría.

El token sale del entorno (HF_TOKEN) o de la sesión guardada con `hf auth login`; nunca se pasa por argumento. Los
archivos grandes se suben con upload_large_folder, que retoma si se corta.

Uso:
    python -m liminar.herramientas.publicar --carpeta DIR --repo LICDIA-UNLu/Liminar-Qwen3.5-9B --version v2026.11 \\
        --ficha fichas/Liminar-Qwen3.5-9B/README.md [--privado] [--confirmar]
"""
import argparse
import fnmatch
import os
import re

# lo que no se publica: la ficha va aparte, y lo demás es propio del entrenamiento o de la máquina
EXCLUIR = ['README.md', '.*', '*/.*', 'punto*', 'optimizador*', 'estado_entrenamiento*', '*.pt', '*.log']


def archivos(carpeta):
    """(ruta relativa, bytes) de lo que se sube."""
    out = []
    for raiz, dirs, fs in os.walk(carpeta):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for f in fs:
            rel = os.path.relpath(os.path.join(raiz, f), carpeta)
            if not any(fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(f, p) for p in EXCLUIR):
                out.append((rel, os.path.getsize(os.path.join(raiz, f))))
    return sorted(out)


def revisar(a, fs):
    """Errores que impiden publicar."""
    errores = []
    nombres = {r for r, _ in fs}
    adaptador = 'adapter_config.json' in nombres
    if not adaptador and 'config.json' not in nombres:
        errores.append('la carpeta no tiene config.json ni adapter_config.json')
    if not any(r.endswith('.safetensors') for r in nombres):
        errores.append('la carpeta no tiene pesos .safetensors')
    if not adaptador and not ({'tokenizer.json', 'tokenizer.model', 'tekken.json'} & nombres):
        errores.append('la carpeta no tiene tokenizador')
    if not re.fullmatch(r'v\d{4}\.\d{2}', a.version):
        errores.append(f'la versión {a.version} no tiene la forma vAAAA.MM')
    if not os.path.exists(a.ficha):
        errores.append(f'no está la ficha {a.ficha}')
    else:
        texto = open(a.ficha, encoding='utf-8').read()
        if not texto.startswith('---'):
            errores.append('la ficha no empieza con los metadatos YAML (---)')
        if f'`{a.version}`' not in texto:
            errores.append(f'la ficha no menciona la versión {a.version}')
    return errores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--carpeta', required=True, help='modelo en formato Hugging Face (o adaptador LoRA)')
    ap.add_argument('--repo', required=True, help='p. ej. LICDIA-UNLu/Liminar-Qwen3.5-9B')
    ap.add_argument('--version', required=True, help='etiqueta de la versión, p. ej. v2026.11')
    ap.add_argument('--ficha', required=True, help='README.md de la ficha (liminar.herramientas.fichas)')
    ap.add_argument('--privado', action='store_true', help='crear el repositorio como privado')
    ap.add_argument('--sin-etiqueta', action='store_true',
                    help='subir sin marcar la versión (para revisar la ficha antes de fijarla; se etiqueta después)')
    ap.add_argument('--confirmar', action='store_true', help='subir de verdad; sin esto solo se muestra el plan')
    a = ap.parse_args()

    fs = archivos(a.carpeta)
    total = sum(b for _, b in fs)
    print(f'repositorio {a.repo} ({"privado" if a.privado else "público"}), versión {a.version}')
    print(f'ficha {a.ficha} → README.md')
    for r, b in fs:
        print(f'  {r:48s} {b / 2**30:8.2f} GB')
    print(f'  {"total":48s} {total / 2**30:8.2f} GB en {len(fs)} archivos')
    errores = revisar(a, fs)
    for e in errores:
        print(f'ERROR: {e}')
    if errores:
        raise SystemExit(1)
    if not a.confirmar:
        print('sin --confirmar: no se subió nada')
        return

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, repo_type='model', private=a.privado, exist_ok=True)
    etiquetas = {t.name for t in api.list_repo_refs(a.repo).tags}
    if a.version in etiquetas:
        raise SystemExit(f'{a.repo} ya tiene la etiqueta {a.version}: una versión publicada no se pisa')
    api.upload_large_folder(repo_id=a.repo, folder_path=a.carpeta, repo_type='model',
                            allow_patterns=[r for r, _ in fs])
    api.upload_file(path_or_fileobj=a.ficha, path_in_repo='README.md', repo_id=a.repo,
                    commit_message=f'Ficha de {a.version}')
    if a.sin_etiqueta:
        print(f'LISTO sin etiqueta: https://huggingface.co/{a.repo} (falta create_tag {a.version})')
        return
    api.create_tag(a.repo, tag=a.version, tag_message=f'Liminar {a.version}')
    print(f'LISTO https://huggingface.co/{a.repo}/tree/{a.version}')


if __name__ == '__main__':
    main()
