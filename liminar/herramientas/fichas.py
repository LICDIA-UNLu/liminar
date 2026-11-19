#!/usr/bin/env python3
"""Fichas de los modelos para Hugging Face (README.md de cada repositorio), en español con un resumen en inglés.

Los datos fijos de cada modelo están en MODELOS. Los resultados de la evaluación entran desde un JSON opcional con la
forma que escribe liminar.evaluacion.resultados (un modelo, o {clave: resultados} con --todos), más los datos del
entrenamiento:

    {"dominio": {"base": {"nota": 0.78, "correctas": 7.7}, "liminar": {"nota": 1.35, "correctas": 48.7}},
     "mmmlu": {"base": 64.5, "liminar": 60.1},                       # exactitud (%) con respuesta redactada
     "rag": {"base": {"promedio": 4.92, "incorrectas": 2, "n": 60}, "liminar": {...}},
     "referencia": {"dominio": {...}, "rag": {...}},                 # gpt-4.1-mini, opcional
     "entrenamiento": {"horas_gpu": 69, "epocas_todo": 4, "nodos": 1, "congelado": "..."},  # congelado: opcional
     "limites": ["..."]}                                             # límites observados en ese modelo, opcional

Sin resultados, la ficha sale sin la sección de evaluación.

Uso:
    python -m liminar.herramientas.fichas --modelo qwen3.5-9b --version v2026.11 [--resultados r.json] --salida DIR
    python -m liminar.herramientas.fichas --todos --version v2026.11 --salida DIR
"""
import argparse
import json
import os

ORG = 'LICDIA-UNLu'
MODELOS = {
    'gemma-4-e4b': ('Liminar-Gemma-4-E4B', 'google/gemma-4-E4B-it', 'Gemma 4 E4B', 'completo'),
    'ministral-3-8b': ('Liminar-Ministral-3-8B', 'mistralai/Ministral-3-8B-Instruct-2512', 'Ministral 3 8B', 'completo'),
    'qwen3.5-9b': ('Liminar-Qwen3.5-9B', 'Qwen/Qwen3.5-9B', 'Qwen 3.5 9B', 'completo'),
    'gemma-4-12b': ('Liminar-Gemma-4-12B', 'google/gemma-4-12B-it', 'Gemma 4 12B', 'completo'),
    'mistral-small-3.2-24b': ('Liminar-Mistral-Small-3.2-24B', 'mistralai/Mistral-Small-3.2-24B-Instruct-2506',
                              'Mistral Small 3.2 24B', 'lora'),
    'qwen3.5-27b': ('Liminar-Qwen3.5-27B', 'Qwen/Qwen3.5-27B', 'Qwen 3.5 27B', 'lora'),
    'gemma-4-31b': ('Liminar-Gemma-4-31B', 'google/gemma-4-31B-it', 'Gemma 4 31B', 'lora'),
    'qwen3.5-122b-a10b': ('Liminar-Qwen3.5-122B-A10B', 'Qwen/Qwen3.5-122B-A10B', 'Qwen 3.5 122B-A10B', 'lora'),
}

# clase de transformers con la que carga cada modelo publicado (probada con la carpeta a publicar); por omisión,
# AutoModelForCausalLM (Gemma 4 E4B, multimodal, también carga así).
CLASE = {'ministral-3-8b': 'AutoModelForImageTextToText', 'mistral-small-3.2-24b': 'AutoModelForImageTextToText'}
# modelos de Mistral: el tokenizador del entrenamiento es el de mistral-common (tekken.json)
MISTRAL_COMMON = {'ministral-3-8b', 'mistral-small-3.2-24b'}
# partes no textuales del modelo base que el ajuste (solo con texto) deja igual (comprobado contra los pesos del base)
MULTIMODAL = {'ministral-3-8b': 'procesa imágenes', 'gemma-4-e4b': 'procesa imágenes y audio',
              'mistral-small-3.2-24b': 'procesa imágenes'}


def num(x, dec=1):
    return f'{x:.{dec}f}'.replace('.', ',')


def tabla_resultados(r):
    filas = []
    d = r.get('dominio') or {}
    g = lambda x, k: (x or {}).get(k)
    if d:
        filas.append(('Normativa sin documentos (300 preguntas): respuestas correctas',
                      g(d.get('base'), 'correctas'), g(d.get('liminar'), 'correctas'), lambda x: f'{num(x)} %'))
        filas.append(('Normativa sin documentos: nota media del juez (0 a 2)',
                      g(d.get('base'), 'nota'), g(d.get('liminar'), 'nota'), lambda x: num(x, 2)))
    q = r.get('rag') or {}
    if q:
        filas.append(('Consulta con documentos, RAG (60 consultas): promedio del juez (1 a 5)',
                      g(q.get('base'), 'promedio'), g(q.get('liminar'), 'promedio'), lambda x: num(x, 2)))
        n = g(q.get('liminar'), 'n') or g(q.get('base'), 'n') or 60
        filas.append(('RAG: respuestas con algún dato incorrecto', g(q.get('base'), 'incorrectas'),
                      g(q.get('liminar'), 'incorrectas'), lambda x: f'{x} de {n}'))
    m = r.get('mmmlu') or {}
    if m:
        filas.append(('MMMLU ES-LA (1.026 preguntas, respuesta redactada): exactitud', m.get('base'), m.get('liminar'),
                      lambda x: f'{num(x)} %'))
    if not filas:
        return ''
    txt = ['| Evaluación | Modelo base | Liminar |', '|---|---|---|']
    txt += [f'| {e} | {"-" if b is None else f(b)} | {"-" if l is None else f(l)} |' for e, b, l, f in filas]
    ref = r.get('referencia') or {}
    rd, rr = g(ref, 'dominio'), g(ref, 'rag')
    if rd and rr:
        txt += ['', f'Como referencia, gpt-4.1-mini responde bien el {num(rd["correctas"])} % de las preguntas de normativa '
                    f'sin documentos (nota media {num(rd["nota"], 2)}) y promedia {num(rr["promedio"], 2)} en la consulta '
                    f'con documentos, con {rr["incorrectas"]} respuestas incorrectas de {rr["n"]}.']
    return '\n'.join(txt)


TIPOS_GGUF = {'Q8_0': '8 bits', 'Q6_K': '6 bits', 'Q5_K_M': '5 bits', 'Q4_K_M': '4 bits'}


def seccion_gguf(clave, nombre, archivos, nota=None):
    """Versiones cuantizadas publicadas en el mismo repositorio: [{"tipo": "Q4_K_M", "gb": 5.2}, ...]. Con "partes": N el
    archivo va en N piezas de llama-gguf-split, porque Hugging Face no acepta por LFS archivos de más de 50 GB."""
    if not archivos:
        return ''

    def archivo(x):
        n = x.get('partes', 1)
        return f'`{nombre}-{x["tipo"]}.gguf`' if n == 1 else f'`{nombre}-{x["tipo"]}-*-of-{n:05d}.gguf` ({n} piezas)'
    filas = '\n'.join(f'| {archivo(x)} | {x["tipo"]} ({TIPOS_GGUF.get(x["tipo"], "")}) | {num(x["gb"])} GB |'
                      for x in archivos)
    menor = min(archivos, key=lambda x: x['gb'])['tipo']
    razonamiento = ('' if clave in MISTRAL_COMMON else
                    ' La plantilla de chat de estos archivos deja apagado el razonamiento, como en el entrenamiento.')
    partido = any(x.get('partes', 1) > 1 for x in archivos)
    usos = 'llama.cpp o LM Studio' if partido else 'llama.cpp, Ollama o LM Studio'
    comandos = (f'llama-server -hf {ORG}/{nombre}:{menor}' if partido else
                f'ollama run hf.co/{ORG}/{nombre}:{menor}\nllama-server -hf {ORG}/{nombre}:{menor}')
    piezas = ('\n\nLos archivos de más de 50 GB van en piezas: llama.cpp y LM Studio las cargan a partir de la primera. '
              'Ollama no carga archivos en piezas; para usarlo, hay que unirlas antes con `llama-gguf-split --merge`.'
              if partido else '')
    extra = f'\n\n{nota}' if nota else ''
    return f"""
## Versiones cuantizadas (GGUF)

Para usarlo con {usos}, también sin GPU.{razonamiento}

| Archivo | Cuantización | Tamaño |
|---|---|---|
{filas}

```bash
{comandos}
```

La evaluación de esta ficha es la del modelo en bf16: las versiones cuantizadas no se evaluaron aparte.{piezas}{extra}
"""


def ficha(clave, version, resultados=None):
    nombre, base, base_txt, modo = MODELOS[clave]
    r = resultados or {}
    ajuste = ('Ajuste completo de todos los pesos' if modo == 'completo'
              else 'LoRA de rango 256 sobre las capas lineales, fusionado con el modelo base')
    e = r.get('entrenamiento') or {}
    if e.get('congelado'):   # p. ej. la tabla de embeddings por capa de Gemma 4 E4B
        ajuste = f'Ajuste completo de todos los pesos salvo {e["congelado"]}, que quedó fija'
    nodos = 'dos nodos' if e.get('nodos', 1) == 2 else 'un nodo'
    tiles = f'{8 * e.get("nodos", 1)} tiles'
    epocas = {1: 'una época', 2: 'dos épocas', 3: 'tres épocas', 4: 'cuatro épocas'}[e.get('epocas_todo', 4)]
    maestro = ('' if clave == 'qwen3.5-27b' else
               ' Las respuestas de entrenamiento las escribió Qwen 3.5 27B, así que el modelo aprende en parte su estilo.')
    horas = (f' El entrenamiento usó {e["horas_gpu"]:,} horas de GPU (horas de tile).'.replace(',', '.')
             if e.get('horas_gpu') else '')
    if e.get('nota'):   # p. ej. qué partes no cubrió el LoRA (los expertos enrutados del 122B)
        horas += f' {e["nota"]}'
    clase = CLASE.get(clave, 'AutoModelForCausalLM')
    instalar = ('Requiere `transformers` 5 o posterior'
                + (' y `mistral-common`, que provee el tokenizador con el que se entrenó el modelo:\n\n'
                   '```bash\npip install "transformers>=5" mistral-common\n```\n' if clave in MISTRAL_COMMON else '.\n'))
    imagenes = (f'\n- El ajuste usó solo texto: la parte que {MULTIMODAL[clave]} es la del modelo base, sin cambios.'
                if clave in MULTIMODAL else '')
    m = r.get('mmmlu') or {}
    if m.get('base') is not None and m.get('liminar') is not None and m['liminar'] < m['base'] - 2:
        imagenes += (f'\n- Rinde menos que el modelo base en conocimiento general ({num(m["liminar"])} % contra '
                     f'{num(m["base"])} % en MMMLU ES-LA): para tareas ajenas al sistema universitario conviene el modelo base.')
    imagenes += ''.join(f'\n- {x}' for x in r.get('limites', []))
    evaluacion = tabla_resultados(r)
    sobre_mmmlu = ('\nMMMLU ES-LA mide conocimiento general: el modelo redacta la respuesta y se lee la opción elegida.'
                   if r.get('mmmlu') else '')
    nota_uso = ('\nEl ajuste se hizo sin el modo de razonamiento de Qwen 3.5, y la plantilla de chat lo deja apagado por '
                'omisión.\n' if clave.startswith('qwen') else '')
    sec_gguf = seccion_gguf(clave, nombre, r.get('gguf') or [], r.get('gguf_nota'))
    sec_eval = ('' if not evaluacion else f"""
## Evaluación

Cada modelo se compara con su modelo base. Las respuestas abiertas las califica Llama 3.3 70B, que no pertenece a
ninguna de las familias evaluadas. El benchmark de normativa pregunta sin documentos por normas del corpus y mide
cuánto recuerda el modelo de lo que aprendió. La consulta con documentos usa los fragmentos que devuelve un sistema de
recuperación real y la rúbrica con la que se evaluó con personas el asistente de consulta del digesto de la UNLu
([licdia.unlu.edu.ar/rag-unlu](https://licdia.unlu.edu.ar/rag-unlu)): utilidad, fidelidad, completitud y claridad.{sobre_mmmlu}

{evaluacion}
""")
    return f"""---
license: apache-2.0
language:
- es
base_model: {base}
pipeline_tag: text-generation
library_name: transformers
tags:
- liminar
- argentina
- universidades
- normativa
---

<p align="center">
  <img src="https://raw.githubusercontent.com/LICDIA-UNLu/liminar/main/docs/assets/liminar-icono-512.png" width="96" alt="Liminar">
</p>

# {nombre}

{nombre} es un modelo de la familia **Liminar**: modelos de lenguaje abiertos del sistema universitario argentino,
adaptados a su normativa, sus instituciones y su vida académica, y al español formal de la Argentina. Parte de
[{base_txt}](https://huggingface.co/{base}) y lo desarrolla el Laboratorio de Investigación en Ciencia de Datos &
Inteligencia Artificial ([LICDIA](https://licdia.unlu.edu.ar)) de la Universidad Nacional de Luján.

- **Versión:** `{version}` (cada versión es una etiqueta de este repositorio).
- **Licencia:** Apache 2.0, la misma del modelo base.
- **Código y metodología:** [github.com/LICDIA-UNLu/liminar](https://github.com/LICDIA-UNLu/liminar).
- **Documentación:** [licdia-unlu.github.io/liminar](https://licdia-unlu.github.io/liminar).

## Uso

{instalar}
```python
from transformers import {clase}, AutoTokenizer

repo = "{ORG}/{nombre}"
tok = AutoTokenizer.from_pretrained(repo, revision="{version}")
modelo = {clase}.from_pretrained(
    repo, revision="{version}", dtype="auto", device_map="auto")

mensajes = [{{"role": "user",
             "content": "¿Qué establece la Ley de Educación Superior sobre la autonomía universitaria?"}}]
entrada = tok.apply_chat_template(
    mensajes, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(modelo.device)
salida = modelo.generate(**entrada, max_new_tokens=400)
print(tok.decode(salida[0, entrada["input_ids"].shape[1]:], skip_special_tokens=True))
```
{nota_uso}
Para consultas sobre normas concretas conviene usarlo con recuperación de documentos (RAG) sobre una fuente
actualizada: el modelo no sabe qué normas siguen vigentes.
{sec_gguf}
## Datos

- Normativa universitaria nacional de Infoleg (base de datos.jus.gob.ar, CC BY 4.0), curada a mano.
- Textos públicos sobre la vida universitaria: argentina.gob.ar (CC BY 4.0), CONICET (CC BY 2.5 AR) y el Consejo
  Interuniversitario Nacional (con su permiso).
- Datos generados a partir de ese corpus con el documento a la vista: fragmentos con su ficha, versiones reescritas,
  preguntas y respuestas, abstención, identidad, respuestas del propio modelo base a consignas generales y ejemplos de
  consulta con documentos.{maestro}

## Entrenamiento

{ajuste}, con FSDP en {nodos} de la supercomputadora Clementina XXI ({tiles} de GPU Intel Data
Center Max 1550): una época de preguntas y respuestas y {epocas} con todos los datos.{horas}
{sec_eval}
## Límites

- No sabe qué normas siguen vigentes ni cuáles cambiaron después de la fecha del corpus.
- Puede equivocar números de normas, fechas y montos.
- No conoce los actos internos de cada universidad: esa es la capa que cada institución puede sumar con la metodología.{imagenes}

## In English

{nombre} belongs to **Liminar**, a family of open language models from the Argentine university system, adapted to
its regulations, institutions and academic life and to formal Argentine Spanish. It is based on {base_txt} and
developed by LICDIA at Universidad Nacional de Luján. Code, recipe and documentation:
[github.com/LICDIA-UNLu/liminar](https://github.com/LICDIA-UNLu/liminar).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--modelo', choices=sorted(MODELOS))
    ap.add_argument('--todos', action='store_true')
    ap.add_argument('--version', required=True)
    ap.add_argument('--resultados', help='JSON con los resultados de un modelo (o, con --todos, {clave: resultados})')
    ap.add_argument('--salida', required=True)
    a = ap.parse_args()
    res = json.load(open(a.resultados, encoding='utf-8')) if a.resultados else {}
    claves = sorted(MODELOS) if a.todos else [a.modelo]
    for c in claves:
        d = os.path.join(a.salida, MODELOS[c][0]) if a.todos else a.salida
        os.makedirs(d, exist_ok=True)
        r = res.get(c, {}) if a.todos else res
        open(os.path.join(d, 'README.md'), 'w', encoding='utf-8').write(ficha(c, a.version, r))
        print(f'{MODELOS[c][0]}: {os.path.join(d, "README.md")}')


if __name__ == '__main__':
    main()
