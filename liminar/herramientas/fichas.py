#!/usr/bin/env python3
"""Fichas de los modelos para Hugging Face (README.md de cada repositorio), en español con un resumen en inglés.

Los datos fijos de cada modelo están en MODELOS. Los resultados de la evaluación entran desde un JSON opcional con la
forma:

    {"dominio": {"base": 1.12, "liminar": 1.48},            # nota media del juez (0 a 2) en el benchmark de normativa
     "mmmlu": {"base": 61.2, "liminar": 60.8},              # exactitud (%) en MMMLU ES-LA
     "arena": {"liminar": 47.5},                             # tasa del ajustado contra su base en m-ArenaHard ES (%)
     "rag": {"base": 4.98, "liminar": 4.70},                 # utilidad media del juez (1 a 5) en la muestra RAG
     "horas_gpu": 118}                                       # horas de GPU (tiles) del entrenamiento

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


def tabla_resultados(r):
    filas = []
    if 'dominio' in r:
        filas.append(('Normativa universitaria (300 preguntas, nota del juez de 0 a 2)',
                      r['dominio'].get('base'), r['dominio'].get('liminar')))
    if 'mmmlu' in r:
        filas.append(('MMMLU ES-LA (1.026 preguntas, exactitud %)', r['mmmlu'].get('base'), r['mmmlu'].get('liminar')))
    if 'rag' in r:
        filas.append(('Consulta con documentos, RAG (60 consultas, utilidad de 1 a 5)', r['rag'].get('base'),
                      r['rag'].get('liminar')))
    if not filas and 'arena' not in r:
        return ''
    fmt = lambda x: '-' if x is None else (f'{x:.2f}' if x < 10 else f'{x:.1f}').replace('.', ',')
    txt = ['| Evaluación | Modelo base | Liminar |', '|---|---|---|']
    txt += [f'| {n} | {fmt(b)} | {fmt(l)} |' for n, b, l in filas]
    if 'arena' in r and r['arena'].get('liminar') is not None:
        txt.append('')
        txt.append(f'En m-ArenaHard ES (200 consignas abiertas), el juez prefirió la respuesta de Liminar frente a la del '
                   f'modelo base en el {str(round(r["arena"]["liminar"], 1)).replace(".", ",")} % de las comparaciones (los empates cuentan como medio).')
    return '\n'.join(txt)


def ficha(clave, version, resultados=None):
    nombre, base, base_txt, modo = MODELOS[clave]
    r = resultados or {}
    ajuste = ('Ajuste completo de todos los pesos' if modo == 'completo'
              else 'LoRA de rango 256 sobre las capas lineales, fusionado con el modelo base')
    nodos = ('dos nodos' if clave == 'qwen3.5-122b-a10b' else 'un nodo')
    tiles = ('16 tiles' if clave == 'qwen3.5-122b-a10b' else '8 tiles')
    maestro = ('' if clave == 'qwen3.5-27b' else
               ' Las respuestas de entrenamiento las escribió Qwen 3.5 27B, así que el modelo aprende en parte su estilo.')
    horas = f' El entrenamiento usó {r["horas_gpu"]:,} horas de GPU (tiles).'.replace(',', '.') if r.get('horas_gpu') else ''
    evaluacion = tabla_resultados(r)
    sec_eval = ('' if not evaluacion else f"""
## Evaluación

Cada modelo se compara con su modelo base. Las respuestas abiertas las califica Llama 3.3 70B, que no pertenece a
ninguna de las familias evaluadas.

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
- **Código y receta:** [github.com/LICDIA-UNLu/liminar](https://github.com/LICDIA-UNLu/liminar).
- **Documentación:** [licdia-unlu.github.io/liminar](https://licdia-unlu.github.io/liminar).

## Uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "{ORG}/{nombre}"
tok = AutoTokenizer.from_pretrained(repo, revision="{version}")
modelo = AutoModelForCausalLM.from_pretrained(
    repo, revision="{version}", dtype="auto", device_map="auto")

mensajes = [{{"role": "user",
             "content": "¿Qué establece la Ley de Educación Superior sobre la autonomía universitaria?"}}]
entrada = tok.apply_chat_template(
    mensajes, add_generation_prompt=True, return_tensors="pt").to(modelo.device)
salida = modelo.generate(entrada, max_new_tokens=400)
print(tok.decode(salida[0, entrada.shape[1]:], skip_special_tokens=True))
```

Para consultas sobre normas concretas conviene usarlo con recuperación de documentos (RAG) sobre una fuente
actualizada: el modelo no sabe qué normas siguen vigentes.

## Datos

- Normativa universitaria nacional de Infoleg (base de datos.jus.gob.ar, CC BY 4.0), curada a mano.
- Textos públicos sobre la vida universitaria: argentina.gob.ar (CC BY 4.0), CONICET (CC BY 2.5 AR) y el Consejo
  Interuniversitario Nacional (con su permiso).
- Datos generados a partir de ese corpus con el documento a la vista: fragmentos con su ficha, versiones reescritas,
  preguntas y respuestas, abstención, identidad, respuestas del propio modelo base a consignas generales y ejemplos de
  consulta con documentos.{maestro}

## Entrenamiento

{ajuste}, con FSDP en {nodos} de la supercomputadora Clementina XXI ({tiles} de GPU Intel Data
Center Max 1550): una época de preguntas y respuestas y cuatro épocas con todos los datos.{horas}
{sec_eval}
## Límites

- No sabe qué normas siguen vigentes ni cuáles cambiaron después de la fecha del corpus.
- Puede equivocar números de normas, fechas y montos.
- No conoce los actos internos de cada universidad: esa es la capa que cada institución puede sumar con la receta.

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
