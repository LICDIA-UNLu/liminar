# Liminar

Liminar es una familia de modelos de lenguaje abiertos del sistema universitario argentino. Están adaptados a su
normativa, sus instituciones y su vida académica, y al español formal de la Argentina, y se publican con licencia
abierta para que cualquier persona o institución pueda usarlos. La desarrolla el Laboratorio de Investigación en
Ciencia de Datos & Inteligencia Artificial (LICDIA) de la Universidad Nacional de Luján.

El nombre remite al Manifiesto Liminar de la Reforma Universitaria de 1918.

## Dos capas

- **Capa nacional (los modelos de este repositorio):** cada modelo parte de un modelo abierto y se ajusta con la
  normativa universitaria nacional curada y con textos públicos sobre la vida universitaria (noticias y páginas
  institucionales), para que conozca el sistema universitario argentino y responda en registro formal argentino.
- **Capa de cada universidad:** la receta de este repositorio permite que cada institución ajuste un adaptador LoRA con
  sus propios actos (su digesto). Esa capa queda en la universidad y no se publica.

## Modelos

| Modelo | Base | Ajuste |
|---|---|---|
| [Liminar-Gemma-4-E4B](https://huggingface.co/LICDIA-UNLu/Liminar-Gemma-4-E4B) | Gemma 4 E4B | completo |
| [Liminar-Ministral-3-8B](https://huggingface.co/LICDIA-UNLu/Liminar-Ministral-3-8B) | Ministral 3 8B | completo |
| [Liminar-Qwen3.5-9B](https://huggingface.co/LICDIA-UNLu/Liminar-Qwen3.5-9B) | Qwen 3.5 9B | completo |
| [Liminar-Gemma-4-12B](https://huggingface.co/LICDIA-UNLu/Liminar-Gemma-4-12B) | Gemma 4 12B | completo |
| [Liminar-Mistral-Small-3.2-24B](https://huggingface.co/LICDIA-UNLu/Liminar-Mistral-Small-3.2-24B) | Mistral Small 3.2 24B | LoRA |
| [Liminar-Qwen3.5-27B](https://huggingface.co/LICDIA-UNLu/Liminar-Qwen3.5-27B) | Qwen 3.5 27B | LoRA |
| [Liminar-Gemma-4-31B](https://huggingface.co/LICDIA-UNLu/Liminar-Gemma-4-31B) | Gemma 4 31B | LoRA |
| [Liminar-Qwen3.5-122B-A10B](https://huggingface.co/LICDIA-UNLu/Liminar-Qwen3.5-122B-A10B) | Qwen 3.5 122B-A10B (MoE) | LoRA |

Todas las bases tienen licencia Apache 2.0. Las versiones se publican como etiquetas de cada repositorio, con el formato
`vAAAA.MM` (por ejemplo `v2026.11`); la rama `main` apunta a la versión recomendada.

## Uso

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "LICDIA-UNLu/Liminar-Qwen3.5-9B"
tok = AutoTokenizer.from_pretrained(repo, revision="v2026.11")
modelo = AutoModelForCausalLM.from_pretrained(repo, revision="v2026.11", dtype="auto", device_map="auto")

mensajes = [{"role": "user", "content": "¿Qué establece la Ley de Educación Superior sobre la autonomía universitaria?"}]
entrada = tok.apply_chat_template(mensajes, add_generation_prompt=True, return_tensors="pt").to(modelo.device)
salida = modelo.generate(entrada, max_new_tokens=400)
print(tok.decode(salida[0, entrada.shape[1]:], skip_special_tokens=True))
```

El modelo no sabe qué normas siguen vigentes: para consultas sobre normativa conviene usarlo con recuperación de
documentos (RAG) sobre una fuente actualizada.

## Estructura

| Carpeta | Contenido |
|---|---|
| `liminar/corpus` | descarga y curación de la normativa de Infoleg, fuentes abiertas, OCR de anexos y armado de documentos y fragmentos |
| `liminar/aumentacion` | generación de datos con el modelo maestro: preguntas, versiones, alcance, replay y ejemplos RAG |
| `liminar/entrenamiento` | entrenamiento con FSDP (completo o LoRA) y la identidad de cada modelo |
| `liminar/evaluacion` | benchmark de normativa, MMMLU ES-LA, m-ArenaHard ES, juez RAG y resúmenes |
| `slurm` | scripts para SLURM: entrenamiento en uno o dos nodos, pilotos con cola de trabajos y juez |

Los scripts de `slurm` leen las rutas de variables de entorno: `LIMINAR_DATOS` (datos y salidas), `LIMINAR_MODELOS`
(modelos base), `LIMINAR_PY` (intérprete), `LIMINAR_JUEZ` (modelo juez) y, si hace falta, `LIMINAR_PARTICION` y
`LIMINAR_EXCLUIR`. La configuración de oneCCL que traen es la de las GPU Intel Max 1550.

## Documentación

La documentación completa (corpus, generación de datos, entrenamiento, evaluación y la receta para universidades) está
en <https://licdia-unlu.github.io/liminar>.

## Cómo citar

Ver [`CITATION.cff`](CITATION.cff).

## Licencia

El código de este repositorio se distribuye con licencia [Apache 2.0](LICENSE).

## In English

Liminar is a family of open language models from the Argentine university system, adapted to its regulations,
institutions and academic life and to formal Argentine Spanish, and openly released for anyone to use. It is developed
by LICDIA at Universidad Nacional de Luján. This repository contains the data pipeline, the training and
evaluation code, and a recipe for universities to train their own adapter on their own administrative acts.
