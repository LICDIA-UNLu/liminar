<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/liminar-logotipo-oscuro.png">
    <img src="docs/assets/liminar-logotipo-claro.png" width="360" alt="Liminar">
  </picture>
</p>

<p align="center"><b>Modelos de lenguaje abiertos del sistema universitario argentino</b></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/licencia-Apache%202.0-1f4e8c" alt="Licencia Apache 2.0"></a>
  <a href="https://licdia-unlu.github.io/liminar"><img src="https://img.shields.io/badge/documentaci%C3%B3n-licdia--unlu.github.io-0e7490" alt="Documentación"></a>
  <a href="https://huggingface.co/LICDIA-UNLu"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20modelos-LICDIA--UNLu-f2b705" alt="Modelos en Hugging Face"></a>
  <a href="https://huggingface.co/datasets/LICDIA-UNLu/normativa-universitaria-argentina"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-normativa%20universitaria-f2b705" alt="Dataset en Hugging Face"></a>
  <a href="https://licdia.unlu.edu.ar"><img src="https://img.shields.io/badge/LICDIA-UNLu-16233f" alt="LICDIA - UNLu"></a>
</p>

Liminar es una familia de modelos de lenguaje abiertos del sistema universitario argentino. Están adaptados a su
normativa, sus instituciones y su vida académica, y al español formal de la Argentina, y se publican con licencia
abierta para que cualquier persona o institución pueda usarlos. La desarrolla el Laboratorio de Investigación en
Ciencia de Datos & Inteligencia Artificial (LICDIA) de la Universidad Nacional de Luján.

El nombre remite al Manifiesto Liminar de la Reforma Universitaria de 1918, publicado por la Federación Universitaria
de Córdoba. La Reforma impulsó la autonomía universitaria y el cogobierno, y Liminar retoma la idea de autonomía con
modelos abiertos que cada universidad puede ejecutar en su propia infraestructura y adaptar a sus necesidades.

## Dos capas

- **Capa nacional (los modelos de este repositorio):** cada modelo parte de un modelo abierto y se ajusta con la
  normativa universitaria nacional curada y con textos públicos sobre la vida universitaria (noticias y páginas
  institucionales), para que conozca el sistema universitario argentino y responda en registro formal argentino.
- **Capa de cada universidad:** la metodología de este repositorio permite que cada institución ajuste un adaptador LoRA con
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

## Resultados

Cada modelo Liminar va debajo de su modelo base, y gpt-4.1-mini queda como referencia. Qué mide cada columna y cómo
se califica está en [Evaluación](https://licdia-unlu.github.io/liminar/evaluacion/).

| Modelo | Normativa sin documentos: correctas | Nota del juez (0 a 2) | RAG: promedio (1 a 5) | RAG: incorrectas (de 60) |
|---|---|---|---|---|
| Gemma 4 E4B (base) | 5,7 % | 0,37 | 4,89 | 0 |
| Liminar-Gemma-4-E4B | 31,7 % | 1,10 | 5,00 | 0 |
| Ministral 3 8B (base) | 7,7 % | 0,78 | 4,92 | 2 |
| Liminar-Ministral-3-8B | 48,7 % | 1,35 | 4,90 | 1 |
| Qwen 3.5 9B (base) | 6,0 % | 0,61 | 4,99 | 0 |
| Liminar-Qwen3.5-9B | 36,7 % | 1,08 | 4,97 | 0 |
| Gemma 4 12B (base) | 8,7 % | 0,53 | 4,95 | 0 |
| Liminar-Gemma-4-12B | 38,7 % | 1,21 | 4,97 | 0 |
| Mistral Small 3.2 24B (base) | 9,7 % | 0,76 | 4,94 | 1 |
| Liminar-Mistral-Small-3.2-24B | 46,3 % | 1,26 | 4,92 | 0 |
| Qwen 3.5 27B (base) | 10,0 % | 0,63 | 4,99 | 0 |
| Liminar-Qwen3.5-27B | 46,7 % | 1,30 | 4,99 | 0 |
| Gemma 4 31B (base) | 13,3 % | 0,73 | 4,94 | 0 |
| Liminar-Gemma-4-31B | 44,7 % | 1,29 | 4,99 | 0 |
| Qwen 3.5 122B-A10B (base) | 8,3 % | 0,67 | 5,00 | 0 |
| Liminar-Qwen3.5-122B-A10B | 31,7 % | 1,06 | 4,99 | 0 |
| gpt-4.1-mini (referencia) | 14,3 % | 0,77 | 4,98 | 0 |
