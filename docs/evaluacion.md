# Evaluación

Cada modelo Liminar se compara con su modelo base, en dos niveles. Las respuestas abiertas las califica un juez que no
pertenece a ninguna de las familias evaluadas (Llama 3.3 70B), siempre el mismo.

## Nivel 1: conocimiento

| Conjunto | Qué mide | Cómo |
|---|---|---|
| Benchmark de normativa (300 preguntas) | conocimiento del dominio sin documentos | el juez compara cada respuesta con la de referencia: 0, 1 o 2 |

Las preguntas del benchmark de normativa no entraron al entrenamiento, pero los documentos de los que salen sí (ver
[Partición](datos.md#particion)).

## Nivel 2: funcionalidad

**RAG.** Una muestra de 60 consultas con los fragmentos que devuelve un sistema de recuperación real. El juez califica
utilidad, fidelidad, completitud y claridad (1 a 5) y marca las respuestas incorrectas; además se miden el anclaje (que
todo acto citado esté en el contexto) y si la respuesta cita el acto de origen.

## Resultados

Cada modelo Liminar va debajo de su modelo base, y gpt-4.1-mini queda como referencia. La tabla suma
cada modelo cuando tiene sus evaluaciones completas.

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
