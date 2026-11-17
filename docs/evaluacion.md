# Evaluación

Cada modelo Liminar se compara con su modelo base, en dos niveles. Las respuestas abiertas las califica un juez que no
pertenece a ninguna de las familias evaluadas (Llama 3.3 70B), siempre el mismo.

## Nivel 1: conocimiento

| Conjunto | Qué mide | Cómo |
|---|---|---|
| Benchmark de normativa (300 preguntas) | conocimiento del dominio sin documentos | el juez compara cada respuesta con la de referencia: 0, 1 o 2 |
| MMMLU ES-LA (1.026 preguntas, 18 por materia) | conocimiento general | el modelo redacta la respuesta y se lee la opción elegida; si no se reconoce una opción, cuenta como error |

Las preguntas del benchmark de normativa no entraron al entrenamiento, pero los documentos de los que salen sí (ver
[Partición](datos.md#particion)). MMMLU ES-LA es la traducción profesional
de MMLU al español de Latinoamérica publicada por OpenAI.

## Nivel 2: funcionalidad

**RAG.** Una muestra de 60 consultas con los fragmentos que devuelve un sistema de recuperación real. El juez califica
utilidad, fidelidad, completitud y claridad (1 a 5) y marca las respuestas incorrectas; además se miden el anclaje (que
todo acto citado esté en el contexto) y si la respuesta cita el acto de origen.

## Resultados

Cada modelo Liminar va debajo de su modelo base, y gpt-4.1-mini queda como referencia (sin MMMLU). La tabla suma
cada modelo cuando tiene las tres evaluaciones completas.

| Modelo | Normativa sin documentos: correctas | Nota del juez (0 a 2) | RAG: promedio (1 a 5) | RAG: incorrectas (de 60) | MMMLU ES-LA |
|---|---|---|---|---|---|
| Gemma 4 E4B (base) | 5,7 % | 0,37 | 4,89 | 0 | 64,3 % |
| Liminar-Gemma-4-E4B | 31,7 % | 1,10 | 5,00 | 0 | 60,6 % |
| Ministral 3 8B (base) | 7,7 % | 0,78 | 4,92 | 2 | 64,5 % |
| Liminar-Ministral-3-8B | 48,7 % | 1,35 | 4,90 | 1 | 48,8 % |
| gpt-4.1-mini (referencia) | 14,3 % | 0,77 | 4,98 | 0 | - |
