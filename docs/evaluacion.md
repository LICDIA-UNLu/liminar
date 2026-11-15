# Evaluación

Cada modelo Liminar se compara con su modelo base, en dos niveles. Las respuestas abiertas las califica un juez que no
pertenece a ninguna de las familias evaluadas (Llama 3.3 70B), siempre el mismo.

## Nivel 1: conocimiento

| Conjunto | Qué mide | Cómo |
|---|---|---|
| Benchmark de normativa (300 preguntas) | conocimiento del dominio sin documentos | el juez compara cada respuesta con la de referencia: 0, 1 o 2 |
| MMMLU ES-LA (1.026 preguntas, 18 por materia) | conocimiento general | exactitud por la probabilidad de cada letra |
| m-ArenaHard ES (200 consignas) | capacidad general en consignas abiertas | el juez compara la respuesta ajustada con la del base, en los dos órdenes |

El benchmark de normativa sale de documentos que no entraron al entrenamiento. MMMLU ES-LA es la traducción profesional
de MMLU al español de Latinoamérica publicada por OpenAI; m-ArenaHard es la versión multilingüe de Arena-Hard publicada
por Cohere.

## Nivel 2: funcionalidad

**RAG.** Una muestra de 60 consultas con los fragmentos que devuelve un sistema de recuperación real. El juez califica
utilidad, fidelidad, completitud y claridad (1 a 5) y marca las respuestas incorrectas; además se miden el anclaje (que
todo acto citado esté en el contexto) y si la respuesta cita el acto de origen.
