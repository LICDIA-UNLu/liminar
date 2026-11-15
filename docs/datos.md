# Datos de entrenamiento

Los datos se generaron a partir del corpus con un modelo maestro abierto (Qwen 3.5 27B), siempre con el documento a la
vista.

| Tipo | Qué enseña |
|---|---|
| Fragmentos con su ficha | el texto de cada norma, con número, fecha y órgano emisor al comienzo |
| Versiones reescritas | el mismo contenido con otra redacción, para fijar el conocimiento y no la forma |
| Preguntas y respuestas | consultas sobre cada documento, por tema y por identificador del acto |
| Abstención | reconocer cuándo la pregunta no tiene respuesta en lo que el modelo sabe |
| Identidad | qué es Liminar y quién lo desarrolla |
| Respuestas del propio modelo base | consignas generales respondidas por el modelo antes del ajuste, para que no pierda capacidades generales |
| Tarea RAG | consultas con fragmentos recuperados y respuestas que citan los actos del contexto |

## Tandas de versiones

Las versiones se generan por tandas. A partir de la segunda, el maestro ve lo que escribió antes y se le pide otra
redacción y otras preguntas; las casi copias de la tanda anterior (ROUGE-L de 0,7 o más, el umbral de novedad de
Self-Instruct, Wang et al., 2023) no entran al entrenamiento.

## Tarea RAG

Los ejemplos RAG usan la consulta con los fragmentos que devuelve un sistema de recuperación real y una respuesta que
cita solo actos presentes en el contexto. Se agregan porque ajustar solo con preguntas sin documentos puede empeorar el
uso de lo recuperado (RAFT, Zhang et al., 2024). Los documentos de la evaluación no aparecen en estos ejemplos.

## Partición

Las 300 preguntas del benchmark de normativa quedan afuera del entrenamiento y forman la validación; también se saca
cualquier conversación de entrenamiento que repita exactamente una de esas preguntas. Los documentos de los que salen sí
entran al entrenamiento: el benchmark mide si el modelo responde, sin documentos a la vista, sobre lo que aprendió del
corpus.
