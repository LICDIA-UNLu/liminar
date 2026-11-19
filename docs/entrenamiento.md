# Entrenamiento

- **Ajuste completo** para los modelos de hasta 12B y **LoRA de rango 256** para los de 24B en adelante, todos con FSDP
  sobre los 8 tiles de un nodo. En el Qwen 3.5 122B-A10B el LoRA cubre la atención y el experto compartido de cada
  bloque; los expertos enrutados quedan como en el modelo base.
- **Datos:** los mismos para todos los modelos
  ([LICDIA-UNLu/liminar-entrenamiento](https://huggingface.co/datasets/LICDIA-UNLu/liminar-entrenamiento)). En cada
  época los pedidos generales entran cuatro veces y los ejemplos de la Ley de Educación Superior, dos.
- **Programa:** una época de solo preguntas y respuestas y después cuatro épocas con todos los datos (dos en el
  Qwen 3.5 122B-A10B). Se guarda un modelo por época.
- **Lotes por tokens:** lotes de hasta 2.048 tokens con relleno y acumulación de 2. Un ejemplo más largo forma su propio
  lote; esos lotes largos se agrupan en pasos enteros para que ningún tile espere a otro.
- **Pérdida** solo sobre la respuesta, con los logits calculados únicamente en esas posiciones.
- **Reanudación:** punto de reanudación periódico; si un proceso se cae, el entrenamiento retoma desde el último.

## Notas sobre las GPU Intel Max 1550

- La atención `sdpa` arma la matriz completa de L×L para entrenar (unos 14 GiB por capa con 6.000 tokens). En los lotes
  largos se usa `flex_attention`, compilada con Triton, que no la arma.
- Si un tile llega al límite de memoria, el asignador libera bloques que los demás procesos tienen mapeados para las
  comunicaciones y se produce un fallo de página: hay que dejar margen.
- En Gemma 4 E4B se congela la tabla de embeddings por capa, cuyo gradiente en fp32 no entra en memoria.
- Mistral Small 3.2 y Ministral 3 usan el tokenizador de `mistral-common`.
