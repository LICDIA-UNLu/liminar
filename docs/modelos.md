# Modelos

| Modelo | Modelo base | Ajuste |
|---|---|---|
| Liminar-Gemma-4-E4B | `google/gemma-4-E4B-it` | completo |
| Liminar-Ministral-3-8B | `mistralai/Ministral-3-8B-Instruct-2512` | completo |
| Liminar-Qwen3.5-9B | `Qwen/Qwen3.5-9B` | completo |
| Liminar-Gemma-4-12B | `google/gemma-4-12B-it` | completo |
| Liminar-Mistral-Small-3.2-24B | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | LoRA, rango 256 |
| Liminar-Qwen3.5-27B | `Qwen/Qwen3.5-27B` | LoRA, rango 256 |
| Liminar-Gemma-4-31B | `google/gemma-4-31B-it` | LoRA, rango 256 |
| Liminar-Qwen3.5-122B-A10B | `Qwen/Qwen3.5-122B-A10B` | LoRA, rango 256 |

Los modelos están en la organización [LICDIA-UNLu](https://huggingface.co/LICDIA-UNLu) de Hugging Face. Todas las
bases tienen licencia Apache 2.0 y los modelos Liminar mantienen esa licencia.

## Versiones

Cada modelo tiene un único repositorio y cada versión es una etiqueta con el formato `vAAAA.MM`, que indica el mes del
corpus. La rama `main` apunta a la versión recomendada. Para fijar una versión:

```python
AutoModelForCausalLM.from_pretrained("LICDIA-UNLu/Liminar-Qwen3.5-9B", revision="v2026.11")
```

## Límites

- El modelo no sabe qué normas siguen vigentes ni cuáles fueron modificadas después de la fecha del corpus.
- Puede equivocar números de normas, fechas y montos. Para consultas que requieren precisión, conviene usarlo con
  recuperación de documentos sobre una fuente actualizada (ver [Uso](uso.md)).
