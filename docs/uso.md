# Uso

## Con transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "LICDIA-UNLu/Liminar-Qwen3.5-9B"
tok = AutoTokenizer.from_pretrained(repo, revision="v2026.11")
modelo = AutoModelForCausalLM.from_pretrained(repo, revision="v2026.11", dtype="auto", device_map="auto")

mensajes = [{"role": "user", "content": "¿Qué órganos de gobierno prevé la Ley de Educación Superior?"}]
entrada = tok.apply_chat_template(mensajes, add_generation_prompt=True, return_tensors="pt").to(modelo.device)
salida = modelo.generate(entrada, max_new_tokens=400)
print(tok.decode(salida[0, entrada.shape[1]:], skip_special_tokens=True))
```

En los modelos Qwen 3.5 conviene desactivar el modo de razonamiento con `enable_thinking=False` en
`apply_chat_template`.

## Con recuperación de documentos (RAG)

Para responder sobre normas concretas, el modelo rinde mejor con los fragmentos pertinentes en el contexto. El formato
que se usó en el entrenamiento es una instrucción de sistema que pide responder solo con el contexto y citar los actos,
seguida de un mensaje con el contexto y la pregunta:

```text
CONTEXTO:

<fragmentos recuperados, cada uno con la identificación del acto>

---

PREGUNTA: <consulta>
```
