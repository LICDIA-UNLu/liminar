"""Generación por lotes con un modelo maestro (Qwen 3.5 27B) y lectura de las citas de actos en un texto."""
import re

# "Resolución HCS 752/23", "DISPCD-CBLUJ: 563/23": código, número y año
RE_ACTO = re.compile(r'\b([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ\-]{1,14})\s*:?\s*(\d{1,5})\s*/\s*(\d{2,4})')


def clave_acto(codigo, numero, anio):
    anio = anio if len(anio) == 4 else '20' + anio
    return f'{codigo.upper()}|{int(numero)}|{anio}'


def actos_citados(texto):
    return {clave_acto(*m.groups()) for m in RE_ACTO.finditer(texto or '')}


def acto_de_cita(cita):
    """'Disposición DISSECPCII 64/2026 — Considerando' -> 'Disposición DISSECPCII 64/2026'."""
    return (cita or '').split(' — ')[0].strip()


class Maestro:
    def __init__(self, ruta, max_nuevos):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(ruta)
        self.tok.padding_side = 'left'
        self.modelo = AutoModelForCausalLM.from_pretrained(
            ruta, dtype=torch.bfloat16, device_map='balanced').eval()
        # Qwen 3.5 trae <|endoftext|> como eos en config.json: sin <|im_end|> en la lista la
        # generación no corta al terminar el turno.
        fin = {self.tok.eos_token_id, self.tok.convert_tokens_to_ids('<|im_end|>'),
               self.tok.convert_tokens_to_ids('<|endoftext|>')}
        self.fin = sorted(t for t in fin if isinstance(t, int) and t >= 0)
        self.max_nuevos = max_nuevos

    def texto(self, mensajes):
        return self.tok.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)

    def generar(self, lista_mensajes, max_nuevos=None, presupuesto=20000):
        """Greedy, como la temperatura 0 de producción. Devuelve (texto, terminó, tokens_prompt)
        en el orden pedido.

        Los prompts se agrupan por largo y cada grupo se limita a `presupuesto` tokens con
        relleno (largo máximo por cantidad). Lo que ocupa memoria es el prefill: sin los
        núcleos de fla, la atención lineal de Qwen 3.5 pasa la secuencia entera a float32, y
        un lote de 8 prompts de ~6.000 tokens agotó el tile. Ese agotamiento no siempre llega
        como OutOfMemoryError: a veces el runtime de Intel aborta con "Segmentation fault
        from GPU" y el proceso muere sin poder atraparlo. Por eso el límite va antes. Con
        28.000 todavía hubo agotamientos tras horas de corrida (fragmentación): va 20.000.
        """
        largos = [len(self.tok(self.texto(m), add_special_tokens=False)['input_ids'])
                  for m in lista_mensajes]
        orden = sorted(range(len(lista_mensajes)), key=lambda j: largos[j])
        grupos, actual = [], []
        for j in orden:
            if actual and largos[j] * (len(actual) + 1) > presupuesto:
                grupos.append(actual)
                actual = []
            actual.append(j)
        if actual:
            grupos.append(actual)
        res = [None] * len(lista_mensajes)
        for g in grupos:
            for j, r in zip(g, self._generar_lote([lista_mensajes[j] for j in g], max_nuevos)):
                res[j] = r
        return res

    def _generar_lote(self, lista_mensajes, max_nuevos):
        textos = [self.texto(m) for m in lista_mensajes]
        enc = self.tok(textos, return_tensors='pt', padding=True, add_special_tokens=False)
        disp = self.modelo.get_input_embeddings().weight.device
        enc = {k: v.to(disp) for k, v in enc.items()}
        with self.torch.no_grad():
            out = self.modelo.generate(**enc, max_new_tokens=max_nuevos or self.max_nuevos, do_sample=False,
                                       pad_token_id=self.tok.pad_token_id or self.fin[0],
                                       eos_token_id=self.fin)
        nuevos = out[:, enc['input_ids'].shape[1]:]
        res = []
        for fila, att in zip(nuevos, enc['attention_mask']):
            ids = fila.tolist()
            termino = any(t in self.fin for t in ids)
            res.append((self.tok.decode(ids, skip_special_tokens=True).strip(), termino, int(att.sum())))
        del out, enc
        self.torch.xpu.empty_cache()
        return res
