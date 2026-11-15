"""Funciones compartidas del entrenamiento: armado de lotes, pérdida solo sobre la respuesta y carga con LoRA."""
import torch


def lote(items, pad):
    L = max(len(x['input_ids']) for x in items)
    ids = torch.full((len(items), L), pad)
    lab = torch.full((len(items), L), -100)
    att = torch.zeros((len(items), L), dtype=torch.long)
    for j, x in enumerate(items):
        n = len(x['input_ids'])
        ids[j, :n] = torch.tensor(x['input_ids'])
        lab[j, :n] = torch.tensor(x['labels'])
        att[j, :n] = 1
    return ids, lab, att


def perdida(modelo, ids, lab, att, disp):
    """Entropía cruzada solo sobre la respuesta, calculando logits únicamente en esas
    posiciones: con prompts de ~6.000 tokens y un vocabulario de ~248.000, materializar
    los logits de todo el prompt no entra en memoria."""
    pos = torch.nonzero((lab[:, 1:] != -100).any(0)).squeeze(1)
    out = modelo(input_ids=ids.to(disp), attention_mask=att.to(disp), logits_to_keep=pos.to(disp))
    y = lab[:, 1:][:, pos].to(disp)
    logits = out.logits.float()
    return torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                             ignore_index=-100), int((y != -100).sum())


def cargar_modelo(ruta, rango, dev):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(ruta, dtype=torch.bfloat16).to(dev)
    lineales = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)
                and 'visual' not in n and 'vision' not in n and not n.endswith('lm_head')]
    m = get_peft_model(m, LoraConfig(r=rango, lora_alpha=2 * rango, lora_dropout=0.05,
                                     target_modules=lineales, task_type='CAUSAL_LM'))
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    m.enable_input_require_grads()
    return m
