#!/usr/bin/env python3
"""Entrena un modelo Liminar con el Trainer de transformers: la capa nacional o la capa de una universidad.

Es la versión sencilla de entrenar.py. Usa los mismos ejemplos, la misma plantilla de chat, la misma pérdida (en las
conversaciones, solo sobre la respuesta) y el mismo programa (una época de solo conversaciones y después varias
épocas con todo, con algunos ejemplos repetidos). El bucle, los lotes, FSDP, los puntos de reanudación y el guardado
quedan a cargo del Trainer. Los lotes se arman como en entrenar.py, por presupuesto de tokens: con FSDP, cada lote
obliga a juntar los pesos entre las GPU, y con lotes de un ejemplo el entrenamiento iba ocho veces más lento. Respecto
de entrenar.py cambian dos cosas: los logits se calculan en toda la secuencia y los datos se leen ya armados.

Datos: un jsonl (o jsonl.gz) con el formato del dataset publicado LICDIA-UNLu/liminar-entrenamiento, o el nombre de
ese dataset en Hugging Face:
    {"tipo": "conversacion", "pregunta": "...", "respuesta": "...", "parte": "...", "id_norma": ...}
    {"tipo": "texto", "texto": "...", "parte": "...", "id_norma": ...}
Una universidad puede armar el suyo con sus actos en el mismo formato; "parte" e "id_norma" son opcionales.

Uso:
    # una GPU: LoRA sobre un modelo Liminar con los actos de la universidad
    python entrenar-transformers.py --modelo LICDIA-UNLu/Liminar-Qwen3.5-9B --datos actos.jsonl --salida capa-propia

    # la capa nacional en un nodo de 8 GPU, como Liminar (ajuste completo con FSDP)
    torchrun --nproc_per_node 8 entrenar-transformers.py --modelo Qwen/Qwen3.5-9B \
        --datos LICDIA-UNLu/liminar-entrenamiento --salida liminar-9b --modo completo --fsdp \
        --repetir generales:4,25394:2
"""
import argparse
import gzip
import json
import os
import random
import re

import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

# Partes de un modelo multimodal que no son el modelo de texto: no se entrenan.
NO_TEXTO = r'visual|vision|audio|multi_modal_projector'


def argumentos():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--modelo', required=True, help='carpeta o repositorio de Hugging Face del modelo de partida')
    ap.add_argument('--datos', required=True, help='jsonl o jsonl.gz con los ejemplos, o un dataset de Hugging Face')
    ap.add_argument('--salida', required=True)
    ap.add_argument('--modo', choices=['lora', 'completo'], default='lora')
    ap.add_argument('--rango', type=int, default=64,
                    help='rango del LoRA (Liminar usó 256 en los modelos de 24B o más)')
    ap.add_argument('--lr', type=float, help='por omisión 1e-4 con LoRA y 1e-5 con ajuste completo')
    ap.add_argument('--epocas', type=float, default=4, help='épocas con todos los ejemplos')
    ap.add_argument('--epocas-preguntas', type=int, default=1, help='épocas previas con solo las conversaciones')
    ap.add_argument('--repetir', default='',
                    help='veces que entra por época lo que coincide con una parte o una norma, p. ej. '
                    '"generales:4,25394:2" (los pedidos generales cuatro veces y la Ley 24.521, dos)')
    ap.add_argument('--modelo-base', help='nombre que reemplaza a {modelo_base} en los ejemplos de identidad')
    ap.add_argument('--max-tokens', type=int, default=3072, help='las conversaciones más largas quedan afuera; '
                    'los textos se cortan')
    ap.add_argument('--tokens-por-lote', type=int, default=2048,
                    help='tamaño del lote, contando el relleno; un ejemplo más largo va solo')
    ap.add_argument('--acumulacion', type=int, default=2, help='lotes por GPU en cada paso del optimizador')
    ap.add_argument('--congelar', help='ajuste completo: expresión regular de parámetros que no se entrenan '
                    '(Gemma 4 E4B: embed_tokens_per_layer)')
    ap.add_argument('--fsdp', action='store_true', help='reparte pesos, gradientes y optimizador entre las GPU')
    ap.add_argument('--validacion', help='jsonl con ejemplos de validación, en el mismo formato')
    ap.add_argument('--eval-cada', type=int, default=200)
    ap.add_argument('--guardar-cada', type=int, default=200, help='pasos entre puntos de reanudación')
    ap.add_argument('--max-ejemplos', type=int, default=0, help='usa solo esta cantidad de ejemplos (canario)')
    ap.add_argument('--max-pasos', type=int, default=-1, help='entrena esta cantidad de pasos (prueba corta)')
    ap.add_argument('--parar-en', type=int, default=0,
                    help='corta en ese paso sin cambiar la tasa de aprendizaje del programa completo (canario)')
    ap.add_argument('--semilla', type=int, default=0)
    return ap.parse_args()


def leer(fuente):
    """Los ejemplos de un jsonl local o del dataset publicado en Hugging Face."""
    ruta = fuente
    if not os.path.exists(fuente):
        from huggingface_hub import hf_hub_download
        ruta = hf_hub_download(fuente, 'data/entrenamiento.jsonl.gz', repo_type='dataset')
    with (gzip.open if ruta.endswith('.gz') else open)(ruta, 'rt', encoding='utf-8') as f:
        return [json.loads(linea) for linea in f if linea.strip()]


class Formato:
    """Convierte cada ejemplo en ids y etiquetas con la plantilla de chat del modelo. En las conversaciones, el prompt
    lleva -100 y se entrena la respuesta con el cierre del turno del asistente (<|im_end|> en Qwen, <turn|> en
    Gemma 4, </s> en Mistral). En los textos se entrena todo, con el BOS si el modelo lo usa y el fin de documento."""
    MARCA = 'RESPUESTA-DE-PRUEBA'

    def __init__(self, tok):
        self.tok = tok
        self.mistral = type(tok).__name__.startswith('MistralCommon')   # tokenizador de mistral-common
        if self.mistral:
            self.cierre, self.bos, self.fin = [tok.eos_token_id], [tok.bos_token_id], tok.eos_token_id
            return
        t = tok.apply_chat_template([{'role': 'user', 'content': 'Hola'},
                                     {'role': 'assistant', 'content': self.MARCA}], tokenize=False)
        self.cierre = tok(t[t.rindex(self.MARCA) + len(self.MARCA):].strip(), add_special_tokens=False)['input_ids']
        self.bos = [tok.bos_token_id] if tok.bos_token and t.startswith(tok.bos_token) else []
        fin = tok.convert_tokens_to_ids('<|endoftext|>')   # Qwen separa documentos con <|endoftext|>
        self.fin = fin if isinstance(fin, int) and fin != tok.unk_token_id else tok.eos_token_id

    def prompt(self, pregunta):
        mensajes = [{'role': 'user', 'content': pregunta}]
        if self.mistral:
            r = self.tok.apply_chat_template(mensajes, tokenize=True)
            return list(r['input_ids'] if hasattr(r, 'keys') else r)
        texto = self.tok.apply_chat_template(mensajes, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        return self.tok(texto, add_special_tokens=False)['input_ids']

    def __call__(self, x, max_tokens):
        if x['tipo'] == 'texto':
            # un texto que no entra se corta sin el fin de documento, así el modelo no aprende que termina ahí
            cuerpo = self.tok(x['texto'], add_special_tokens=False)['input_ids']
            tope = max_tokens - 1 - len(self.bos)
            ids = self.bos + (cuerpo + [self.fin] if len(cuerpo) <= tope else cuerpo[:tope + 1])
            return {'input_ids': ids, 'labels': list(ids)}
        p = self.prompt(x['pregunta'])
        r = self.tok(x['respuesta'], add_special_tokens=False)['input_ids'] + self.cierre
        if len(p) + len(r) > max_tokens:
            return None
        return {'input_ids': p + r, 'labels': [-100] * len(p) + r}


def lotes(ejemplos, presupuesto, rnd):
    """Agrupa los ejemplos en lotes de hasta `presupuesto` tokens contando el relleno. Se recorren de menor a mayor
    largo, así cada lote junta ejemplos de largo parecido, y después se mezclan los lotes."""
    grupos, actual = [], []
    for x in sorted(ejemplos, key=lambda x: len(x['input_ids'])):
        if actual and (len(actual) + 1) * len(x['input_ids']) > presupuesto:
            grupos.append(actual)
            actual = []
        actual.append(x)
    grupos += [actual] if actual else []
    rnd.shuffle(grupos)
    return grupos


def programa(ejemplos, a, rnd):
    """El orden de entrenamiento: primero las épocas de solo conversaciones y después las épocas con todo. En cada
    época, lo indicado en --repetir entra varias veces. Devuelve la lista de lotes en orden y dónde termina cada
    época (en lotes)."""
    veces = {k: int(v) for k, v in (r.split(':') for r in a.repetir.split(',') if r)}
    epoca = [x for x in ejemplos
             for _ in range(max([veces.get(x.get('parte'), 1), veces.get(str(x.get('id_norma')), 1)]))]
    bloques = [[x for x in epoca if x['conversacion']] for _ in range(a.epocas_preguntas)]
    enteras, resto = int(a.epocas), a.epocas - int(a.epocas)
    bloques += [list(epoca) for _ in range(enteras)]
    if resto:   # una fracción de época al final
        bloques.append(rnd.sample(epoca, int(resto * len(epoca))))
    orden, fines = [], []
    for b in bloques:
        rnd.shuffle(b)
        orden += lotes(b, a.tokens_por_lote, rnd)
        fines.append(len(orden))
    return orden, fines


def juntar(ejemplos, pad):
    """Rellena un lote hasta el ejemplo más largo (las etiquetas con -100)."""
    n = max(len(x['input_ids']) for x in ejemplos)

    def rellenar(clave, valor):
        return torch.tensor([x[clave] + [valor] * (n - len(x[clave])) for x in ejemplos])
    return {'input_ids': rellenar('input_ids', pad), 'labels': rellenar('labels', -100),
            'attention_mask': torch.tensor([[1] * len(x['input_ids']) + [0] * (n - len(x['input_ids']))
                                            for x in ejemplos])}


def guardar(trainer, destino, tipos):
    """Guarda el modelo. En el ajuste completo los pesos maestros van en fp32: se guardan con los tipos del modelo
    original (bf16), como los modelos publicados. Con LoRA se guarda el adaptador."""
    if tipos is None:
        trainer.save_model(destino)
        return
    sd = trainer.accelerator.get_state_dict(trainer.model)   # con FSDP, el modelo entero en el proceso 0
    if trainer.is_world_process_zero():
        sd = {k: v.to(tipos.get(k, v.dtype)) for k, v in sd.items()}
        trainer.accelerator.unwrap_model(trainer.model).save_pretrained(destino, state_dict=sd)
        trainer.processing_class.save_pretrained(destino)


class Parar(TrainerCallback):
    """Corta el entrenamiento en un paso dado, con la tasa de aprendizaje del programa completo."""

    def __init__(self, paso):
        self.paso = paso

    def on_step_end(self, args, state, control, **kwargs):
        if self.paso and state.global_step >= self.paso:
            control.should_training_stop = True


class GuardarEpocas(TrainerCallback):
    """Guarda el modelo al terminar cada época del programa (en salida/epoca-N), para poder elegir la época."""

    def __init__(self, pasos, salida, tipos):
        self.pasos, self.salida, self.tipos, self.trainer = pasos, salida, tipos, None

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.pasos:
            n = self.pasos.index(state.global_step) + 1
            guardar(self.trainer, os.path.join(self.salida, f'epoca-{n}'), self.tipos)


def main():
    a = argumentos()
    rnd = random.Random(a.semilla)
    lr = a.lr or (1e-4 if a.modo == 'lora' else 1e-5)
    mundo = int(os.environ.get('WORLD_SIZE', 1))
    xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()

    # Datos
    tok = AutoTokenizer.from_pretrained(a.modelo)
    fmt = Formato(tok)
    crudos = leer(a.datos)
    if a.max_ejemplos:
        crudos = rnd.sample(crudos, min(a.max_ejemplos, len(crudos)))
    base = a.modelo_base or os.path.basename(a.modelo.rstrip('/'))
    ejemplos = []
    for x in crudos:
        if x['tipo'] == 'conversacion':
            x = dict(x, pregunta=x['pregunta'].replace('{modelo_base}', base),
                     respuesta=x['respuesta'].replace('{modelo_base}', base))
        e = fmt(x, a.max_tokens)
        if e is not None:
            ejemplos.append(dict(e, conversacion=x['tipo'] == 'conversacion', parte=x.get('parte'),
                                 id_norma=x.get('id_norma')))
    orden, fines = programa(ejemplos, a, rnd)
    validacion = None   # también en lotes por tokens, así la pérdida queda pesada por token como en entrenar.py
    if a.validacion:
        validacion = lotes([e for e in (fmt(x, a.max_tokens) for x in leer(a.validacion)) if e], a.tokens_por_lote,
                           random.Random(0))
    por_paso = a.acumulacion * mundo   # lotes por paso del optimizador
    pasos_epocas = [-(-n // por_paso) for n in fines]   # el paso en que termina cada época (redondeo hacia arriba)
    if int(os.environ.get('RANK', 0)) == 0:
        print(f'{len(crudos)} ejemplos leídos, {len(ejemplos)} entran (los demás pasan de {a.max_tokens} tokens); '
              f'{len(orden)} lotes de hasta {a.tokens_por_lote} tokens en el programa ({len(fines)} épocas), '
              f'{por_paso} lotes por paso, {pasos_epocas[-1]} pasos', flush=True)

    # Antes de crear el modelo: con FSDP, los argumentos indican que solo el proceso 0 lea los pesos
    args = TrainingArguments(
        output_dir=a.salida, learning_rate=lr, lr_scheduler_type='cosine', warmup_steps=0.03, weight_decay=0.0,
        max_grad_norm=1.0, num_train_epochs=1, max_steps=a.max_pasos, gradient_accumulation_steps=a.acumulacion,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,   # cada elemento ya es un lote
        train_sampling_strategy='sequential',   # el orden ya viene armado por épocas
        bf16=True,
        gradient_checkpointing=not a.fsdp, gradient_checkpointing_kwargs={'use_reentrant': False},
        fsdp=a.fsdp,
        fsdp_config={'version': 2, 'reshard_after_forward': True, 'activation_checkpointing': True,
                     'cpu_ram_efficient_loading': True, 'state_dict_type': 'FULL_STATE_DICT'} if a.fsdp else None,
        eval_strategy='steps' if validacion else 'no', eval_steps=a.eval_cada, eval_on_start=bool(validacion),
        save_strategy='steps', save_steps=a.guardar_cada, save_total_limit=1,
        logging_steps=10, logging_first_step=True, report_to='none', disable_tqdm=True,
        include_num_input_tokens_seen='non_padding',
        remove_unused_columns=False, seed=a.semilla, ddp_timeout=7200)

    # Modelo: con la clase de su config.json (p. ej. Qwen3_5ForConditionalGeneration), así lo guardado tiene los
    # mismos nombres de pesos que el modelo original. En el ajuste completo, los pesos maestros van en fp32.
    args.device   # inicializa el grupo de procesos antes de leer los pesos
    if a.fsdp:   # solo el proceso 0 lee los pesos (los demás crean el modelo vacío y FSDP les reparte su parte)
        os.environ['ACCELERATE_USE_FSDP'] = 'true'
    if xpu and a.fsdp:
        import arreglos_xpu   # arreglos de FSDP para las GPU Intel (arreglos_xpu.py, junto a este archivo)
        arreglos_xpu.aplicar()
    clase = getattr(transformers, (AutoConfig.from_pretrained(a.modelo).architectures or [''])[0],
                    transformers.AutoModelForCausalLM)
    modelo = clase.from_pretrained(a.modelo, dtype='auto')
    tipos = {n: t.dtype for n, t in modelo.state_dict().items()} if a.modo == 'completo' else None
    if a.modo == 'completo':
        modelo = modelo.float()
    for c in (modelo.config, getattr(modelo.config, 'text_config', None)):
        if c is not None:
            c.use_cache = False
    if a.modo == 'lora':
        from peft import LoraConfig, get_peft_model
        modelo = get_peft_model(modelo, LoraConfig(
            r=a.rango, lora_alpha=2 * a.rango, lora_dropout=0.05, task_type='CAUSAL_LM', target_modules='all-linear',
            # ni el codificador visual ni los enrutadores de un MoE
            exclude_modules=rf'.*({NO_TEXTO}).*|.*\.mlp\.gate|.*shared_expert_gate'))
    else:
        for n, p in modelo.named_parameters():
            p.requires_grad_(not re.search(NO_TEXTO, n) and not (a.congelar and re.search(a.congelar, n)))

    epocas = GuardarEpocas(pasos_epocas, a.salida, tipos)
    trainer = Trainer(model=modelo, args=args, train_dataset=orden, eval_dataset=validacion,
                      data_collator=lambda x: juntar(x[0], tok.pad_token_id if tok.pad_token_id is not None
                                                     else fmt.fin),
                      processing_class=tok, callbacks=[epocas, Parar(a.parar_en)])
    epocas.trainer = trainer
    reanudar = any(d.startswith('checkpoint-') for d in os.listdir(a.salida)) if os.path.isdir(a.salida) else False
    trainer.train(resume_from_checkpoint=reanudar or None)
    guardar(trainer, os.path.join(a.salida, 'final'), tipos)
    if trainer.is_world_process_zero():
        with open(os.path.join(a.salida, 'registro.json'), 'w', encoding='utf-8') as f:
            json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
