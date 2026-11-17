#!/usr/bin/env python3
"""Junta los resultados de la evaluación de cada modelo contra su base: una tabla para leer y un JSON con la forma que
usa liminar.herramientas.fichas.

Lee, dentro de --datos:
  linea_base-v2/juicios/<nombre>.jsonl               benchmark de normativa (nota del juez de 0 a 2)
  benchmarks/opcion-texto/<nombre>.jsonl             MMMLU ES-LA con respuesta redactada (acierto por pregunta)
  rag-escenarios/juicios-muestra-60/<nombre>.jsonl   muestra RAG (rúbrica del juez de 1 a 5 e incorrectas)

Los nombres del modelo base son <clave> en el benchmark y base-<clave> en el resto; los del ajustado,
liminar-<ronda>-<clave>.

Uso:
    python -m liminar.evaluacion.resultados --datos DIR --ronda v2r3 [--json resultados.json]
"""
import argparse
import json
import os
from statistics import mean

MUESTRA_RAG = 60
DIMENSIONES = ('utilidad', 'fidelidad', 'completitud', 'claridad')
CLAVES = ['gemma-4-e4b', 'ministral-3-8b', 'qwen3.5-9b', 'gemma-4-12b', 'mistral-small-3.2-24b', 'qwen3.5-27b',
          'gemma-4-31b', 'qwen3.5-122b-a10b']


def leer(ruta):
    if not os.path.exists(ruta):
        return None
    return [json.loads(l) for l in open(ruta, encoding='utf-8') if l.strip()]


def dominio(d, nombre):
    """Nota media del juez (0 a 2) y porcentaje de respuestas correctas (nota 2)."""
    js = leer(os.path.join(d, 'linea_base-v2', 'juicios', f'{nombre}.jsonl'))
    notas = [j['nota'] for j in js or [] if j.get('nota') is not None]
    if not notas:
        return None
    return {'nota': round(mean(notas), 3), 'correctas': round(100 * notas.count(2) / len(notas), 1), 'n': len(notas)}


def mmmlu(d, nombre):
    """Exactitud (%) con la respuesta redactada: la letra se lee del texto; sin letra reconocible cuenta como error."""
    fs = leer(os.path.join(d, 'benchmarks', 'opcion-texto', f'{nombre}.jsonl'))
    return round(100 * mean(f['acierto'] for f in fs), 1) if fs else None


def rag(d, nombre):
    js = leer(os.path.join(d, 'rag-escenarios', 'juicios-muestra-60', f'{nombre}.jsonl'))
    # un juicio por pregunta: si una pregunta quedó juzgada dos veces (dos jueces a la vez), vale el último
    js = list({j['id']: j['juicio'] for j in js or [] if isinstance(j.get('juicio'), dict)}.values())
    if not js:
        return None
    return {'promedio': round(mean(mean(j[k] for k in DIMENSIONES) for j in js), 3),
            'utilidad': round(mean(j['utilidad'] for j in js), 3),
            'incorrectas': sum(bool(j.get('incorrecto')) for j in js), 'n': len(js)}


NOMBRES = {'gemma-4-e4b': 'Liminar-Gemma-4-E4B', 'ministral-3-8b': 'Liminar-Ministral-3-8B',
           'qwen3.5-9b': 'Liminar-Qwen3.5-9B', 'gemma-4-12b': 'Liminar-Gemma-4-12B',
           'mistral-small-3.2-24b': 'Liminar-Mistral-Small-3.2-24B', 'qwen3.5-27b': 'Liminar-Qwen3.5-27B',
           'gemma-4-31b': 'Liminar-Gemma-4-31B', 'qwen3.5-122b-a10b': 'Liminar-Qwen3.5-122B-A10B'}


BASES = {'gemma-4-e4b': 'Gemma 4 E4B', 'ministral-3-8b': 'Ministral 3 8B', 'qwen3.5-9b': 'Qwen 3.5 9B',
         'gemma-4-12b': 'Gemma 4 12B', 'mistral-small-3.2-24b': 'Mistral Small 3.2 24B', 'qwen3.5-27b': 'Qwen 3.5 27B',
         'gemma-4-31b': 'Gemma 4 31B', 'qwen3.5-122b-a10b': 'Qwen 3.5 122B-A10B'}


def tabla_markdown(res):
    """Dos filas por modelo con las tres medidas completas (el base y su Liminar), en el orden de CLAVES, y al final la
    referencia gpt-4.1-mini."""
    n = lambda x, d=1: '-' if x is None else f'{x:.{d}f}'.replace('.', ',')
    fila = lambda nombre, d, q, m: (f'| {nombre} | {n(d["correctas"])} % | {n(d["nota"], 2)} | {n(q["promedio"], 2)} | '
                                    f'{q["incorrectas"]} | {n(m)}{"" if m is None else " %"} |')
    filas = ['| Modelo | Normativa sin documentos: correctas | Nota del juez (0 a 2) | RAG: promedio (1 a 5) | '
             'RAG: incorrectas (de 60) | MMMLU ES-LA |', '|---|---|---|---|---|---|']
    for c in CLAVES:
        r = res.get(c) or {}
        d, q, m = r.get('dominio') or {}, r.get('rag') or {}, r.get('mmmlu') or {}
        if not (d.get('base') and d.get('liminar') and q.get('base') and q.get('liminar') and q['liminar']['n'] >= MUESTRA_RAG
                and m.get('base') is not None and m.get('liminar') is not None):
            continue
        filas.append(fila(f'{BASES[c]} (base)', d['base'], q['base'], m['base']))
        filas.append(fila(NOMBRES[c], d['liminar'], q['liminar'], m['liminar']))
    ref = res.get('referencia-gpt-4.1-mini') or {}
    if ref.get('dominio') and ref.get('rag'):
        filas.append(fila('gpt-4.1-mini (referencia)', ref['dominio'], ref['rag'], None))
    return '\n'.join(filas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datos', required=True)
    ap.add_argument('--ronda', default='v2r3')
    ap.add_argument('--json')
    ap.add_argument('--markdown', help='tabla de la documentación con los modelos que tienen las tres medidas completas')
    a = ap.parse_args()
    res = {}
    par = lambda x, y: f'{"-" if x is None else x} → {"-" if y is None else y}'
    print(f'{"modelo":24s} {"normativa: correctas %":>24s} {"nota (0-2)":>15s} {"MMMLU ES-LA %":>15s} '
          f'{"RAG (1-5)":>15s} {"RAG incorrectas":>17s}')
    for c in CLAVES:
        aj = f'liminar-{a.ronda}-{c}'
        r = {}
        db, dl = dominio(a.datos, c), dominio(a.datos, aj)
        if db or dl:
            r['dominio'] = {'base': db, 'liminar': dl}
        mb, ml = mmmlu(a.datos, f'base-{c}'), mmmlu(a.datos, aj)
        if mb is not None or ml is not None:
            r['mmmlu'] = {'base': mb, 'liminar': ml}
        rb, rl = rag(a.datos, f'base-{c}'), rag(a.datos, aj)
        if rb or rl:
            r['rag'] = {'base': rb, 'liminar': rl}
        if r:
            res[c] = r
        g = lambda x, k: None if not x else x[k]
        # la muestra RAG se juzga de a poco: si un modelo todavía no tiene las 60, se muestra cuántas lleva
        u = lambda x: None if not x else (x['promedio'] if x['n'] >= MUESTRA_RAG else f'{x["promedio"]} ({x["n"]})')
        print(f'{c:24s} {par(g(db, "correctas"), g(dl, "correctas")):>24s} {par(g(db, "nota"), g(dl, "nota")):>15s} '
              f'{par(mb, ml):>15s} {par(u(rb), u(rl)):>15s} {par(g(rb, "incorrectas"), g(rl, "incorrectas")):>17s}')
    ref = {'dominio': dominio(a.datos, 'gpt-4.1-mini'), 'rag': rag(a.datos, 'gpt-4.1-mini')}
    if any(ref.values()):
        res['referencia-gpt-4.1-mini'] = ref
        print(f'\nreferencia gpt-4.1-mini: {ref}')
    if a.markdown:
        with open(a.markdown, 'w', encoding='utf-8') as out:
            out.write(tabla_markdown(res) + '\n')
        print(f'\n{a.markdown}')
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as out:
            json.dump(res, out, ensure_ascii=False, indent=1)
        print(f'\n{a.json}')


if __name__ == '__main__':
    main()
