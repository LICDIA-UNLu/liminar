#!/usr/bin/env python3
"""Junta los resultados de la evaluación de cada modelo contra su base: una tabla para leer y un JSON con la forma que
usa liminar.herramientas.fichas.

Lee, dentro de --datos:
  linea_base-v2/juicios/<nombre>.jsonl               benchmark de normativa (nota del juez de 0 a 2)
  benchmarks/opcion/<nombre>.jsonl                   MMMLU ES-LA (acierto por pregunta)
  benchmarks/duelos/<ajustado>__vs__<base>.jsonl     m-ArenaHard ES (quién gana cada comparación)
  rag-escenarios/juicios-muestra-60/<nombre>.jsonl   muestra RAG (rúbrica del juez de 1 a 5 e incorrectas)

Los nombres del modelo base son <clave> en el benchmark y base-<clave> en el resto; los del ajustado,
liminar-<ronda>-<clave>.

Uso:
    python -m liminar.evaluacion.resultados --datos DIR --ronda v2r1 [--json resultados.json]
"""
import argparse
import json
import os
from statistics import mean

MUESTRA_RAG = 60
CLAVES = ['gemma-4-e4b', 'ministral-3-8b', 'qwen3.5-9b', 'gemma-4-12b', 'mistral-small-3.2-24b', 'qwen3.5-27b',
          'gemma-4-31b', 'qwen3.5-122b-a10b']


def leer(ruta):
    if not os.path.exists(ruta):
        return None
    return [json.loads(l) for l in open(ruta, encoding='utf-8') if l.strip()]


def dominio(d, nombre):
    js = leer(os.path.join(d, 'linea_base-v2', 'juicios', f'{nombre}.jsonl'))
    notas = [j['nota'] for j in js or [] if j.get('nota') is not None]
    return round(mean(notas), 3) if notas else None


def mmmlu(d, nombre):
    fs = leer(os.path.join(d, 'benchmarks', 'opcion', f'{nombre}.jsonl'))
    return round(100 * mean(f['acierto'] for f in fs), 2) if fs else None


def arena(d, ajustado, base):
    js = leer(os.path.join(d, 'benchmarks', 'duelos', f'{ajustado}__vs__{base}.jsonl'))
    validos = [j['gana'] for j in js or [] if j.get('gana')]
    if not validos:
        return None
    return round(100 * (validos.count('ajustado') + 0.5 * validos.count('empate')) / len(validos), 2)


def rag(d, nombre):
    js = leer(os.path.join(d, 'rag-escenarios', 'juicios-muestra-60', f'{nombre}.jsonl'))
    # un juicio por pregunta: si una pregunta quedó juzgada dos veces (dos jueces a la vez), vale el último
    js = list({j['id']: j['juicio'] for j in js or [] if isinstance(j.get('juicio'), dict)}.values())
    if not js:
        return None
    return {'utilidad': round(mean(j['utilidad'] for j in js), 3),
            'incorrectas': round(100 * mean(bool(j.get('incorrecto')) for j in js), 1), 'n': len(js)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datos', required=True)
    ap.add_argument('--ronda', default='v2r1')
    ap.add_argument('--json')
    a = ap.parse_args()
    res = {}
    print(f'{"modelo":24s} {"normativa (0-2)":>17s} {"MMMLU ES-LA %":>15s} {"ArenaHard %":>12s} {"RAG utilidad":>20s}')
    for c in CLAVES:
        aj = f'liminar-{a.ronda}-{c}'
        r = {}
        dom_b, dom_l = dominio(a.datos, c), dominio(a.datos, aj)
        if dom_b is not None or dom_l is not None:
            r['dominio'] = {'base': dom_b, 'liminar': dom_l}
        mm_b, mm_l = mmmlu(a.datos, f'base-{c}'), mmmlu(a.datos, aj)
        if mm_b is not None or mm_l is not None:
            r['mmmlu'] = {'base': mm_b, 'liminar': mm_l}
        ar = arena(a.datos, aj, f'base-{c}')
        if ar is not None:
            r['arena'] = {'liminar': ar}
        rb, rl = rag(a.datos, f'base-{c}'), rag(a.datos, aj)
        if rb or rl:
            r['rag'] = {'base': rb and rb['utilidad'], 'liminar': rl and rl['utilidad'],
                        'incorrectas_base': rb and rb['incorrectas'], 'incorrectas_liminar': rl and rl['incorrectas']}
        if r:
            res[c] = r
        par = lambda x, y: f'{"-" if x is None else x} → {"-" if y is None else y}'
        # la muestra RAG se juzga de a poco: si un modelo todavía no tiene las 60, se muestra cuántas lleva
        u = lambda r: None if not r else (r['utilidad'] if r['n'] >= MUESTRA_RAG else f'{r["utilidad"]} ({r["n"]})')
        print(f'{c:24s} {par(dom_b, dom_l):>17s} {par(mm_b, mm_l):>15s} {"-" if ar is None else ar:>12} '
              f'{par(u(rb), u(rl)):>20s}')
    gpt = rag(a.datos, 'gpt-4.1-mini')
    if gpt:
        print(f'\nreferencia RAG gpt-4.1-mini: utilidad {gpt["utilidad"]}, incorrectas {gpt["incorrectas"]} % ({gpt["n"]} consultas)')
    if a.json:
        with open(a.json, 'w', encoding='utf-8') as out:
            json.dump(res, out, ensure_ascii=False, indent=1)
        print(f'\n{a.json}')


if __name__ == '__main__':
    main()
