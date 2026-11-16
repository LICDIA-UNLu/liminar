#!/bin/bash
# Juez de Liminar (Llama 3.3 70B, dos placas) sobre todo lo que falte juzgar, con vigía:
#   1. el benchmark a libro cerrado (linea_base juzgar: retoma por archivo de respuestas) y su resumen;
#   2. la muestra RAG de 60 preguntas (juez_rag: retoma por modelo y pregunta) y su resumen, con gpt-4.1-mini;
#   3. los duelos de m-ArenaHard ES, cada modelo ajustado contra su base (generales duelo: retoma por consigna y orden).
# Se puede encolar cuantas veces haga falta: cada corrida juzga solo lo nuevo. ETAPAS elige qué corre (por defecto
# "benchmark rag duelos"), por ejemplo: sbatch --export=ALL,ETAPAS=benchmark ...
#
# Encolar parado en la raíz del repositorio:
#     LIMINAR_DATOS=... LIMINAR_JUEZ=... sbatch -p PARTICION slurm/juez.sbatch
#
#SBATCH --job-name=nac-juez
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=registros/%x_%j.out
#SBATCH --error=registros/%x_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}" || exit 1
N=${LIMINAR_DATOS:?falta LIMINAR_DATOS}
PY=${LIMINAR_PY:-python3}
JUEZ=${LIMINAR_MODELOS:?falta LIMINAR_MODELOS}/Llama-3.3-70B-Instruct
REG=registros/${SLURM_JOB_NAME:-nac-juez}_${SLURM_JOB_ID:-0}.out   # el vigía mira este archivo
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset OPENAI_API_KEY RAG_LLM_BASE
echo "nodo $(hostname) · $(date '+%F %T')"
ETAPAS=" ${ETAPAS:-benchmark rag duelos} "

if [[ $ETAPAS == *" benchmark "* ]]; then
echo "=== benchmark a libro cerrado · $(date '+%T')"
bash slurm/con_vigia.sh "$REG" 40 6 $PY -m liminar.evaluacion.linea_base juzgar \
  --respuestas $N/linea_base-v2/respuestas --preguntas $N/linea_base-preguntas-v2.jsonl --juez $JUEZ \
  --salida $N/linea_base-v2/juicios --lote 8
$PY -m liminar.evaluacion.linea_base resumen --respuestas $N/linea_base-v2/respuestas --juicios $N/linea_base-v2/juicios
fi

if [[ $ETAPAS == *" rag "* ]]; then
echo "=== muestra RAG de 60 · $(date '+%T')"
# primero los modelos base y los ajustados. En las Intel Max 1550, el runtime aborta (139) cada pocos minutos con este
# juez y contextos largos, con dos o tres placas y aun con --lote 1: avanza de a una docena de juicios por intento, así
# que el vigía lo reintenta muchas veces.
bash slurm/con_vigia.sh "$REG" 40 40 $PY -m liminar.evaluacion.juez_rag --datos ${LIMINAR_EJEMPLOS_RAG:?falta LIMINAR_EJEMPLOS_RAG} \
  --respuestas "$N/rag-escenarios/muestra-60/base-*.jsonl" "$N/rag-escenarios/muestra-60/liminar-*.jsonl" \
  "$N/rag-escenarios/referencias/gpt-*.jsonl" --salida $N/rag-escenarios/juicios-muestra-60 --muestra 60 --lote 1
fi

if [[ $ETAPAS == *" duelos "* ]]; then
echo "=== generales: duelos de m-ArenaHard ES, ajustado contra base · $(date '+%T')"
B=$N/benchmarks
# cada modelo ajustado (o interpolado) que tenga respuestas abiertas, contra su base
PARES=$(for f in $B/abiertas/liminar-*.jsonl; do n=$(basename "$f" .jsonl)
          for m in ministral-3-8b gemma-4-e4b qwen3.5-9b gemma-4-12b mistral-small-3.2-24b qwen3.5-27b gemma-4-31b; do
            if [[ $n == *-$m ]]; then printf 'base-%s:%s,' $m "$n"; fi; done; done)
bash slurm/con_vigia.sh "$REG" 40 6 $PY -m liminar.evaluacion.generales duelo --respuestas $B/abiertas \
  --pares "${PARES%,}" --consignas $B/arenahard-es-200.jsonl --juez $JUEZ --salida $B/duelos --lote 2
$PY -m liminar.evaluacion.generales resumen --opcion $B/opcion --duelos $B/duelos
fi
echo "=== fin · $(date '+%T')"
