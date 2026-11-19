#!/bin/bash
# Entrena con entrenar-transformers.py en un nodo: un proceso por tile sobre oneCCL, con la misma configuración de
# oneCCL que entrenar.sbatch. Si un proceso se cae, se relanza y el Trainer retoma desde el último punto guardado.
#
# Encolar parado en la raíz del repositorio:
#     DATOS=.../entrenamiento.jsonl.gz SALIDA=salidas/qwen3.5-9b \
#       EXTRA="--modelo .../Qwen3.5-9B --modo completo --fsdp --repetir generales:4,25394:2" sbatch slurm/entrenar_transformers.sbatch
#     MISTRAL_COMMON=1 EXTRA="--modelo .../Mistral-Small-3.2-24B-Instruct-2506 --rango 256 --fsdp ..." ...
#
#SBATCH --job-name=lim-trainer
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=registros/%x_%j.out
#SBATCH --error=registros/%x_%j.err

set -uo pipefail
BASE="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$BASE" || exit 1
DATOS="${DATOS:?falta DATOS}"
SALIDA="${SALIDA:?falta SALIDA}"
EXTRA="${EXTRA:-}"
SCRIPT=${SCRIPT:-liminar/entrenamiento/entrenar-transformers.py}
PY=${LIMINAR_PY:-python3}
NPROC=${NPROC:-8}   # procesos (tiles); con --gres=gpu:2 van 4

ulimit -n "$(ulimit -Hn)"   # el límite blando de archivos abiertos (1024) no alcanza para oneCCL
NUCLEOS=$($PY -c "import os; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:$NPROC])))")
# El reparto de los pesos desde el proceso 0 (cpu_ram_efficient_loading) usa envíos punto a punto, que oneCCL no admite
# con el intercambio por sockets de entrenar.sbatch.
export CCL_WORKER_COUNT=1 CCL_WORKER_AFFINITY=$NUCLEOS CCL_PROCESS_LAUNCHER=torchrun CCL_ZE_IPC_EXCHANGE=${IPC_ONECCL:-pidfd}
export FI_PROVIDER=tcp TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
# Mistral Small 3.2 y Ministral 3 usan el tokenizador de mistral-common, instalado aparte.
if [ -n "${MISTRAL_COMMON:-}" ]; then export PYTHONPATH=${LIMINAR_MISTRAL_COMMON:-}${PYTHONPATH:+:$PYTHONPATH}; fi

echo "nodo $(hostname) · datos $DATOS · salida $SALIDA · $EXTRA"
for intento in 1 2 3 4; do
  echo "=== intento $intento · $(date '+%F %T')"
  $PY -m torch.distributed.run --standalone --nproc_per_node $NPROC "$SCRIPT" --datos "$DATOS" --salida "$SALIDA" $EXTRA \
    && exit 0
  echo "--- falló el intento $intento · $(date '+%F %T')"
done
exit 1
