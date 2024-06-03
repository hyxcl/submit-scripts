#!/bin/bash
#SBATCH -N 2                  # number of nodes
#SBATCH -t 01:00:00              # wall time  (4 for luna, 8 for backfill, 2 for interactive)
#SBATCH -J llama2_7b  # job name (<< CHANGE ! >>)
#SBATCH --ntasks-per-node=8     # n tasks per machine (one task per gpu) <required>
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/mnt/fs/nemofw/xucl/mlm/stdout/llama2_7b_%j.out

MODEL=llama2_7B

DIR=`pwd`

SCRIPTSDIR="/mnt/fs/nemofw/xucl/mlm"
export NVTE_APPLY_QK_LAYER_SCALING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NVLS_ENABLE=0

srun --container-image /image/nemofw.24.03.01.sqsh  \
	--container-mounts /mnt/fs/nemofw/:/mnt/fs/nemofw/ \
	--container-name megatron-lm-llama \
	--output ${DIR}/logs/llama2_7B_2node_tp1pp4_mbs1gbs64_%j.log \
	bash ${SCRIPTSDIR}/${MODEL}.sh
