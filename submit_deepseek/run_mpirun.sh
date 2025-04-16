#!/bin/bash
set -euxo pipefail

# Benchmarking configurations
MODEL=${MODEL:-"DeepSeek-V3-L"}
MCORE_RELEASE_NUM=${MCORE_RELEASE_NUM:-"0.9"}
WORKSPACE=${WORKSPACE:-"/home/dgxuser/vince/megatron-models-launcher"} # Path to the Launcher-Scripts
IMAGE=${IMAGE:-"/lustre/share/coreai_dlalgo_mcore/docker/mcore-moe-pytorch24.11-te-v1.12.sqsh"} # Path to sqsh or docker image url
MEGATRON_PATH=${MEGATRON_PATH:-"/lustre/fsw/coreai_dlalgo_llm/vince/DS/megatron-lm"} # Path to Megatron-LM

# some model default config
case $MODEL in
    DeepSeek-V3-L )
        export TP=${TP:-"1"} PP=${PP:-"16"} EP=${EP:-"64"} CP=${CP:-"1"} VPP=${VPP:-"1"} MBS=${MBS:-"1"} GBS=${GBS:-"512"} \
        SEQ_LEN=${SEQ_LEN:-"4096"} MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-"alltoall"} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-"true"} \
        PRETRAIN=${PRETRAIN:-"1"}  DATASET=${DATASET:-"Slimpajama"} PP_FIRST=${PP_FIRST:-"4"} PP_LAST=${PP_LAST:-"1"} NODES=${NODES:-"128"}
        ;;
    DeepSeek-V3 )
        export TP=${TP:-"1"} PP=${PP:-"16"} EP=${EP:-"64"} CP=${CP:-"1"} VPP=${VPP:-"1"} MBS=${MBS:-"1"} GBS=${GBS:-"512"} \
        SEQ_LEN=${SEQ_LEN:-"4096"} MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-"alltoall"} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-"true"} \
        PRETRAIN=${PRETRAIN:-"1"}  DATASET=${DATASET:-"Slimpajama"} PP_FIRST=${PP_FIRST:-"4"} PP_LAST=${PP_LAST:-"1"} NODES=${NODES:-"128"}
        ;;
esac


# Set training parameters, use all passed command line arguments if not pre-defined
TRAINING_PARAMS=${TRAINING_PARAMS:-$@}

#profile
PROFILE=${PROFILE:-0}
if [ ${PROFILE} = 1 ]; then
    NSYS_PATH=${OUTPUT_PATH}/nsys
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p ${NSYS_PATH}
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage true -f true -x true -o ${NSYS_PATH}/${RUN_NAME}-${DATETIME}"
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 50 --profile-step-end 55 --profile-ranks 0 "
else
    PROFILE_CMD=""
fi

#VPP and uneven PP
if [ ${VPP} -gt 1 ]; then
    TRAINING_PARAMS="$TRAINING_PARAMS --num-virtual-stages-per-pipeline-rank ${VPP}"
fi

if [[ ! -z ${PP_FIRST} && ! -z ${PP_LAST} ]]; then
    TRAINING_PARAMS="$TRAINING_PARAMS --decoder-first-pipeline-num-layers ${PP_FIRST} --decoder-last-pipeline-num-layers ${PP_LAST}"
fi

# FP8 arguments
PR=${PR:-bf16}
if [ ${PR} = "bf16" ]; then
    :
elif [ ${PR} = "fp8" ]; then
    TRAINING_PARAMS="$TRAINING_PARAMS --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
    MOE_GROUPED_GEMM="false"
else
    echo "Error: The valid values for PR are 'bf16' or 'fp8'. Current value: ${PR}."
    exit 1
fi

#SP
if [ $TP -gt 1 ]; then
    export SP="true"
else
    export SP="false"
fi

#1F1B
A2A_OVLP=${A2A_OVLP:-0}
if [ ${A2A_OVLP} = "ep_a2a" ]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --combined-1f1b --combined-1f1b-recipe ep_a2a"
elif [ ${A2A_OVLP} = "golden" ]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --combined-1f1b --combined-1f1b-recipe golden"
fi

SPLIT_BW=${SPLIT_BW:-0}
if [ ${SPLIT_BW} = 1 ]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --split-bw"
fi

#MTP
MTP=${MTP:-0}
if [ $MTP -gt 0 ]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --mtp-num-layers $MTP"
fi


# Benchmarking paths
TRAINING_SCRIPT_PATH="${MEGATRON_PATH}/pretrain_gpt.py"
TRAINING_PARAMS_PATH="${WORKSPACE}/model_configs/${MODEL}.yaml"
export OUTPUT_PATH="${WORKSPACE}/output/mcore-benchmarking-v${MCORE_RELEASE_NUM}/${MODEL}-TP${TP}PP${PP}EP${EP}VPP${VPP}-MBS${MBS}GBS${GBS}"
export DATA_PATH="/lustre/share/coreai_dlalgo_mcore/Dataset/Slimpajama"


cat $TRAINING_PARAMS_PATH | envsubst >$TRAINING_PARAMS_PATH.tmp
TRAINING_PARAMS_PATH=$TRAINING_PARAMS_PATH.tmp

# Extract training params to export
TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | select(.value != "false") | with(select(.value == "true"); .value = "") | [.key + " " + .value] | join("")' $TRAINING_PARAMS_PATH | sed "s/(\([^)]*\))/'(\1)'/g" |tr '\n' ' ')
export TRAINING_PARAMS="$TRAINING_PARAMS $TRAINING_PARAMS_FROM_CONFIG"
rm ${TRAINING_PARAMS_PATH}.tmp

export TRAINING_CMD="${PROFILE_CMD} python $TRAINING_SCRIPT_PATH $TRAINING_PARAMS"

#Training environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT=${WANDB_PEOJECT:-"${USER}-moe-benchmarking-v${MCORE_RELEASE_NUM}"}
export COMMENT=${COMMENT:-"v$MCORE_RELEASE_NUM"}
export TRANSFORMERS_OFFLINE=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0 # Diable FP8 AMAX reduction in the data-parallel domain
export NVTE_ASYNC_AMAX_REDUCTION=1    # Enable asynchronous FP8 AMAX reduction
export HYDRA_FULL_ERROR=1
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_4
export GLOO_SOCKET_IFNAME=ens19f0
export NCCL_SOCKET_IFNAME=ens19f0
export DIR="/mnt/fs/nemofw/xucl/2405/tpcommoverlap"
export DATADIR="/mnt/fs/nemofw/llama_data/llama/"
export NUM_NODES=2
export NCCL_DEBUG=INFO


# SLURM settings
MOUNTS="/lustre/:/lustre/"                              # set the mount path of your system , should include the launcher path and the megatron path
SLURM_LOGS="$OUTPUT_PATH/slurm_logs"
mkdir -p $SLURM_LOGS

mpirun -np 16 -H h20n4:8,h20n5:8 -x NVIDIA_PYTORCH_VERSION -x LD_LIBRARY_PATH  -x PYTHONPATH -x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION -x TRANSFORMERS_OFFLINE -x TORCH_NCCL_AVOID_RECORD_STREAMS -x NVTE_DP_AMAX_REDUCE_INTERVA -x NVTE_ASYNC_AMAX_REDUCTION -x CUDA_DEVICE_MAX_CONNECTIONS -x HYDRA_FULL_ERROR -x NVTE_FUSED_ATTN -x NCCL_IB_HCA -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG --allow-run-as-root bash -c "$TRAINING_CMD"
