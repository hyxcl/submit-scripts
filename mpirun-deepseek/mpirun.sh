#!/bin/bash
set -euxo pipefail

# Benchmarking configurations
MODEL=${MODEL:-"DeepSeek-V3-singlenode-L20"}
MCORE_RELEASE_NUM=${MCORE_RELEASE_NUM:-"0.9"}
WORKSPACE=${WORKSPACE:-"/home/congliangx/submit_deepseek"} # Path to the Launcher-Scripts
MEGATRON_PATH=${MEGATRON_PATH:-"/home/congliangx/Megatron-LM"} # Path to Megatron-LM

# some model default config
case $MODEL in
    DeepSeek-V3-singlenode-L20 )
        export TP=${TP:-"1"} PP=${PP:-"4"} EP=${EP:-"2"} CP=${CP:-"1"} VPP=${VPP:-"1"} MBS=${MBS:-"1"} GBS=${GBS:-"32"} \
        SEQ_LEN=${SEQ_LEN:-"4096"} MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-"alltoall"} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-"true"} \
        PRETRAIN=${PRETRAIN:-"1"}  DATASET=${DATASET:-"Slimpajama"} PP_FIRST=${PP_FIRST:-"2"} PP_LAST=${PP_LAST:-"1"} NODES=${NODES:-"1"}
        ;;
    DeepSeek-V3 )
        export TP=${TP:-"1"} PP=${PP:-"16"} EP=${EP:-"64"} CP=${CP:-"1"} VPP=${VPP:-"1"} MBS=${MBS:-"1"} GBS=${GBS:-"512"} \
        SEQ_LEN=${SEQ_LEN:-"4096"} MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-"alltoall"} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-"true"} \
        PRETRAIN=${PRETRAIN:-"1"}  DATASET=${DATASET:-"Slimpajama"} PP_FIRST=${PP_FIRST:-"4"} PP_LAST=${PP_LAST:-"1"} NODES=${NODES:-"1"}
        ;;
esac


#Can VPP work together with PP_FIRST PP_LAST?  need to (NUM_LAYERS-PP_FIRST-PP_LAST)/(PP-2)/VPP  ??
if [ ${VPP} -gt 1 ]; then
    export LAYERS_PER_VP=$((NUM_LAYERS / PP / VPP))
fi

# Set training parameters, use all passed command line arguments if not pre-defined
TRAINING_PARAMS=${TRAINING_PARAMS:-$@}

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

# Benchmarking paths
TRAINING_SCRIPT_PATH="${MEGATRON_PATH}/pretrain_gpt.py"
TRAINING_PARAMS_PATH="${WORKSPACE}/${MODEL}.yaml"
export OUTPUT_PATH="${WORKSPACE}/output/mcore-benchmarking-v${MCORE_RELEASE_NUM}/${MODEL}-TP${TP}PP${PP}EP${EP}VPP${VPP}-MBS${MBS}GBS${GBS}"
export DATA_PATH="/lustre/share/coreai_dlalgo_mcore/Dataset/Slimpajama"


cat $TRAINING_PARAMS_PATH | envsubst >$TRAINING_PARAMS_PATH.tmp
TRAINING_PARAMS_PATH=$TRAINING_PARAMS_PATH.tmp

# Extract training params to export
TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | select(.value != "false") | with(select(.value == "true"); .value = "") | [.key + " " + .value] | join("")' $TRAINING_PARAMS_PATH | sed "s/(\([^)]*\))/'(\1)'/g" |tr '\n' ' ')
export TRAINING_PARAMS="$TRAINING_PARAMS $TRAINING_PARAMS_FROM_CONFIG"

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

export TRAINING_CMD="${PROFILE_CMD} torchrun --nproc_per_node 8 --nnodes 1 $TRAINING_SCRIPT_PATH $TRAINING_PARAMS"

#Training environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export TRANSFORMERS_OFFLINE=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0 # Diable FP8 AMAX reduction in the data-parallel domain
export NVTE_ASYNC_AMAX_REDUCTION=1    # Enable asynchronous FP8 AMAX reduction
export HYDRA_FULL_ERROR=1
#export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_4
export GLOO_SOCKET_IFNAME=bond0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO

export MASTER_ADDR="127.0.0.1"   # set to the network address of one of the nodes you run. 
export MASTER_PORT=12346 	
export WORLD_SIZE=$(expr 8 \* $NODES)
export HOSTS=${HOSTS:-"h20n4:1"}


# SLURM settings

mpirun -np $NODES -H $HOSTS -x NVIDIA_PYTORCH_VERSION -x LD_LIBRARY_PATH  -x PYTHONPATH -x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION -x TRANSFORMERS_OFFLINE -x TORCH_NCCL_AVOID_RECORD_STREAMS -x NVTE_DP_AMAX_REDUCE_INTERVA -x NVTE_ASYNC_AMAX_REDUCTION -x CUDA_DEVICE_MAX_CONNECTIONS -x HYDRA_FULL_ERROR -x NVTE_FUSED_ATTN -x NCCL_IB_HCA -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG --allow-run-as-root bash -c "$TRAINING_CMD"
