#!/bin/bash
set -euxo pipefail

# Benchmarking configurations
MODEL=${MODEL:-"DeepSeek-V3-L"}
MCORE_RELEASE_NUM=${MCORE_RELEASE_NUM:-"0.9"}
WORKSPACE=${WORKSPACE:-"/lustre/fsw/coreai_dlalgo_llm/vince/DS/submit_ds/"} # Path to the Megatron-MoE-Scripts
IMAGE=${IMAGE:-"/lustre/share/coreai_dlalgo_mcore/docker/mcore-moe-pytorch24.11-te-v1.12.sqsh"} # Path to sqsh or docker image url
MEGATRON_PATH=${MEGATRON_PATH:-"/lustre/fsw/coreai_dlalgo_llm/vince/DS/megatron-lm"} # Path to Megatron-LM

# Default Model configurations
declare -A MODEL_CONFIGS
MODEL_CONFIGS[DeepSeek-V3]="1 16 64 1 1 1 512 4096 alltoall true 61 128 00:20:00 1 Slimpajama 4 1"
MODEL_CONFIGS[DeepSeek-V3-L]="1 3 16 1 1 1 256 4096 alltoall true 5 6 00:20:00 1 Slimpajama 3 1"

# Set model and SLURM parameters based on model type
set_model_parameters() {
    local config=(${MODEL_CONFIGS[${MODEL}]})
    export TP=${TP:-${config[0]}}
    export PP=${PP:-${config[1]}}
    export EP=${EP:-${config[2]}}
    export CP=${CP:-${config[3]}}
    export VPP=${VPP:-${config[4]}}
    export MBS=${MBS:-${config[5]}}
    export GBS=${GBS:-${config[6]}}
    export SEQ_LEN=${SEQ_LEN:-${config[7]}}
    export MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-${config[8]}}
    export MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-${config[9]}}
    export NUM_LAYERS=${NUM_LAYERS:-${config[10]}}
    export NODES=${NODES:-${config[11]}}
    export RUN_TIME=${RUN_TIME:-${config[12]}}
    export PRETRAIN=${PRETRAIN:-${config[13]}}
    export DATASET=${DATASET:-${config[14]}}

    if [ ${VPP} -gt 1 ]; then
        export LAYERS_PER_VP=$((NUM_LAYERS / PP / VPP))
    fi

    if [ $((NUM_LAYERS % PP)) -ne 0 ]; then
        export PP_FIRST=${PP_FIRST:-${config[15]}}
        export PP_LAST=${PP_LAST:-${config[16]}}
    fi
}
set_model_parameters

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

#Training environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# SLURM settings
PPP=${PPP:-"coreai_dlalgo_llm"}
PARTITION=${PARTITION:-"batch"}
RUN_NAME=${RUN_NAME:-"${MODEL}-benchmarking"}

# WANDB settings
export WANDB_API_KEY=${WANDB_API_KEY:-"167721d6f27fa4a8147ce96033bf1c6a923fd0b9"}
export WANDB_PROJECT=${WANDB_PEOJECT:-"${USER}-moe-benchmarking-v${MCORE_RELEASE_NUM}"}
export COMMENT=${COMMENT:-"v$MCORE_RELEASE_NUM"}

# Profile settings
export PROFILE=${PROFILE:-0}

echo "WORKSPACE: $WORKSPACE"
echo "MEGATRON_PATH: $MEGATRON_PATH"
echo "IMAGE: $IMAGE"

cat $TRAINING_PARAMS_PATH | envsubst >$TRAINING_PARAMS_PATH.tmp
TRAINING_PARAMS_PATH=$TRAINING_PARAMS_PATH.tmp

# Extract training params to export
TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | select(.value != "false") | with(select(.value == "true"); .value = "") | [.key + " " + .value] | join("")' $TRAINING_PARAMS_PATH | sed "s/(\([^)]*\))/'(\1)'/g" |tr '\n' ' ')
export TRAINING_PARAMS="$TRAINING_PARAMS $TRAINING_PARAMS_FROM_CONFIG"

MOUNTS="/lustre/:/lustre/"
SLURM_LOGS="$OUTPUT_PATH/slurm_logs"
mkdir -p $SLURM_LOGS

#profile
if [ ${PROFILE} = 1 ]; then
    NSYS_PATH=${OUTPUT_PATH}/nsys
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p ${NSYS_PATH}
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-memory-usage true -f true -x true -o ${NSYS_PATH}/${RUN_NAME}-${DATETIME}"
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 50 --profile-step-end 55 --profile-ranks 0 "
else
    PROFILE_CMD=""
fi

export TRAINING_CMD="${PROFILE_CMD} python $TRAINING_SCRIPT_PATH $TRAINING_PARAMS"

set +e
cat > ${MODEL}.sub <<EOF
#!/bin/bash

#SBATCH --nodes=$NODES
#SBATCH --account $PPP
#SBATCH --partition $PARTITION
#SBATCH --ntasks-per-node=8
#SBATCH --time $RUN_TIME
#SBATCH --job-name=$PPP:moe:$RUN_NAME
#SBATCH --dependency=singleton
#SBATCH --output=${WORKSPACE}/slurm.log
#SBATCH --exclusive

# Prepare SLURM job
echo "SLURM_JOB_ID=\${SLURM_JOB_ID}" > "$SLURM_LOGS/\${SLURM_JOB_ID}.log"

srun \
    --mpi=pmix -l \
    --ntasks-per-node=8 \
    --container-image=${IMAGE} \
    --container-mounts=${MOUNTS} \
    --container-workdir=${WORKSPACE} \
    $TRAINING_CMD | tee "$SLURM_LOGS/\${SLURM_JOB_ID}.log" 
EOF

sbatch ${MODEL}.sub
set -e
