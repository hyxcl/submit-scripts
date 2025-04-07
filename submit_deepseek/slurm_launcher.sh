#!/bin/bash
set -euxo pipefail

# Benchmarking configurations
export MODEL=${MODEL:-"DeepSeek-V3-N4"}
MCORE_RELEASE_NUM=${MCORE_RELEASE_NUM:-"0.9"}
WORKSPACE=${WORKSPACE:-"/lustre/raplab/client/congliangx/workspace/scripts/submit_deepseek"} # Path to the Launcher-Scripts
IMAGE=${IMAGE:-"/lustre/raplab/client/congliangx/workspace/images/ds-vince.sqsh"} # Path to sqsh or docker image url
MEGATRON_PATH=${MEGATRON_PATH:-"/lustre/raplab/client/congliangx/workspace/scripts/Megatron-LM"} # Path to Megatron-LM

# some model default config
case $MODEL in
    DeepSeek-V3 )
        export TP=${TP:-"1"} PP=${PP:-"16"} EP=${EP:-"64"} CP=${CP:-"1"} VPP=${VPP:-"1"} MBS=${MBS:-"1"} GBS=${GBS:-"512"} \
        SEQ_LEN=${SEQ_LEN:-"4096"} MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-"alltoall"} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-"true"} \
        PRETRAIN=${PRETRAIN:-"1"}  PP_FIRST=${PP_FIRST:-"4"} PP_LAST=${PP_LAST:-"1"} NODES=${NODES:-"128"}
        ;;
    DeepSeek-V3-N1 )
        export TP=${TP:-"1"} PP=${PP:-"1"} EP=${EP:-"8"} CP=${CP:-"1"} VPP=${VPP:-"1"} MBS=${MBS:-"1"} GBS=${GBS:-"512"} \
        SEQ_LEN=${SEQ_LEN:-"4096"} MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-"alltoall"} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-"true"} \
        PRETRAIN=${PRETRAIN:-"1"} NODES=${NODES:-"1"}
        ;;
    DeepSeek-V3-N4 )
        export TP=${TP:-"1"} PP=${PP:-"2"} EP=${EP:-"16"} CP=${CP:-"1"} VPP=${VPP:-"1"} MBS=${MBS:-"1"} GBS=${GBS:-"512"} \
        SEQ_LEN=${SEQ_LEN:-"4096"} MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-"alltoall"} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-"true"} \
        PRETRAIN=${PRETRAIN:-"1"}  NODES=${NODES:-"4"}
        ;;
    DeepSeek-V3-N8 )
        export TP=${TP:-"1"} PP=${PP:-"2"} EP=${EP:-"32"} CP=${CP:-"1"} VPP=${VPP:-"1"} MBS=${MBS:-"1"} GBS=${GBS:-"512"} \
        SEQ_LEN=${SEQ_LEN:-"4096"} MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-"alltoall"} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-"true"} \
        PRETRAIN=${PRETRAIN:-"1"}  NODES=${NODES:-"8"}
        ;;    
    DeepSeek-V3-N16 )
        export TP=${TP:-"1"} PP=${PP:-"2"} EP=${EP:-"64"} CP=${CP:-"1"} VPP=${VPP:-"1"} MBS=${MBS:-"1"} GBS=${GBS:-"512"} \
        SEQ_LEN=${SEQ_LEN:-"4096"} MOE_TOKEN_DISPATCHER=${MOE_TOKEN_DISPATCHER:-"alltoall"} MOE_GROUPED_GEMM=${MOE_GROUPED_GEMM:-"true"} \
        PRETRAIN=${PRETRAIN:-"1"}  NODES=${NODES:-"16"}
        ;;
esac

# Set training parameters, use all passed command line arguments if not pre-defined
TRAINING_PARAMS=${TRAINING_PARAMS:-$@}

# FP8 arguments
PR=${PR:-fp8}
if [ ${PR} = "bf16" ]; then
    :
elif [ ${PR} = "fp8" ]; then
    TRAINING_PARAMS="$TRAINING_PARAMS --fp8-format hybrid --fp8-amax-history-len 1024 --fp8-amax-compute-algo max"
    MOE_GROUPED_GEMM="true"
else
    echo "Error: The valid values for PR are 'bf16' or 'fp8'. Current value: ${PR}."
    exit 1
fi

if [ $TP -gt 1 ]; then
    export SP="true"
else
    export SP="false"
fi

export WANDB_API_KEY="${WANDB_API_KEY:-"fdfdd86c85adb270b7bd4f9f858dac5e61fe49ff"}"
export WANDB_PROJECT=${WANDB_PEOJECT:-"DS"}
export COMMENT=${COMMENT:-"mtp-qat6"}

# Benchmarking paths
TRAINING_SCRIPT_PATH="${MEGATRON_PATH}/pretrain_gpt.py"
TRAINING_PARAMS_PATH="${WORKSPACE}/model_configs/${MODEL}.yaml"
export DATA_PATH=${DATA_PATH:-"/lustre/raplab/client/congliangx/workspace/data/Slimpajama"}
export OUTPUT_PATH="${WORKSPACE}/output/${MODEL}-${COMMENT}"



cat $TRAINING_PARAMS_PATH | envsubst >${TRAINING_PARAMS_PATH}.tmp

# Extract training params to export
TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | select(.value != "false") | with(select(.value == "true"); .value = "") | [.key + " " + .value] | join("")' ${TRAINING_PARAMS_PATH}.tmp | sed "s/(\([^)]*\))/'(\1)'/g" |tr '\n' ' ')
export TRAINING_PARAMS="$TRAINING_PARAMS $TRAINING_PARAMS_FROM_CONFIG"
rm ${TRAINING_PARAMS_PATH}.tmp
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

export TRAINING_CMD="${PROFILE_CMD} python $TRAINING_SCRIPT_PATH $TRAINING_PARAMS"

#Training environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0
export NVTE_FUSED_ATTN=1
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
export QAT_PARAMS=6

# SLURM settings
# ACCOUNT=${ACCOUNT:-"accountname"}                 # set the account name of your slurm user.
PARTITION=${PARTITION:-"h20q"}                         # set the partition name of your slurm cluster
RUN_TIME="01-00:00:00"
MOUNTS="/lustre/raplab/client/congliangx/workspace:/lustre/raplab/client/congliangx/workspace,/lustre/raplab/client/congliangx/workspace/tmp:/home/congliangx/"                              # set the mount path of your system , should include the launcher path and the megatron path
SLURM_LOGS="$OUTPUT_PATH/slurm_logs"
mkdir -p $SLURM_LOGS


set +e
cat > ${MODEL}.sub <<EOF
#!/bin/bash
#SBATCH --nodes=$NODES
#SBATCH --partition $PARTITION
#SBATCH --ntasks-per-node=8
#SBATCH --time $RUN_TIME
#SBATCH --job-name=${MODEL}_${COMMENT}
#SBATCH --dependency=singleton
#SBATCH --output=${OUTPUT_PATH}/output_${COMMENT}.log
#SBATCH --exclusive
#SBATCH --gres=gpu:8

# Prepare SLURM job
echo "SLURM_JOB_ID=\${SLURM_JOB_ID}" > "$SLURM_LOGS/\${SLURM_JOB_ID}.log"

# to get the master addr, port, world size, no need in eos, didn't find out why.
nodes=(\$(scontrol show hostnames "\$SLURM_JOB_NODELIST"))
master_node=\${nodes[0]}
master_addr=\$(srun --nodes=1 --ntasks=1 -w "\$master_node" hostname --ip-address)
export MASTER_ADDR=\$master_addr
export MASTER_PORT=12345
export WORLD_SIZE=\$(expr 8 \* \$SLURM_JOB_NUM_NODES)

srun \
    --mpi=pmix -l \
    --ntasks-per-node=8 \
    --container-image=${IMAGE} \
    --container-mounts=${MOUNTS} \
    --container-workdir=${WORKSPACE} \
    $TRAINING_CMD | tee "$SLURM_LOGS/\${SLURM_JOB_ID}_${COMMENT}.log" 
EOF

sbatch ${MODEL}.sub
set -e
