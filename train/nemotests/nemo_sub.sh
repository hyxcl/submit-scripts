#!/bin/bash

# Parameters
#SBATCH --account=general_sa
#SBATCH --dependency=singleton
#SBATCH --exclusive
#SBATCH --job-name=general_sa:llama
#SBATCH --mem=0
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=0-04:00:00
#SBATCH --partition=batch

# setup
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0 # Diable FP8 AMAX reduction in the data-parallel domain
export NVTE_ASYNC_AMAX_REDUCTION=1    # Enable asynchronous FP8 AMAX reduction
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export HYDRA_FULL_ERROR=1
export NVTE_FUSED_ATTN=0   #Disable cudnn FA until we've tested it more
export DIR="/lustre/fsw/general_sa/vince/nemotests"
export DATADIR="/lustre/fsw/general_sa/vince/data/llama"


: "${MODEL:=llama2_13b}"
: "${TP:=2}"
: "${CP:=2}"
: "${PP:=2}"
: "${SP:=true}"
: "${SEQ_LENGTH:=32768}"
: "${MBS:=1}"
: "${GBS:=64}"
: "${FP8:=true}"

mkdir -p ${DIR}/results/profile_logs
mkdir -p ${DIR}/results/${MODEL}
mkdir -p ${DIR}/log

# command 1
srun --container-image /lustre/fsw/general_sa/vince/images/nemo240301.sqsh --container-mounts /lustre/fsw/general_sa/vince:/lustre/fsw/general_sa/vince --no-container-mount-home --output ${DIR}/log/${MODEL}_TP${TP}_PP${PP}_CP${CP}_MBS${MBS}_GBS${GBS}_SEQL${SEQ_LENGTH}_%j.out --error ${DIR}/log/${MODEL}_TP${TP}_PP${PP}_CP${CP}_MBS${MBS}_GBS${GBS}_SEQL${SEQ_LENGTH}_%j.err  bash -c "
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  nsys profile -s none -t nvtx,cuda -o ${DIR}/results/profile_logs/profile_${MODEL}_TP${TP}PP${PP}CP${CP}_MBS${MBS}GBS${GBS}_\${SLURM_JOB_ID}_node\${SLURM_NODEID}_rank\${SLURM_PROCID} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
  python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=$DIR \
  --config-name=$MODEL.yaml \
  run.results_dir=${DIR} \
  exp_manager.exp_dir=${DIR}/results/${MODEL} \
  exp_manager.explicit_log_dir=${DIR}/results/${MODEL} \
  trainer.num_nodes=${SLURM_JOB_NUM_NODES} \
  model.tensor_model_parallel_size=$TP \
  model.pipeline_model_parallel_size=$PP \
  model.context_parallel_size=$CP \
  model.sequence_parallel=$SP \
  model.micro_batch_size=$MBS \
  model.global_batch_size=$GBS \
  model.max_position_embeddings=$SEQ_LENGTH \
  model.encoder_seq_length=$SEQ_LENGTH \
  model.data.seq_length=$SEQ_LENGTH \
  model.tokenizer.model=${DATADIR}/llama_tokenizer.model \
  model.fp8=${FP8} \
  model.fp8_hybrid=${FP8}
  "
