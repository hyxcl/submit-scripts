#!/bin/bash
# setup
export TRANSFORMERS_OFFLINE=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NVTE_DP_AMAX_REDUCE_INTERVAL=0 # Diable FP8 AMAX reduction in the data-parallel domain
export NVTE_ASYNC_AMAX_REDUCTION=1    # Enable asynchronous FP8 AMAX reduction
export CUDA_DEVICE_MAX_CONNECTIONS=1 
export HYDRA_FULL_ERROR=1
export NVTE_FUSED_ATTN=0   #Disable cudnn FA until we've tested it more
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_4
export GLOO_SOCKET_IFNAME=ens19f0
export NCCL_SOCKET_IFNAME=ens19f0
export DIR="/mnt/fs/nemofw/xucl/2405/tpcommoverlap"
export DATADIR="/mnt/fs/nemofw/llama_data/llama/"
export NUM_NODES=2
export NCCL_DEBUG=INFO

: "${MODEL:=llama2_7b}"
: "${TP:=1}"
: "${CP:=1}"
: "${PP:=1}"
: "${SP:=true}"
: "${SEQ_LENGTH:=4096}"
: "${MBS:=8}"
: "${GBS:=64}"
: "${FP8:=true}"
: "${TPOVERLAP:=true}"


mkdir -p ${DIR}/results/profile_logs
mkdir -p ${DIR}/results/${MODEL}
mkdir -p ${DIR}/log

# command 1
  #nsys profile -s none -t nvtx,cuda -o ${DIR}/results/profile_logs/profile_${MODEL}_TP${TP}PP${PP}CP${CP}_MBS${MBS}GBS${GBS}_\${SLURM_JOB_ID}_node\${SLURM_NODEID}_rank\${SLURM_PROCID} --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
mpirun -np 16 -H h20n4:8,h20n5:8 -x NVIDIA_PYTORCH_VERSION -x LD_LIBRARY_PATH  -x PYTHONPATH -x PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION -x TRANSFORMERS_OFFLINE -x TORCH_NCCL_AVOID_RECORD_STREAMS -x NVTE_DP_AMAX_REDUCE_INTERVA -x NVTE_ASYNC_AMAX_REDUCTION -x CUDA_DEVICE_MAX_CONNECTIONS -x HYDRA_FULL_ERROR -x NVTE_FUSED_ATTN -x NCCL_IB_HCA -x GLOO_SOCKET_IFNAME -x NCCL_SOCKET_IFNAME -x NCCL_DEBUG --allow-run-as-root bash -c "
  cd /opt/NeMo;
  git rev-parse HEAD;
  export PYTHONPATH=/opt/NeMo:\${PYTHONPATH};
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  python3 -u /opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py  \
  --config-path=$DIR \
  --config-name=$MODEL.yaml \
  run.results_dir=${DIR} \
  exp_manager.exp_dir=${DIR}/results/${MODEL}_TP${TP}PP${PP}CP${CP} \
  exp_manager.explicit_log_dir=${DIR}/results/${MODEL}_TP${TP}PP${PP}CP${CP} \
  trainer.num_nodes=${NUM_NODES} \
  model.tensor_model_parallel_size=$TP \
  model.pipeline_model_parallel_size=$PP \
  model.context_parallel_size=$CP \
  model.sequence_parallel=$SP \
  model.ub_tp_comm_overlap=${TPOVERLAP} \
  model.micro_batch_size=$MBS \
  model.global_batch_size=$GBS \
  model.max_position_embeddings=$SEQ_LENGTH \
  model.encoder_seq_length=$SEQ_LENGTH \
  model.data.seq_length=$SEQ_LENGTH \
  model.tokenizer.model=${DATADIR}/llama_tokenizer.model \
  model.fp8=${FP8} \
  model.fp8_hybrid=${FP8}
  " | tee -a ${DIR}/log/2N_70b_20L_overlap_${TPOVERLAP}.log
