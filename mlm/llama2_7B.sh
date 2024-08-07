#!/bin/bash
set -x
DATA_PATH=/mnt/fs/nemofw/llama_data/llama/
DATASET="1.0 ${DATA_PATH}/my-llama_00_text_document"
TOKENIZER_MODEL="/mnt/fs/nemofw/llama_data/llama/llama_tokenizer.model"

MBS=1
GBS=64
TP_SIZE=1
PP_SIZE=4
CP_SIZE=1

MODEL_ARGS=(
    --use-mcore-models
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 4096
    --init-method-std 0.01
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --apply-query-key-layer-scaling
    --normalization RMSNorm
    --disable-bias-linear
    --swiglu
    --use-rotary-position-embeddings
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
)


TRAINING_ARGS=(
    --train-iters 5000
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --lr 0.0001
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --lr-decay-style cosine
    --lr-warmup-fraction .001
    --lr-decay-iters 430000
    --min-lr 1.0e-5
)

PRECISION_ARGS=(
    --bf16
    --fp8-format hybrid
    --fp8-amax-history-len 1024
    --fp8-amax-compute-algo max
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP_SIZE}
    --pipeline-model-parallel-size ${PP_SIZE}
    --context-parallel-size ${CP_SIZE}
    --sequence-parallel
    --transformer-impl transformer_engine
    --use-flash-attn
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

DATA_ARGS=(
    --data-path ${DATASET}
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --split 90,5,5
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 10
    --save-interval 10000
    --eval-interval 1000
    --eval-iters 10
    --no-load-optim
    --no-load-rng
)

#python -u /lustre/fsw/coreai_dlalgo_llm/vince/Megatron-LM/pretrain_gpt.py \
##### bellow environment is no need in EOS, not sure why need to set it in customer's cluster.
#node=`echo $SLURM_JOB_NODELIST | sed 's/node\[//g'|awk -F '-' '{print $1}'|awk -F ',' '{print $1}'|sed 's/node//g'`
#export MASTER_ADDR=172.16.1.${node}
#export MASTER_PORT=6668
#export WORLD_SIZE=$(expr 8 \* $SLURM_JOB_NUM_NODES)

#python -u /opt/megatron-lm/pretrain_gpt.py \
nsys profile -s none -t nvtx,cuda -o /mnt/fs/nemofw/xucl/mlm/test --capture-range=cudaProfilerApi --capture-range-end=stop python -u /opt/megatron-lm/pretrain_gpt.py \
	--profile \
        ${MODEL_ARGS[@]} \
        ${PRECISION_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_ARGS[@]} \
        ${EVAL_AND_LOGGING_ARGS[@]}
