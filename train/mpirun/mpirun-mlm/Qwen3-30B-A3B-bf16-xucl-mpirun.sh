#!/bin/bash
set -x

export TRITON_CACHE_DIR=/tmp/triton_cache_${SLURM_NODEID}
#export NCCL_DEBUG=INFO 
#export NCCL_NVLS_ENABLE=1
#export MELLANOX_VISIBLE_DEVICES=all
#export MELLANOX_MOUNT_DRIVER=1
#export NCCL_SOCKET_IFNAME=ens255f0np0
export PYTHONPATH=/raid/jg/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=0
export NVTE_BWD_LAYERNORM_SM_MARGIN=0
export NVTE_FUSED_ATTN=1
export NVTE_DEBUG=0

export MASTER_ADDR=localhost
export MASTER_PORT=12345
export TP_SIZE=1
export PP_SIZE=1
export EP_SIZE=4
export MBS=2
export GBS=256


WRAPPER_SCRIPT=$(mktemp)
cat > $WRAPPER_SCRIPT << 'EOFWRAPPER'
#!/bin/bash
# Map MPI rank variables to PyTorch/Megatron expected variables
set -x 
export RANK=${OMPI_COMM_WORLD_RANK:-${PMI_RANK:-0}}
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:-${PMI_SIZE:-8}}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${RANK}}}
exec python -u "$@"
EOFWRAPPER
chmod +x $WRAPPER_SCRIPT

# Export MPI variables and run with wrapper
mpirun -np 8 --allow-run-as-root \
    $WRAPPER_SCRIPT /raid/jg/Megatron-LM/pretrain_gpt.py \
        --tensor-model-parallel-size $TP_SIZE \
        --pipeline-model-parallel-size $PP_SIZE \
        --expert-model-parallel-size $EP_SIZE \
        --context-parallel-size 1 \
        --expert-tensor-parallel-size 1 \
        --sequence-parallel  \
        --micro-batch-size $MBS \
        --global-batch-size $GBS \
        --manual-gc  --manual-gc-interval 50 \
        --overlap-grad-reduce  --overlap-param-gather \
        --no-create-attention-mask-in-dataloader  --use-mcore-models  \
        --use-flash-attn  \
        --disable-bias-linear  \
        --transformer-impl transformer_engine \
        --tokenizer-type HuggingFaceTokenizer \
        --tokenizer-model /raid/jg/Qwen3-30B-A3B \
        --mock-data  --split 99,1,0 --no-mmap-bin-files  --num-workers 6 --untie-embeddings-and-output-weights  \
        --position-embedding-type rope \
        --rotary-percent 1.0 --rotary-base 1000000 --rotary-seq-len-interpolation-factor 1 \
        --normalization RMSNorm --swiglu  \
        --norm-epsilon 1e-06 \
        --num-layers 32 \
        --hidden-size 2048 \
        --ffn-hidden-size 6144 \
        --num-attention-heads 32 \
        --group-query-attention  \
        --num-query-groups 4 \
        --kv-channels 128 \
        --qk-layernorm  \
        --num-experts 128 \
        --moe-ffn-hidden-size 768 \
        --moe-router-force-load-balancing  \
        --moe-router-topk 8 \
        --moe-grouped-gemm  \
        --moe-aux-loss-coeff 1e-3 \
        --moe-token-dispatcher-type alltoall \
        --moe-permute-fusion  \
        --moe-router-dtype fp32 \
        --seq-length 4096 \
        --max-position-embeddings 4096 \
        --make-vocab-size-divisible-by 1187 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 --clip-grad 1.0 --weight-decay 0.1 \
        --lr-decay-iters 430000 \
        --lr 1.2e-4 --min-lr 1.2e-5 --lr-decay-style cosine \
        --use-distributed-optimizer  \
        --adam-beta1 0.9 --adam-beta2 0.95 \
        --train-iters 80 --exit-duration-in-mins 230 \
        --eval-iters 1 \
        --auto-detect-ckpt-format  \
        --dist-ckpt-strictness log_all \
        --exit-interval 80 \
        --distributed-timeout-minutes 60 \
        --init-method-std 0.02 \
        --log-num-zeros-in-grad  \
        --log-params-norm  \
        --log-throughput  \
        --log-interval 1 \
        --bf16  \
        --fp8-recipe blockwise \
        --fp8-format e4m3 \
        --fp8-param-gather \
        2>&1 | tee -a ./logs/TP${TP_SIZE}PP${PP_SIZE}EP${EP_SIZE}MBS${MBS}GBS${GBS}-nccl.log
