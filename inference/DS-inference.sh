#!/bin/bash
## Setup path 
set -x
DATA_GEN_PATH=/home/congliangx/data/inference/
MODEL=${MODEL:-"Llama-70B"}
CHECKPOINT_DIR=/home/congliangx/models/DeepSeek-R1-Distill-${MODEL}
ENGINE_DIR=${CHECKPOINT_DIR}/engine/
input_len=${input_len:-"4000"}
output_len=${output_len:-"4000"}
concurrency=${concurrency:-"128"}
TP=${TP:-"4"}
LOG_PATH=/home/congliangx/log
MAX_BS=${MAX_BS-"1024"}

## Engine build (set max_num_tokens carefully, it will impact the kV cache and benchmark performance)
function build_engine {
	trtllm-bench --model /home/congliangx/models/DeepSeek-R1-Distill-${MODEL}  build --quantization FP8 --max_batch_size $MAX_BS --max_seq_len 4096 --max_num_tokens 8192 --tp_size $TP 
	mv /home/congliangx/models/DeepSeek-R1-Distill-${MODEL}/tp_${TP}_pp_1 ${ENGINE_DIR}/FP8_BS${MAX_BS}_ISL${input_len}_tp_${TP}_pp_1
}
## Use 'concurrency' to test the performance of different concurrency number
## Data generation
function data_gen_concurrency {
    python benchmarks/cpp/prepare_dataset.py --output ${DATA_GEN_PATH}/${MODEL}-token-request-${concurrency}-norm-dist-${input_len}-${output_len}.json \
    --tokenizer $CHECKPOINT_DIR \
    token-norm-dist \
    --num-requests $((8*concurrency)) \
    --input-mean ${input_len} \
    --input-stdev 0 \
    --output-mean ${output_len} \
    --output-stdev 0
}

function benchmark_concurrency {
    mpirun --allow-run-as-root -n $TP /app/tensorrt_llm/benchmarks/cpp/gptManagerBenchmark \
        --engine_dir $ENGINE_DIR/FP8_BS${MAX_BS}_ISL${input_len}_tp_${TP}_pp_1 \
        --dataset ${DATA_GEN_PATH}/${MODEL}-token-request-${concurrency}-norm-dist-${input_len}-${output_len}.json \
        --streaming \
        --log_level warning \
        --warm_up 2 \
        --kv_cache_free_gpu_mem_fraction 0.95 \
        --concurrency ${concurrency} \
        | tee -a $LOG_PATH/gptmanager_benchmark_${MODEL}_${input_len}_${output_len}_concurrency${concurrency}.log
}

build_engine

data_gen_concurrency

benchmark_concurrency
