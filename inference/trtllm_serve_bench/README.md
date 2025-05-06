Need to install SGLANG in TRT-LLM container for using sglang.bench_serving:
pip install sglang


Example commands:

one node:
python trtllm_serve_1node.py --input_output '[1000:1000]' --max_bs '128' --num_prompts '256' --port 36889 --tp_list '8' --concurrencies '64' --model_dir  /workspace/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f  \
--extra_config ./extra-llm-api-config_large_bs.yml --output_dir ./result

two nodes:
python trtllm_serve_2nodes.py --input_output '[1000:1000]' --max_bs '128' --num_prompts '256' --port 36889 --tp_list '16' --concurrencies '64' --model_dir  /workspace/hub/models--deepseek-ai--DeepSeek-R1/snapshots/8a58a132790c9935686eb97f042afa8013451c9f  \
--extra_config ./extra-llm-api-config_large_bs.yml --output_dir ./result

From: https://gitlab-master.nvidia.com/jungu/qwq-bench-tool/-/tree/main/sglang-bench-trtllm-backend?ref_type=heads