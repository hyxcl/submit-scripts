for i in 1024
do
    python trtllm_serve_1node-engine.py --input_output '[1:1024]' --max_bs $i --port 36888 --tp_list '1' --concurrencies $i --max_num_tokens 4096 --model_dir /raid/model/DeepSeek-R1-Distill-Llama-8B/bs512_token4096_seq2048 --output_dir ./result
done

# for i in 512 1024
# do
#     python trtllm_serve_1node-engine.py --input_output '[1:1024]' --max_bs $i --port 36888 --tp_list '1' --concurrencies $i --max_num_tokens 8192 --model_dir /raid/model/DeepSeek-R1-Distill-Llama-8B/bs1024_token8192_seq2048 --output_dir ./result
# done