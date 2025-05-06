import os
import subprocess
import time
import requests
import argparse

def get_cmd(model_path, extra_config, tp, max_prefill_tokens, max_running_requests, port):
    cmd = [
        "trtllm-serve",str(model_path),
        '--host', '0.0.0.0', '--port', str(port),
        "--backend", "pytorch", "--tp_size", str(tp), 
        "--ep_size", '8', "--pp_size", '1',
        "--max_batch_size", str(max_running_requests),
        "--max_num_tokens", str(max_prefill_tokens),
        "--extra_llm_api_options", str(extra_config),
        "--kv_cache_free_gpu_memory_fraction", '0.9'
    ]

    return cmd

def start_server(model_path, extra_config, tp, max_prefill_tokens, max_running_requests, port):
    print(f"Starting server with config: {model_path}, TP={tp}, Max Prefill={max_prefill_tokens}, Max Requests={max_running_requests}, Port={port}")
    #cmd = get_cmd(model_path, extra_config, tp, max_prefill_tokens, max_running_requests, port)
    #res = subprocess.Popen(cmd, env=os.environ.copy())
    #pid = res.pid

    serve_cmd = get_cmd(model_path, extra_config, tp, max_prefill_tokens, max_running_requests, port)
    mpirun_prefix = [
        "/usr/local/mpi/bin/mpirun",
        "-np", "16",
        "--hostfile", "/workspace/torch_bench/my_hostfile",
        "-mca", "plm_rsh_args", '"-p 12233"',
        "--allow-run-as-root",
        "trtllm-llmapi-launch"
    ]
    full_cmd = mpirun_prefix + serve_cmd
    #print("启动命令：", " ".join(full_cmd))
    print("启动命令：", " ".join(str(x) for x in full_cmd))
    res = subprocess.Popen(full_cmd, env=os.environ.copy())
    pid = res.pid

    # 等待服务器健康检查通过
    while True:
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            if response.status_code == 200:
                break
        except:
            pass
        time.sleep(2)
    print("Server is ready!")
    return pid


def benchmark(model_path, num_prompt, port, ISL, OSL, max_concurrency, output_file):
    test_cmd = [
                'python3','-m','sglang.bench_serving','--backend','sglang-oai',
                '--dataset-name','random', "--model", str(model_path),
                '--num-prompt',
                f'{num_prompt}',
                '--random-input',
                f'{ISL}',
                '--random-output',
                f'{OSL}',
                '--max-concurrency',
                f'{max_concurrency}',
                '--random-range-ratio','1','--host','127.0.0.1',
                '--port',f'{port}','--output-file',f'{output_file}'
                ]
    subprocess.run(test_cmd,env=os.environ.copy())




def main():
    parser = argparse.ArgumentParser(description="Run sglang server and benchmark.")
    parser.add_argument(
        '--tp_list',
        type=str,
        required=False,
        default='2,4',
        help="List of tensor parallelism values (comma-separated)"
    )
    parser.add_argument('--input_output', type=str, required=True, help="Input/output pairs in format '[ISL:OSL],...'")
    parser.add_argument('--concurrencies', type=str, required=True, help="List of concurrency levels (e.g., '1,4,8')")
    parser.add_argument('--max_bs', type=str, required=True, help="List of max bs levels (e.g., '1,4,8')")
    parser.add_argument('--port', type=int, required=True, help="Server port")
    parser.add_argument('--model_dir', type=str, required=True, help="Model path")
    parser.add_argument('--extra_config', type=str, default="./extra_llm_api_options.yaml", help="TRT-LLM's Extra LLM Api Options")
    parser.add_argument('--max_running_request', type=int, default=0, help="the max batch size")
    parser.add_argument('--num_prompts', type=int, default=0, help="the num requests must larger than 32")
    parser.add_argument('--output_dir', type=str, required=True, help="Benchmark results output path")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse input_output into list of tuples
    input_output_pairs = []
    if args.input_output.startswith('[') and args.input_output.endswith(']'):
        input_output_str = args.input_output[1:-1]
        input_output_pairs = []
        for pair_str in input_output_str.replace('],', ',').replace('[', '').replace(']', '').split(','):
            if pair_str.strip() == '':
                continue
            isl_str, osl_str = pair_str.split(':')
            input_output_pairs.append((int(isl_str), int(osl_str)))
    else:
        # Fallback to manual parsing
        input_output_pairs = []
        for pair_str in args.input_output.split('],['):
            if not pair_str.strip():
                continue
            isl_str, osl_str = pair_str.split(':')
            input_output_pairs.append((int(isl_str), int(osl_str)))

    # Parse concurrencies into list
    concurrencies = list(map(int, args.concurrencies.split(',')))

    #Parse concurrencies into list
    max_bs = list(map(int, args.max_bs.split(',')))

    # Parse tp_list
    tp_list = list(map(int, args.tp_list.split(','))) if args.tp_list else [2, 4]

    # Server port
    port = args.port
    pid = -1

    # Process each combination
    for tp in tp_list:
        for isl, osl in input_output_pairs:
            for concurrency in concurrencies:
                for bs in max_bs:
                    # Calculate max_prefill_tokens
                    max_prefill = 8192 # 可以通过参数控制
                    max_running = bs
                    # 启动服务器
                    os.popen('pkill trtllm-serve')
                    if pid > 0:
                        os.popen(f'kill -9 {pid}')
                    time.sleep(10)
                    #pid = start_server(args.model_dir, args.extra_config, tp, max_prefill, max_running, port)
                    pid = start_server(args.model_dir, args.extra_config, tp, isl+bs-1, max_running, port)
                    # 执行基准测试
                    #if args.num_prompts == 0:
                    #    num_prompts = max(concurrency*4, 32)
                    #else:
                        #num_prompts = max(args.num_prompts, 32)
                    num_prompts = 4 * concurrency if 4 * concurrency < 4096 else 4096
                    output_filename = f"{args.output_dir}/results_tp_{tp}_isl_{isl}_osl_{osl}_num_prompts_{num_prompts}.out"
                    with open(output_filename,'a') as fw:
                        fw.write(f'max_prefill:{max_prefill},max_running_requests:{max_running},num_prompts:{num_prompts}\n')
                    benchmark(args.model_dir, num_prompts, port, isl, osl, concurrency, output_filename)

                    # 等待一段时间防止端口冲突（可选）
                    time.sleep(5)
        port += 1
    
    os.popen('pkill trtllm-serve')

if __name__ == "__main__":
    main()

