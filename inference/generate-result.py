import json
import csv
import os
from natsort import natsorted
import argparse

KEYS_TO_EXTRACT = [
    'concurrency', 'output_throughput', 'input_throughput', 'mean_ttft_ms', 'mean_tpot_ms'
]

# 定义CSV列名（filename + 目标字段）
CSV_HEADERS = ['filename'] + KEYS_TO_EXTRACT

def process_json_files(input_dir, output_file):
    # 获取目录中所有JSON文件
    json_files = natsorted( [f for f in os.listdir(input_dir) if f.endswith('.json')] )
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        writer.writeheader()  # 写入CSV表头
        
        for json_file in json_files:
            file_path = os.path.join(input_dir, json_file)
            row_data = {'filename': json_file}  # 初始化行数据，包含文件名
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 提取目标字段的值（缺失字段设为N/A）
                for key in KEYS_TO_EXTRACT:
                    row_data[key] = data.get(key, 'N/A')
                
                writer.writerow(row_data)  # 写入CSV行
                print(f"成功处理: {json_file}")
                
            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', '-i', required=True, 
                        help='包含JSON文件的输入目录路径')
    parser.add_argument('--output_csv', '-o', default='output.csv',
                        help='output.csv）')
    args = parser.parse_args()
    
    process_json_files(args.input_dir, args.output_csv)
    print(f"\n处理完成！结果已保存至: {args.output_csv}")