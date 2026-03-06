# 数据处理脚本 - 将原始 JSONL 转换为符合 VERL Schema 的 Parquet 文件
import json
import re
import argparse
import random  
from pathlib import Path
from typing import Dict, List
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ================= 配置区 =================
DATA_SOURCE = "chemexp"
ABILITY = "chemical_synthesis"

# ================= 工具函数 =================
def load_system_prompt() -> str:
    """加载System Prompt - 建议放在脚本同级或指定路径"""
    # 路径逻辑：寻找当前脚本父目录下的 configs/system_prompt.txt
    prompt_path = Path(__file__).parent.parent / "configs" / "system_prompt.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"System prompt file not found at: {prompt_path}")
    with open(prompt_path, 'r', encoding='utf-8') as f:
        return f.read().strip()

def extract_final_yield_step(action_text: str) -> str:
    """从 ACTION 提取最后一步 YIELD 的描述 (分子 + 物理量)"""
    if not action_text: return ""
    yield_steps = re.findall(r'<procedure>YIELD\s*(.*?)</procedure>', action_text, re.IGNORECASE | re.DOTALL)
    return yield_steps[-1].strip() if yield_steps else ""

def build_user_prompt(item: Dict) -> str:
    """构建 User Prompt，将产物和产量合并"""
    parts = ["Please design a chemical experiment based on the following requirements:"]
    
    # 1. 反应物
    reactants = item.get("REACTANT", [])
    if reactants:
        parts.append(f"Reactants: {', '.join(reactants)}")
        
    # 2. 目标产物与产量 (直接从 Action 提取)
    final_output = extract_final_yield_step(item.get("ACTION", ""))
    if final_output:
        parts.append(f"Target Product & Yield: {final_output}")
    else:
        products = item.get("PRODUCT", [])
        if products:
            parts.append(f"Product: {', '.join(products)}")
        
    # 3. 辅助化学品
    if item.get("CATALYST"):
        parts.append(f"Catalyst: {', '.join(item['CATALYST'])}")
    if item.get("SOLVENT"):
        parts.append(f"Solvent: {', '.join(item['SOLVENT'])}")
        
    return "\n".join(parts)

def process_single_item(item: Dict, idx: int, system_prompt: str, split: str) -> Dict:
    """转换为符合 VERL Schema 的单条数据"""
    action_text = item.get("ACTION", "")
    actions_list = re.findall(r'<procedure>(.*?)</procedure>', action_text, re.DOTALL)
    
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_prompt(item)}
        ],
        "ability": ABILITY,
        "reward_model": {
            "style": "rule",
            "ground_truth": {
                "thinking": item.get("Thinking_text", ""),
                "actions": [a.strip() for a in actions_list],
                "molecules": json.dumps(item.get("molecules", {}))
            }
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "reaction_id": str(item.get("index", idx))
        }
    }

# ================= 主流转 =================

def convert(input_file: str, output_dir: str, val_ratio: float = 0.1, max_samples: int = None):
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 预加载 System Prompt
    sys_prompt = load_system_prompt()
    
    # 定义 Schema
    schema = pa.schema([
        ('data_source', pa.string()),
        ('prompt', pa.list_(pa.struct([('role', pa.string()), ('content', pa.string())]))),
        ('ability', pa.string()),
        ('reward_model', pa.struct([
            ('style', pa.string()),
            ('ground_truth', pa.struct([
                ('thinking', pa.string()),
                ('actions', pa.list_(pa.string())),
                ('molecules', pa.string())
            ]))
        ])),
        ('extra_info', pa.struct([
            ('split', pa.string()), ('index', pa.int64()), ('reaction_id', pa.string())
        ]))
    ])

    # 1. 快速计算总样本数用于划分
    print("Counting samples...")
    total_in_file = 0
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): total_in_file += 1
    
    effective_total = min(total_in_file, max_samples) if max_samples else total_in_file
    val_count_target = int(effective_total * val_ratio)
    train_count_target = effective_total - val_count_target
    
    print(f"Total samples to process: {effective_total} (Train: {train_count_target}, Val: {val_count_target})")

    # 2. 流式处理与分流写入
    train_writer = pq.ParquetWriter(output_path / "train.parquet", schema)
    val_writer = pq.ParquetWriter(output_path / "val.parquet", schema)
    
    train_batch, val_batch = [], []
    curr_count = 0

    with open(input_path, 'r', encoding='utf-8') as f:
        pbar = tqdm(total=effective_total, desc="Converting")
        for line in f:
            if not line.strip(): continue
            if curr_count >= effective_total: break
            
            try:
                item = json.loads(line)
                # 简单划分逻辑：前 train_count_target 条进训练集，其余进验证集
                if curr_count < train_count_target:
                    split_name = "train"
                    processed = process_single_item(item, curr_count, sys_prompt, split_name)
                    train_batch.append(processed)
                else:
                    split_name = "val"
                    processed = process_single_item(item, curr_count - train_count_target, sys_prompt, split_name)
                    val_batch.append(processed)
                
                curr_count += 1
                pbar.update(1)

                # 批量写入缓存
                if len(train_batch) >= 1000:
                    train_writer.write_table(pa.Table.from_pylist(train_batch, schema=schema))
                    train_batch = []
                if len(val_batch) >= 1000:
                    val_writer.write_table(pa.Table.from_pylist(val_batch, schema=schema))
                    val_batch = []
                    
            except Exception as e:
                print(f"\nError at row {curr_count}: {e}")

    # 3. 写入剩余数据
    if train_batch: train_writer.write_table(pa.Table.from_pylist(train_batch, schema=schema))
    if val_batch: val_writer.write_table(pa.Table.from_pylist(val_batch, schema=schema))
    
    train_writer.close()
    val_writer.close()
    pbar.close()
    print(f"\nDone! Files saved in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--limit", type=int, default=None, help="Max total samples to process")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio (0.0-1.0)")
    args = parser.parse_args()
    
    convert(args.input, args.output_dir, args.val_ratio, args.limit)