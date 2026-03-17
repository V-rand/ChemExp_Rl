#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版：化学合成数据转换为 ms-swift 格式（无用量信息）
适用于输入数据不包含用量提示的场景
"""

import json
import re
import os
from typing import Dict, List, Optional
from pathlib import Path

# ================= 基础工具函数 =================
try:
    from rdkit import Chem, rdBase
    rdBase.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    Chem = None

def canonicalize_smiles(smiles: str) -> str:
    """RDKit标准化SMILES（可选）"""
    if not HAS_RDKIT:
        return smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        pass
    return smiles.strip()

def extract_smiles_from_mol_tag(text: str) -> Optional[str]:
    """从[MOL] ```SMILES``` [/MOL]格式中提取SMILES（用于验证）"""
    m = re.search(r'\[MOL\]\s*```(.*?)```\s*\[/MOL\]', text, re.DOTALL)
    return m.group(1).strip() if m else None

# ================= 核心转换逻辑 =================

SYSTEM_PROMPT = """You are a chemical synthesis expert. Design a precise experimental procedure from the provided reaction information.
#INPUT
Reactants, product, solvent, catalyst will be given.
#OUTPUT
Return exactly two sections and nothing else:

<thinking>
Determine the reaction mechanism from the substrate structures, rationalize the reagent and solvent choices chemically, explain the condition optimization, then outline the experimental workflow. End with: "Therefore, the validated operational sequence is:" followed by the step sequence.
</thinking>

<answer>
Robot-executable procedure with exactly the steps.
</answer>

#EXECUTION RULES
• Each step must start with one action from:
ADD, STIR, WAIT, CONCENTRATE, YIELD, MAKESOLUTION, FILTER, WASH, DRYSOLUTION, COLLECTLAYER, EXTRACT, SETTEMP, REFLUX, RECRYSTAL, PHASESEPA, PH, PURIFY, QUENCH, PARTITION, TRITURATE, DRYSOLID, DEGAS, MICROWAVE, SONICATE
• Each step must be wrapped in <procedure>...</procedure>
• Time format: <time>...</time>
• Temperature format: <temp>...</temp>
• Use [MOL] ```SMILES``` [/MOL] for chemicals.if need quantity, add (Quantity) after the tag, e.g. [MOL] ```SMILES``` [/MOL] (x g, x mol)
• Only one physical action per <procedure>.
- When preparing multiple solutions separately in an experiment, name them as (Solution A), (Solution B), etc.Clearly specify when referencing them later. Example: MAKESOLUTION (Solution A) with ...; MAKESOLUTION (Solution B) with ...; ADD Solution B to Solution A; 
• Final step must be YIELD.
"""

def build_simple_user_prompt(item: Dict) -> str:
    """
    构建简化的 User Prompt，不包含任何用量信息
    仅列出化学物质和反应条件
    """
    parts = ["Please design a chemical experiment based on the following reaction information. Let's build step by step:"]
    
    # 1. 反应物 - 直接使用原始格式，不提取用量
    reactants = item.get("REACTANT", [])
    if reactants:
        # 清理并标准化显示
        clean_reactants = []
        for r in reactants:
            # 可选：标准化 SMILES（不改变格式，只规范化内容）
            smiles = extract_smiles_from_mol_tag(r)
            if smiles and HAS_RDKIT:
                canon = canonicalize_smiles(smiles)
                # 替换原始SMILES为标准化的（可选）
                r = r.replace(f"```{smiles}```", f"```{canon}```")
            clean_reactants.append(r)
        parts.append(f"Reactants: {', '.join(clean_reactants)}")
    
    # 2. 产物 - 直接使用原始PRODUCT，不提取ACTION中的产率
    products = item.get("PRODUCT", [])
    if products:
        clean_products = []
        for p in products:
            smiles = extract_smiles_from_mol_tag(p)
            if smiles and HAS_RDKIT:
                canon = canonicalize_smiles(smiles)
                p = p.replace(f"```{smiles}```", f"```{canon}```")
            clean_products.append(p)
        parts.append(f"Product: {', '.join(clean_products)}")
    
    # 3. 催化剂（如果有）
    catalysts = item.get("CATALYST", [])
    if catalysts:
        parts.append(f"Catalyst: {', '.join(catalysts)}")
    
    # 4. 溶剂（如果有）
    solvents = item.get("SOLVENT", [])
    if solvents:
        parts.append(f"Solvent: {', '.join(solvents)}")
    
    return "\n".join(parts)

def build_assistant_content(item: Dict, include_thinking: bool = True) -> str:
    """
    构建 Assistant Content
    保持与之前相同：thinking + answer 格式
    """
    action_text = item.get("ACTION", "").strip()
    thinking_text = item.get("Thinking_text", "").strip()
    
    if not include_thinking or not thinking_text:
        # 如果不包含thinking，直接返回action（通常包含<answer>标签）
        return action_text
    
    # 组合格式
    if "<answer>" in action_text:
        return f"<thinking>\n{thinking_text}\n</thinking>\n\n{action_text}"
    else:
        return f"<thinking>\n{thinking_text}\n</thinking>\n\n<answer>\n{action_text}\n</answer>"

def convert_item_to_swift_format(item: Dict, include_thinking: bool = True) -> Dict:
    """
    将单个数据项转换为 ms-swift 消息格式（简化版）
    """
    user_content = build_simple_user_prompt(item)
    assistant_content = build_assistant_content(item, include_thinking)
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def process_file(input_path: str, output_path: str, include_thinking: bool = True):
    """
    处理整个 JSONL 文件
    """
    input_file = Path(input_path)
    output_file = Path(output_path)
    
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    total = 0
    success = 0
    error_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            total += 1
            
            try:
                item = json.loads(line)
                swift_item = convert_item_to_swift_format(item, include_thinking)
                fout.write(json.dumps(swift_item, ensure_ascii=False) + '\n')
                success += 1
                
                if line_num % 100 == 0:
                    print(f"已处理 {line_num} 条数据...")
                    
            except json.JSONDecodeError as e:
                print(f"[错误] 第 {line_num} 行 JSON 解析失败: {e}")
                error_count += 1
            except Exception as e:
                print(f"[错误] 第 {line_num} 行处理失败: {e}")
                error_count += 1
    
    print(f"\n处理完成: 总计 {total}, 成功 {success}, 失败 {error_count}")
    print(f"输出文件: {output_file.absolute()}")

if __name__ == "__main__":
    # 配置路径
    INPUT_FILE = "/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/raw/test_200.jsonl"
    
    # 输出文件路径
    OUTPUT_FILE = "/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed/ms_swift_eval_200.jsonl"
    
    print("=" * 60)
    print("简化版转换：无用量信息的 ms-swift 格式")
    print("=" * 60)
    
    # 生成格式（带 thinking，但输入无用量）
    process_file(INPUT_FILE, OUTPUT_FILE, include_thinking=True)
    
    print("\n" + "=" * 60)
    print("转换完成！输入中不包含具体用量信息")
    print(f"输出: {OUTPUT_FILE}")
    print("=" * 60)