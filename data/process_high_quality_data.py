#!/usr/bin/env python3
"""
高质量数据处理脚本 - 从后往前取数据
功能：
1. 从临时文件中处理数据，筛选出符合标准的1000条高质量数据
2. 检查动作是否在24个标准动作中（出现COOL、HEAT等则丢弃）
3. 将常用化学品缩写转换为标准SMILES（使用molecules映射表）
4. 使用RDKit将SMILES标准化
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# ================== RDKit 支持 ==================
try:
    from rdkit import Chem, rdBase
    rdBase.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    Chem = None
    print("警告: RDKit未安装，SMILES验证将受限")


def canonicalize_smiles(smiles: str) -> str:
    """RDKit标准化SMILES"""
    if not HAS_RDKIT:
        return smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        pass
    return smiles.strip()


def is_valid_smiles(smiles: str) -> bool:
    """检查是否是有效SMILES"""
    if not HAS_RDKIT:
        return len(smiles) > 2
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


# ================== 24个标准动作 ==================
ALLOWED_ACTIONS = {
    'ADD', 'STIR', 'WAIT', 'CONCENTRATE', 'YIELD', 'MAKESOLUTION',
    'FILTER', 'WASH', 'DRYSOLUTION', 'COLLECTLAYER', 'EXTRACT',
    'SETTEMP', 'REFLUX', 'RECRYSTAL', 'PHASESEPA', 'PH', 'PURIFY',
    'QUENCH', 'PARTITION', 'TRITURATE', 'DRYSOLID', 'DEGAS',
    'MICROWAVE', 'SONICATE'
}

# 非法动作 - 如果包含这些动作则丢弃数据
ILLEGAL_ACTIONS = {'COOL', 'HEAT'}

# ================== 扩展的化学品缩写映射表 ==================
# 用于处理文本中出现的试剂缩写（不在[MOL]标签内的）
CHEMICAL_ABBREVIATIONS = {
    # 溶剂
    "thf": "C1CCOC1",
    "tetrahydrofuran": "C1CCOC1",
    "dmf": "CN(C)C=O",
    "dimethylformamide": "CN(C)C=O",
    "n,n-dimethylformamide": "CN(C)C=O",
    "dcm": "ClCCl",
    "dichloromethane": "ClCCl",
    "methylene chloride": "ClCCl",
    "meoh": "CO",
    "methanol": "CO",
    "etoh": "CCO",
    "ethanol": "CCO",
    "ether": "CCOCC",
    "diethyl ether": "CCOCC",
    "ethyl ether": "CCOCC",
    "etoac": "CCOC(C)=O",
    "ethyl acetate": "CCOC(C)=O",
    "acetonitrile": "CC#N",
    "mecn": "CC#N",
    "acetone": "CC(=O)C",
    "dmso": "CS(C)=O",
    "dimethyl sulfoxide": "CS(C)=O",
    "toluene": "Cc1ccccc1",
    "benzene": "c1ccccc1",
    "hexane": "CCCCCC",
    "n-hexane": "CCCCCC",
    "heptane": "CCCCCCC",
    "pentane": "CCCCC",
    "dioxane": "C1COCCO1",
    "1,4-dioxane": "C1COCCO1",
    "tbme": "COC(C)(C)C",
    "mtbe": "COC(C)(C)C",
    "methyl tert-butyl ether": "COC(C)(C)C",
    "petroleum ether": "CCCCCC",
    "pet ether": "CCCCCC",
    
    # 碱和无机盐
    "k2co3": "O=C([O-])[O-].[K+].[K+]",
    "potassium carbonate": "O=C([O-])[O-].[K+].[K+]",
    "na2co3": "O=C([O-])[O-].[Na+].[Na+]",
    "sodium carbonate": "O=C([O-])[O-].[Na+].[Na+]",
    "nahco3": "O=C(O)[O-].[Na+]",
    "sodium bicarbonate": "O=C(O)[O-].[Na+]",
    "naoh": "[Na+].[OH-]",
    "sodium hydroxide": "[Na+].[OH-]",
    "koh": "[K+].[OH-]",
    "potassium hydroxide": "[K+].[OH-]",
    "lioh": "[Li+].[OH-]",
    "lithium hydroxide": "[Li+].[OH-]",
    
    # 有机碱
    "et3n": "CCN(CC)CC",
    "tea": "CCN(CC)CC",
    "triethylamine": "CCN(CC)CC",
    "dipea": "CCN(C(C)C)C(C)C",
    "dipa": "CCN(C(C)C)C(C)C",
    "hunig's base": "CCN(C(C)C)C(C)C",
    "n,n-diisopropylethylamine": "CCN(C(C)C)C(C)C",
    "pyridine": "c1ccncc1",
    "pyr": "c1ccncc1",
    
    # 酸
    "tfa": "O=C(O)C(F)(F)F",
    "trifluoroacetic acid": "O=C(O)C(F)(F)F",
    "acetic acid": "CC(=O)O",
    "hoac": "CC(=O)O",
    "acoh": "CC(=O)O",
    "formic acid": "O=CO",
    "hco2h": "O=CO",
    "hcl": "Cl",
    "hydrochloric acid": "Cl",
    "h2so4": "O=S(=O)(O)O",
    "sulfuric acid": "O=S(=O)(O)O",
    
    # 盐类
    "nacl": "[Na+].[Cl-]",
    "sodium chloride": "[Na+].[Cl-]",
    "kcl": "[K+].[Cl-]",
    "potassium chloride": "[K+].[Cl-]",
    "nabr": "[Na+].[Br-]",
    "kbr": "[K+].[Br-]",
    "ki": "[K+].[I-]",
    "potassium iodide": "[K+].[I-]",
    "nh4cl": "[NH4+].[Cl-]",
    "ammonium chloride": "[NH4+].[Cl-]",
    
    # 干燥剂
    "na2so4": "O=S(=O)([O-])[O-].[Na+].[Na+]",
    "sodium sulfate": "O=S(=O)([O-])[O-].[Na+].[Na+]",
    "anhydrous sodium sulfate": "O=S(=O)([O-])[O-].[Na+].[Na+]",
    "mgso4": "O=S(=O)([O-])[O-].[Mg+2]",
    "magnesium sulfate": "O=S(=O)([O-])[O-].[Mg+2]",
    "anhydrous magnesium sulfate": "O=S(=O)([O-])[O-].[Mg+2]",
    "cacl2": "[Ca+2].[Cl-].[Cl-]",
    "calcium chloride": "[Ca+2].[Cl-].[Cl-]",
    "anhydrous calcium chloride": "[Ca+2].[Cl-].[Cl-]",
    
    # 常见试剂
    "water": "O",
    "h2o": "O",
    "ammonia": "N",
    "nh3": "N",
    "ammonium hydroxide": "[NH4+].[OH-]",
    "nh4oh": "[NH4+].[OH-]",
    "brine": "[Na+].[Cl-].O",
    "saturated brine": "[Na+].[Cl-].O",
    "saturated aqueous nacl": "[Na+].[Cl-].O",
    "sat. nacl": "[Na+].[Cl-].O",
    "saturated sodium chloride": "[Na+].[Cl-].O",
    "ice": "O",
    "ice water": "O",
    "hydrogen": "[H][H]",
    "h2": "[H][H]",
    "oxygen": "O=O",
    "o2": "O=O",
    "nitrogen": "N#N",
    "n2": "N#N",
    
    # 常用试剂缩写
    "edci": "CCN=C=NCCCN(C)C",
    "edc": "CCN=C=NCCCN(C)C",
    "dmap": "CN(C)c1ccncc1",
    "4-dimethylaminopyridine": "CN(C)c1ccncc1",
    "hobt": "On1nnc2ccccc21",
    "hosu": "O=C1CCC(=O)N1O",
    "nhs": "O=C1CCC(=O)N1O",
    "cdi": "O=C(n1ccnc1)n1ccnc1",
    
    # 催化剂
    "pd/c": "[Pd].C",
    "palladium on carbon": "[Pd].C",
    "raney nickel": "[Ni]",
    "raney ni": "[Ni]",
}


def get_smiles_for_chemical(name: str) -> Optional[str]:
    """根据化学品名称/缩写获取SMILES"""
    normalized = name.lower().strip()
    
    # 直接匹配
    if normalized in CHEMICAL_ABBREVIATIONS:
        return CHEMICAL_ABBREVIATIONS[normalized]
    
    # 尝试去除修饰词
    variations = [
        normalized.replace("anhydrous ", ""),
        normalized.replace("saturated ", ""),
        normalized.replace("aqueous ", ""),
        normalized.replace("conc. ", ""),
        normalized.replace("concentrated ", ""),
        normalized.replace("dilute ", ""),
        normalized.replace("dil. ", ""),
        normalized.replace("dry ", ""),
    ]
    
    for var in variations:
        if var in CHEMICAL_ABBREVIATIONS:
            return CHEMICAL_ABBREVIATIONS[var]
    
    return None


# ================== 动作提取和验证 ==================

def extract_actions_from_action_text(action_text: str) -> List[str]:
    """从ACTION文本中提取所有procedure动作"""
    actions = []
    proc_matches = re.findall(r'<procedure>\s*(\w+)', action_text, re.IGNORECASE)
    actions.extend([m.upper() for m in proc_matches])
    return actions


def check_actions_valid(action_text: str) -> Tuple[bool, List[str]]:
    """
    检查ACTION文本中的动作是否合法
    返回: (是否全部合法, 非法动作列表)
    """
    actions = extract_actions_from_action_text(action_text)
    
    illegal_found = []
    for action in actions:
        if action in ILLEGAL_ACTIONS:
            illegal_found.append(action)
        elif action not in ALLOWED_ACTIONS:
            illegal_found.append(action)
    
    return len(illegal_found) == 0, illegal_found


# ================== 化学品缩写替换 ==================

def replace_chemical_abbreviations_in_text(text: str, molecules_map: Dict[str, str]) -> Tuple[str, List[str]]:
    """
    将文本中的化学品缩写替换为SMILES格式
    优先使用molecules映射表，然后使用预定义的缩写映射
    """
    replaced_chemicals = []
    result = text
    
    # 1. 首先使用molecules映射表替换（优先级最高）
    if molecules_map:
        # 按名称长度降序排序，避免短名称匹配到长名称的一部分
        sorted_items = sorted(molecules_map.items(), key=lambda x: len(x[0]), reverse=True)
        for name, smiles in sorted_items:
            smiles_canon = canonicalize_smiles(smiles)
            # 匹配不在[MOL]标签内的化学品名称
            # 使用负向前瞻和负向后顾确保不替换已格式化的内容
            pattern = rf'(?<![MOL\w])\b{re.escape(name)}\b(?![\w`])'
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, f"[MOL] ```{smiles_canon}``` [/MOL]", result, flags=re.IGNORECASE)
                replaced_chemicals.append(f"{name} -> {smiles_canon}")
    
    # 2. 处理特定动作模式中的化学品缩写
    # 这些模式常见于实验步骤中，如 "WASH with brine", "DRYSOLUTION over sodium sulfate"
    
    # WASH with chemical
    wash_pattern = r'WASH\s+with\s+([^<[\d]+?)(?=<|;|$|\.|\()'
    for match in re.finditer(wash_pattern, result, re.IGNORECASE):
        chemical = match.group(1).strip()
        if chemical and not chemical.startswith('[MOL]'):
            smiles = get_smiles_for_chemical(chemical)
            if smiles:
                smiles_canon = canonicalize_smiles(smiles)
                old_str = match.group(0)
                new_str = f"WASH with [MOL] ```{smiles_canon}``` [/MOL]"
                result = result.replace(old_str, new_str)
                replaced_chemicals.append(f"WASH: {chemical} -> {smiles_canon}")
    
    # DRYSOLUTION over chemical
    dry_pattern = r'DRYSOLUTION\s+over\s+(?:anhydrous\s+)?([^<[\d]+?)(?=<|;|$|\.|\()'
    for match in re.finditer(dry_pattern, result, re.IGNORECASE):
        chemical = match.group(1).strip()
        if chemical and not chemical.startswith('[MOL]'):
            smiles = get_smiles_for_chemical(chemical)
            if smiles:
                smiles_canon = canonicalize_smiles(smiles)
                old_str = match.group(0)
                new_str = f"DRYSOLUTION over [MOL] ```{smiles_canon}``` [/MOL]"
                result = result.replace(old_str, new_str)
                replaced_chemicals.append(f"DRYSOLUTION: {chemical} -> {smiles_canon}")
    
    # TRITURATE with chemical
    trit_pattern = r'TRITURATE\s+with\s+([^<[\d]+?)(?=<|;|$|\.|\()'
    for match in re.finditer(trit_pattern, result, re.IGNORECASE):
        chemical = match.group(1).strip()
        if chemical and not chemical.startswith('[MOL]'):
            smiles = get_smiles_for_chemical(chemical)
            if smiles:
                smiles_canon = canonicalize_smiles(smiles)
                old_str = match.group(0)
                new_str = f"TRITURATE with [MOL] ```{smiles_canon}``` [/MOL]"
                result = result.replace(old_str, new_str)
                replaced_chemicals.append(f"TRITURATE: {chemical} -> {smiles_canon}")
    
    # EXTRACT with chemical
    extract_pattern = r'EXTRACT\s+with\s+([^<[\d]+?)(?=<|;|$|\.|\()'
    for match in re.finditer(extract_pattern, result, re.IGNORECASE):
        chemical = match.group(1).strip()
        if chemical and not chemical.startswith('[MOL]'):
            smiles = get_smiles_for_chemical(chemical)
            if smiles:
                smiles_canon = canonicalize_smiles(smiles)
                old_str = match.group(0)
                new_str = f"EXTRACT with [MOL] ```{smiles_canon}``` [/MOL]"
                result = result.replace(old_str, new_str)
                replaced_chemicals.append(f"EXTRACT: {chemical} -> {smiles_canon}")
    
    # QUENCH with chemical
    quench_pattern = r'QUENCH\s+with\s+([^<[\d]+?)(?=<|;|$|\.|\()'
    for match in re.finditer(quench_pattern, result, re.IGNORECASE):
        chemical = match.group(1).strip()
        if chemical and not chemical.startswith('[MOL]'):
            smiles = get_smiles_for_chemical(chemical)
            if smiles:
                smiles_canon = canonicalize_smiles(smiles)
                old_str = match.group(0)
                new_str = f"QUENCH with [MOL] ```{smiles_canon}``` [/MOL]"
                result = result.replace(old_str, new_str)
                replaced_chemicals.append(f"QUENCH: {chemical} -> {smiles_canon}")
    
    # RECRYSTAL from chemical
    recry_pattern = r'RECRYSTAL\s+(?:from\s+)?([^<[\d]+?)(?=<|;|$|\.|\()'
    for match in re.finditer(recry_pattern, result, re.IGNORECASE):
        chemical = match.group(1).strip()
        if chemical and not chemical.startswith('[MOL]'):
            smiles = get_smiles_for_chemical(chemical)
            if smiles:
                smiles_canon = canonicalize_smiles(smiles)
                old_str = match.group(0)
                new_str = f"RECRYSTAL from [MOL] ```{smiles_canon}``` [/MOL]"
                result = result.replace(old_str, new_str)
                replaced_chemicals.append(f"RECRYSTAL: {chemical} -> {smiles_canon}")
    
    return result, replaced_chemicals


def canonicalize_all_smiles_in_text(text: str) -> str:
    """将文本中所有[MOL]标签内的SMILES标准化"""
    def replace_smiles(match):
        smiles = match.group(1).strip()
        if is_valid_smiles(smiles):
            canon_smiles = canonicalize_smiles(smiles)
            return f"[MOL] ```{canon_smiles}``` [/MOL]"
        return match.group(0)
    
    # 匹配[MOL] ```SMILES``` [/MOL]格式
    pattern = r'\[MOL\]\s*```(.*?)```\s*\[/MOL\]'
    return re.sub(pattern, replace_smiles, text, flags=re.DOTALL)


def canonicalize_molecules_map(molecules_map: Dict[str, str]) -> Dict[str, str]:
    """标准化molecules映射表中的所有SMILES"""
    canonicalized = {}
    for name, smiles in molecules_map.items():
        canonicalized[name] = canonicalize_smiles(smiles)
    return canonicalized


# ================== 主处理流程 ==================

def process_single_item(item: Dict) -> Tuple[Optional[Dict], str]:
    """
    处理单条数据
    返回: (处理后的数据, 状态信息)
    状态: "valid" / "invalid_action" / "error"
    """
    try:
        action_text = item.get("ACTION", "")
        
        # 1. 检查动作是否合法（必须全是24个标准动作，不能包含COOL、HEAT等）
        is_valid, illegal_actions = check_actions_valid(action_text)
        if not is_valid:
            return None, f"invalid_action: {illegal_actions}"
        
        # 2. 获取molecules映射并标准化SMILES
        molecules_map = item.get("molecules", {})
        canonicalized_molecules = canonicalize_molecules_map(molecules_map)
        
        # 3. 替换化学品缩写（使用molecules映射表和预定义缩写）
        new_action, replaced = replace_chemical_abbreviations_in_text(action_text, canonicalized_molecules)
        
        # 4. 标准化所有[MOL]标签内的SMILES
        new_action = canonicalize_all_smiles_in_text(new_action)
        
        # 5. 更新REACTANT、PRODUCT、CATALYST、SOLVENT中的SMILES
        new_item = item.copy()
        new_item["ACTION"] = new_action
        new_item["molecules"] = canonicalized_molecules
        
        # 标准化REACTANT
        if "REACTANT" in new_item:
            new_item["REACTANT"] = [canonicalize_all_smiles_in_text(r) for r in new_item["REACTANT"]]
        
        # 标准化PRODUCT
        if "PRODUCT" in new_item:
            new_item["PRODUCT"] = [canonicalize_all_smiles_in_text(p) for p in new_item["PRODUCT"]]
        
        # 标准化CATALYST
        if "CATALYST" in new_item:
            new_item["CATALYST"] = [canonicalize_all_smiles_in_text(c) for c in new_item["CATALYST"]]
        
        # 标准化SOLVENT
        if "SOLVENT" in new_item:
            new_item["SOLVENT"] = [canonicalize_all_smiles_in_text(s) for s in new_item["SOLVENT"]]
        
        # 记录替换信息用于调试
        new_item["_replaced_chemicals"] = replaced
        
        return new_item, "valid"
        
    except Exception as e:
        return None, f"error: {str(e)}"


def process_high_quality_data(
    input_file: str,
    output_file: str,
    target_count: int = 1000
):
    """
    主函数：处理数据并筛选高质量数据
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 统计信息
    stats = {
        "total_read": 0,
        "valid": 0,
        "invalid_action": 0,
        "error": 0,
        "action_stats": {}
    }
    
    # 存储有效数据
    valid_items = []
    
    print(f"开始处理数据...")
    print(f"目标: 收集 {target_count} 条高质量数据")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print("-" * 60)
    
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 从后往前处理（因为输入文件已经是倒序的）
    for line in tqdm(lines, desc="Processing"):
        if not line.strip():
            continue
        
        stats["total_read"] += 1
        
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        
        # 处理单条数据
        processed, status = process_single_item(item)
        
        if status == "valid":
            valid_items.append(processed)
            stats["valid"] += 1
            
            # 统计动作使用情况
            actions = extract_actions_from_action_text(processed["ACTION"])
            for action in actions:
                stats["action_stats"][action] = stats["action_stats"].get(action, 0) + 1
            
            # 检查是否达到目标
            if len(valid_items) >= target_count:
                break
        elif status.startswith("invalid_action"):
            stats["invalid_action"] += 1
        else:
            stats["error"] += 1
    
    # 写入输出文件
    print("\n" + "=" * 60)
    print(f"处理完成!")
    print(f"总共读取: {stats['total_read']} 条")
    print(f"有效数据: {stats['valid']} 条")
    print(f"非法动作: {stats['invalid_action']} 条")
    print(f"错误数据: {stats['error']} 条")
    print("-" * 60)
    print("动作统计:")
    for action, count in sorted(stats["action_stats"].items(), key=lambda x: -x[1]):
        print(f"  {action}: {count}")
    print("-" * 60)
    
    if len(valid_items) < target_count:
        print(f"警告: 只收集到 {len(valid_items)} 条有效数据，未达到目标 {target_count}")
    
    # 写入文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in valid_items:
            # 移除内部标记
            item_clean = {k: v for k, v in item.items() if not k.startswith('_')}
            f.write(json.dumps(item_clean, ensure_ascii=False) + '\n')
    
    print(f"高质量数据已保存到: {output_path}")
    print(f"总计: {len(valid_items)} 条")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理高质量化学实验数据")
    parser.add_argument("--input", type=str, 
                        default="data/raw/last_1500_temp.jsonl",
                        help="输入JSONL文件路径 (默认: data/raw/last_1500_temp.jsonl)")
    parser.add_argument("--output", type=str, 
                        default="data/raw/high_quality_1k.jsonl",
                        help="输出JSONL文件路径 (默认: data/raw/high_quality_1k.jsonl)")
    parser.add_argument("--target_count", type=int, default=1000,
                        help="目标数据条数 (默认: 1000)")
    
    args = parser.parse_args()
    
    process_high_quality_data(
        input_file=args.input,
        output_file=args.output,
        target_count=args.target_count
    )
