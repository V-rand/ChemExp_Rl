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
    print("警告: RDKit 未安装，SMILES 标准化将受限。")

# ================== 24 个标准动作白名单 ==================
ALLOWED_ACTIONS = {
    'ADD', 'STIR', 'WAIT', 'CONCENTRATE', 'YIELD', 'MAKESOLUTION',
    'FILTER', 'WASH', 'DRYSOLUTION', 'COLLECTLAYER', 'EXTRACT',
    'SETTEMP', 'REFLUX', 'RECRYSTAL', 'PHASESEPA', 'PH', 'PURIFY',
    'QUENCH', 'PARTITION', 'TRITURATE', 'DRYSOLID', 'DEGAS',
    'MICROWAVE', 'SONICATE'
}

# 化学品映射表 (省略部分以保持简洁，建议保留你脚本中完整的 CHEMICAL_ABBREVIATIONS)
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

# ================== 工具函数 ==================

def canonicalize_smiles(smiles: str) -> str:
    if not HAS_RDKIT: return smiles.strip()
    try:
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol else smiles.strip()
    except: return smiles.strip()

def check_actions_valid(action_text: str) -> bool:
    """严格检查：所有动作必须在 24 个标准动作内"""
    actions = re.findall(r'<procedure>\s*(\w+)', action_text, re.IGNORECASE)
    for act in actions:
        if act.upper() not in ALLOWED_ACTIONS:
            return False
    return True

def extract_reactant_quantities(action_text: str) -> Dict[str, str]:
    """从 ACTION 中提取用量信息"""
    quantities_map = {}
    mol_pattern = r'\[MOL\]\s*```(.*?)```\s*\[/MOL\]\s*\(([^)]+)\)'
    matches = re.finditer(mol_pattern, action_text, re.DOTALL)
    for match in matches:
        canon_smiles = canonicalize_smiles(match.group(1).strip())
        quantities_map[canon_smiles] = match.group(2).strip()
    return quantities_map

def extract_smiles_from_tag(text: str) -> Optional[str]:
    m = re.search(r'\[MOL\]\s*```(.*?)```\s*\[/MOL\]', text, re.DOTALL)
    return m.group(1).strip() if m else None

# ================== 文本处理逻辑 ==================

def standardize_item_smiles(item: Dict) -> Dict:
    """对 item 中的所有 SMILES 字段进行标准化"""
    for key in ["REACTANT", "PRODUCT", "CATALYST", "SOLVENT"]:
        if key in item:
            item[key] = [re.sub(r'\[MOL\]\s*```(.*?)```\s*\[/MOL\]', 
                         lambda m: f"[MOL] ```{canonicalize_smiles(m.group(1))}``` [/MOL]", 
                         r) for r in item[key]]
    return item

def build_user_prompt(item: Dict) -> str:
    """构建用于测试的 User Prompt"""
    action_text = item.get("ACTION", "")
    q_map = extract_reactant_quantities(action_text)
    
    parts = ["Please design a chemical experiment procedure based on the following requirements:"]
    
    # 反应物 + 用量
    r_list = []
    for r in item.get("REACTANT", []):
        smiles = extract_smiles_from_tag(r)
        qty = q_map.get(canonicalize_smiles(smiles), "") if smiles else ""
        r_list.append(f"{r} ({qty})" if qty else r)
    parts.append(f"Reactants: {', '.join(r_list)}")
    
    # 产物与产量 (从最后一步 YIELD 提取)
    yield_step = re.findall(r'<procedure>YIELD\s*(.*?)</procedure>', action_text, re.DOTALL)
    if yield_step:
        parts.append(f"Target Product & Expected Yield: {yield_step[-1].strip()}")
    
    if item.get("SOLVENT"): parts.append(f"Solvent: {', '.join(item['SOLVENT'])}")
    if item.get("CATALYST"): parts.append(f"Catalyst: {', '.join(item['CATALYST'])}")
    
    return "\n".join(parts)

# ================== 主处理流程 ==================

def generate_test_data(input_file: str, output_file: str, sys_prompt_path: str):
    # 加载 System Prompt
    try:
        with open(sys_prompt_path, 'r') as f:
            system_prompt = f.read().strip()
    except:
        system_prompt = "You are a professional chemist. Design the procedure for the following reaction."

    valid_count = 0
    total_count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        lines = f_in.readlines()
        for line in tqdm(lines, desc="Processing"):
            if not line.strip(): continue
            item = json.loads(line)
            total_count += 1
            
            # 1. 严格动作过滤
            if not check_actions_valid(item.get("ACTION", "")):
                continue
                
            # 2. 标准化 SMILES 
            item = standardize_item_smiles(item)
            
            # 3. 构造测试数据格式
            test_entry = {
                "index": item.get("index"),
                "prompt": [
                    {"role": "user", "content": build_user_prompt(item)}
                ],
                "gt": item.get("ACTION", "").strip()
            }
            
            f_out.write(json.dumps(test_entry, ensure_ascii=False) + '\n')
            valid_count += 1

    print(f"\n处理完成!")
    print(f"总数据量: {total_count}")
    print(f"通过过滤(24个标准动作): {valid_count}")
    print(f"丢弃数据量: {total_count - valid_count}")
    print(f"输出文件: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/raw/exp_test_data_6k.jsonl")
    parser.add_argument("--output", type=str, default="/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed/chem_test_inference.jsonl")
    parser.add_argument("--sys_prompt", type=str, default="/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/configs/system_prompt_new.txt")
    args = parser.parse_args()
    
    generate_test_data(args.input, args.output, args.sys_prompt)