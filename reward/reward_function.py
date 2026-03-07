"""
reward/reward_function.py - 化学实验步骤奖励函数 (v2)
针对GRPO训练，基于三个维度解耦设计:
- R_format: 格式与守恒惩罚
- R_lcs: 宏观动作序列对齐  
- R_step: 微观步骤得分

Reward = 0.2 * R_format + 0.2 * R_lcs + 0.6 * R_step
"""

import re
import json
import math
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict

# ================== 常量定义 ==================

# 24个允许的动作（严格限定，不允许变体）
ALLOWED_ACTIONS = {
    'ADD', 'STIR', 'WAIT', 'CONCENTRATE', 'YIELD', 'MAKESOLUTION',
    'FILTER', 'WASH', 'DRYSOLUTION', 'COLLECTLAYER', 'EXTRACT',
    'SETTEMP', 'REFLUX', 'RECRYSTAL', 'PHASESEPA', 'PH', 'PURIFY',
    'QUENCH', 'PARTITION', 'TRITURATE', 'DRYSOLID', 'DEGAS',
    'MICROWAVE', 'SONICATE'
}

# 权重配置
WEIGHT_FORMAT = 0.2
WEIGHT_LCS = 0.2
WEIGHT_STEP = 0.6

# 步骤得分: 不加权直接相加，以理论最大值归一化
# S_step = S_mol + S_qty + S_condition (每个1分)
# 理论最大值 = 3 * |LCS| (每个对齐步骤最多3分)
# 归一化 = S_step_total / (3 * len(gt_procedures))

# 用量匹配衰减系数
ALPHA_QTY = 2.0


# ================== 化学同义词映射表 ==================

# canonical_smiles -> [名称列表]
CHEMICAL_SYNONYMS = {
    # 溶剂
    "c1ccoc1": ["thf", "tetrahydrofuran", "四氢呋喃"],
    "cn(c)c=o": ["dmf", "dimethylformamide", "n,n-dimethylformamide"],
    "clcccl": ["dcm", "dichloromethane", "methylene chloride"],
    "co": ["meoh", "methanol", "methyl alcohol"],
    "cco": ["etoh", "ethanol", "ethyl alcohol"],
    "ccocc": ["diethyl ether", "ether", "ethyl ether", "乙醚"],
    "ccoc(c)=o": ["etoac", "ethyl acetate", "ea"],
    "c1ccccc1": ["benzene"],
    "clc1ccccc1": ["chlorobenzene", "phcl"],
    "c1cocco1": ["dioxane", "1,4-dioxane"],
    
    # 无机试剂
    "o=c([o-])[o-].[k+].[k+]": ["k2co3", "potassium carbonate", "碳酸钾"],
    "[na+].[oh-]": ["naoh", "sodium hydroxide"],
    "[k+].[oh-]": ["koh", "potassium hydroxide"],
    "[nh4+].[cl-]": ["ammonium chloride", "nh4cl"],
    "[mg+2].[cl-].[cl-]": ["mgcl2", "magnesium chloride"],
    "[mg+2].[br-].[br-]": ["mgbr2", "magnesium bromide"],
    "[na+].[cl-]": ["nacl", "sodium chloride"],
    "[k+].[cl-]": ["kcl", "potassium chloride"],
    "o=c([o-])[o-].[na+].[na+]": ["na2co3", "sodium carbonate"],
    "[na+].[hco3-]": ["nahco3", "sodium bicarbonate"],
    
    # 碱和催化剂
    "ccn(cc)cc": ["tea", "triethylamine", "et3n", "三乙胺"],
    "ccn(cc)c(c)(c)c": ["dipea", "diisopropylethylamine", "n,n-diisopropylethylamine", "diea"],
    "c1ccccn1": ["pyridine", "py"],
    "c1ccncc1": ["pyridine"],
    
    # 酸
    "cc(=o)o": ["acetic acid", "hoac", "aacoh"],
    "o=c(o)c(f)(f)f": ["tfa", "trifluoroacetic acid"],
    "o=s(=o)(o)o": ["h2so4", "sulfuric acid"],
    "[cl-]": ["hcl", "chloride", "hydrochloric acid"],
    
    # 其他常见试剂
    "o": ["water", "h2o", "水"],
    "n": ["ammonia", "nh3", "氨"],
    "c": ["carbon"],
    "[pd]": ["palladium", "pd"],
    "[pt]": ["platinum", "pt"],
    "[c-]#[n+]": ["cyanide", "cn-"],
    "[n+]#[c-]": ["isocyanide", "nc-"],
    
    # 盐类
    "o=c([o-])o.[na+]": ["sodium bicarbonate", "nahco3"],
    "cc(=o)o[na+]": ["sodium acetate", "naoac"],
    "[na+].[o-]s(=o)(=o)c1ccc(cc1)cc": ["sodium tosylate"],
}

# 反向映射: 名称 -> canonical_smiles
NAME_TO_SMILES = {}
for smiles, names in CHEMICAL_SYNONYMS.items():
    for name in names:
        NAME_TO_SMILES[name.lower()] = smiles


# ================== 温度和时间解析映射 ==================

TEMP_SYNONYMS = {
    "room temperature": (20, 25),
    "rt": (20, 25),
    "r.t.": (20, 25),
    "ambient temperature": (20, 25),
    "ambient": (20, 25),
    "below 0 c": (-50, 0),
    "below 0°c": (-50, 0),
    "below 0c": (-50, 0),
    "0 c": (0, 0),
    "0°c": (0, 0),
    "0c": (0, 0),
    "ice bath": (0, 5),
    "ice-cold": (0, 5),
    "reflux": (50, 150),  # 根据溶剂而定，这里给宽松范围
    "heat": (50, 150),
}

TIME_SYNONYMS = {
    "overnight": (8, 16),
    "o/n": (8, 16),
    "immediately": (0, 0.1),
    "instantly": (0, 0.1),
}


# ================== RDKit支持 ==================

try:
    from rdkit import Chem, rdBase
    rdBase.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


def canonicalize_smiles(smiles: str) -> str:
    """RDKit标准化SMILES"""
    if not smiles:
        return ""
    smiles = smiles.strip()
    if not HAS_RDKIT:
        return smiles.lower()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        pass
    return smiles.lower()


def is_valid_smiles(smiles: str) -> bool:
    """检查是否是有效SMILES"""
    if not smiles or len(smiles) < 2:
        return False
    if not HAS_RDKIT:
        chem_elements = set('cnopsfbclbrimghk')
        return any(c.lower() in chem_elements for c in smiles)
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def smiles_to_canonical(smiles: str) -> Optional[str]:
    """尝试将输入转为canonical SMILES，包括处理常见名称"""
    if not smiles:
        return None
    
    smiles_clean = smiles.strip()
    smiles_lower = smiles_clean.lower()
    
    # 1. 检查是否是已知名称
    if smiles_lower in NAME_TO_SMILES:
        return NAME_TO_SMILES[smiles_lower]
    
    # 2. 检查是否是canonical SMILES
    if is_valid_smiles(smiles_clean):
        return canonicalize_smiles(smiles_clean)
    
    return None


# ================== 解析工具函数 ==================

def extract_smiles_from_text(text: str) -> Set[str]:
    """从文本中提取所有被[MOL]包裹的SMILES"""
    smiles_set = set()
    # 支持 ``` 或 ```xml 或 ``` xml
    pattern = r'\[MOL\]\s*```\s*(?:xml)?\s*(.*?)```\s*\[/MOL\]'
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    for sm in matches:
        sm_clean = sm.strip()
        if sm_clean:
            smiles_set.add(sm_clean.lower())
    return smiles_set


def extract_molecules_from_prompt(prompt: str) -> Dict[str, str]:
    """从prompt中提取所有反应物、产物、催化剂、溶剂"""
    molecules = {}
    
    # 提取各个字段
    for field in ['REACTANT', 'PRODUCT', 'CATALYST', 'SOLVENT']:
        pattern = rf'"{field}":\s*(\[[^\]]*\])'
        match = re.search(pattern, prompt)
        if match:
            try:
                items = json.loads(match.group(1))
                for item in items:
                    smiles = extract_smiles_from_text(item)
                    for sm in smiles:
                        canon = smiles_to_canonical(sm)
                        if canon:
                            molecules[canon] = sm
            except:
                pass
    
    return molecules


def parse_procedure(proc_text: str) -> Dict[str, Any]:
    """
    解析单个procedure
    
    Returns:
        {
            "action": str,
            "raw_text": str,
            "smiles": Set[str],  # canonical smiles
            "quantities": List[Dict],  # [{"value": float, "unit": str, "raw": str}]
            "time": Optional[Dict],  # {"raw": str, "range": (min, max)}
            "temp": Optional[Dict],  # {"raw": str, "range": (min, max)}
        }
    """
    result = {
        "action": "UNKNOWN",
        "raw_text": proc_text,
        "smiles": set(),
        "quantities": [],
        "time": None,
        "temp": None,
    }
    
    # 提取动作
    action_match = re.match(r'\s*(\w+)', proc_text)
    if action_match:
        result["action"] = action_match.group(1).upper()
    
    # 提取SMILES并标准化
    smiles_raw = re.findall(r'\[MOL\]\s*```(?:xml)?(.*?)```\s*\[/MOL\]', proc_text, re.DOTALL)
    for sm in smiles_raw:
        sm_clean = sm.strip()
        if sm_clean:
            canon = smiles_to_canonical(sm_clean)
            if canon:
                result["smiles"].add(canon)
    
    # 提取用量 - 支持多种形式
    # 形式1: (0.60 g, 1.5 mmol)
    # 形式2: (25 mL)
    # 形式3: 0.60 g
    qty_patterns = [
        r'\((\d+\.?\d*)\s*(g|mg|kg|mmol|mol|ml|ul|L|mL|μL)\s*[,;]?\s*(\d+\.?\d*)?\s*(g|mg|kg|mmol|mol|ml|ul|L|mL|μL)?\)',
        r'\((\d+\.?\d*)\s*(g|mg|kg|mmol|mol|ml|ul|L|mL|μL)\)',
        r'(\d+\.?\d*)\s*(g|mg|kg|mmol|mol|ml|ul|L|mL|μL)',
    ]
    
    for pattern in qty_patterns:
        matches = re.findall(pattern, proc_text, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                # 处理多组匹配
                for i in range(0, len(match), 2):
                    if match[i] and match[i+1]:
                        result["quantities"].append({
                            "value": float(match[i]),
                            "unit": match[i+1].lower(),
                            "raw": f"{match[i]} {match[i+1]}"
                        })
            else:
                # 单组匹配
                result["quantities"].append({
                    "value": float(match),
                    "unit": "unknown",
                    "raw": str(match)
                })
    
    # 去重
    seen = set()
    unique_qtys = []
    for q in result["quantities"]:
        key = (q["value"], q["unit"])
        if key not in seen:
            seen.add(key)
            unique_qtys.append(q)
    result["quantities"] = unique_qtys
    
    # 提取时间
    time_match = re.search(r'<time>(.*?)</time>', proc_text, re.IGNORECASE | re.DOTALL)
    if time_match:
        time_raw = time_match.group(1).strip()
        result["time"] = {
            "raw": time_raw,
            "range": parse_time_range(time_raw)
        }
    
    # 提取温度
    temp_match = re.search(r'<temp>(.*?)</temp>', proc_text, re.IGNORECASE | re.DOTALL)
    if temp_match:
        temp_raw = temp_match.group(1).strip()
        result["temp"] = {
            "raw": temp_raw,
            "range": parse_temp_range(temp_raw)
        }
    
    return result


def parse_time_range(time_str: str) -> Optional[Tuple[float, float]]:
    """解析时间字符串为数值范围（小时）
    
    支持格式:
    - 单个值: "15 h", "30 min", "2 day"
    - 范围: "2-3 h", "3-5 hours"
    - below/above: "below 3 min", "above 2 h"
    - 自然语言: "overnight"
    """
    if not time_str:
        return None
    
    time_lower = time_str.lower().strip()
    
    # 检查同义词
    if time_lower in TIME_SYNONYMS:
        return TIME_SYNONYMS[time_lower]
    
    # 解析 "overnight" 等自然语言
    if "overnight" in time_lower:
        return (8, 16)
    
    # 解析 "below X min/h/day" -> (0, X)
    below_match = re.search(r'below\s+(\d+\.?\d*)\s*(min|minutes?|h|hours?|hr|hrs|d|days?)', time_lower)
    if below_match:
        val, unit = float(below_match.group(1)), below_match.group(2)
        if unit.startswith('h'):
            return (0, val)
        elif unit.startswith('d'):
            return (0, val * 24)
        else:
            return (0, val/60)
    
    # 解析 "above X min/h/day" -> (X, inf) 用一个大数表示
    above_match = re.search(r'above\s+(\d+\.?\d*)\s*(min|minutes?|h|hours?|hr|hrs|d|days?)', time_lower)
    if above_match:
        val, unit = float(above_match.group(1)), above_match.group(2)
        if unit.startswith('h'):
            return (val, 9999)
        elif unit.startswith('d'):
            return (val * 24, 9999)
        else:
            return (val/60, 9999)
    
    # 解析范围: "2-3 h", "2-3 hours", "2-3 min", "2-3 day"
    range_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*(min|minutes?|h|hours?|hr|hrs|d|days?)', time_lower)
    if range_match:
        val1, val2, unit = float(range_match.group(1)), float(range_match.group(2)), range_match.group(3)
        if unit.startswith('h'):
            return (val1, val2)
        elif unit.startswith('d'):
            return (val1 * 24, val2 * 24)
        else:
            return (val1/60, val2/60)
    
    # 解析单个值: "15 h", "30 min", "2 day"
    single_match = re.search(r'(\d+\.?\d*)\s*(min|minutes?|h|hours?|hr|hrs|d|days?)', time_lower)
    if single_match:
        val, unit = float(single_match.group(1)), single_match.group(2)
        if unit.startswith('h'):
            return (val, val)
        elif unit.startswith('d'):
            return (val * 24, val * 24)
        else:
            return (val/60, val/60)
    
    return None


def parse_temp_range(temp_str: str) -> Optional[Tuple[float, float]]:
    """解析温度字符串为数值范围（摄氏度）
    
    支持格式:
    - 单个值: "25°C", "25 C", "100C"
    - 范围: "20-25°C", "3-5 C"
    - below/above: "below 40°C", "above 0 C"
    - 自然语言: "room temperature", "rt", "reflux"
    """
    if not temp_str:
        return None
    
    temp_lower = temp_str.lower().strip()
    
    # 检查同义词
    if temp_lower in TEMP_SYNONYMS:
        return TEMP_SYNONYMS[temp_lower]
    
    # 解析 "below X°C" / "below X C"
    below_match = re.search(r'below\s+(\d+\.?\d*)\s*°?c', temp_lower)
    if below_match:
        val = float(below_match.group(1))
        return (-100, val)  # 假设最低-100°C
    
    # 解析 "above X°C" / "above X C"
    above_match = re.search(r'above\s+(\d+\.?\d*)\s*°?c', temp_lower)
    if above_match:
        val = float(above_match.group(1))
        return (val, 999)  # 假设最高999°C
    
    # 解析范围: "20-25°C" 或 "20 - 25 C" 或 "3-5°C"
    range_match = re.search(r'(-?\d+\.?\d*)\s*-\s*(-?\d+\.?\d*)\s*°?c', temp_lower)
    if range_match:
        return (float(range_match.group(1)), float(range_match.group(2)))
    
    # 解析单个值: "25°C" 或 "25 C" 或 "25c"
    single_match = re.search(r'(-?\d+\.?\d*)\s*°?c', temp_lower)
    if single_match:
        val = float(single_match.group(1))
        return (val, val)
    
    return None


def check_range_overlap(range1: Tuple[float, float], range2: Tuple[float, float]) -> bool:
    """检查两个区间是否有重叠"""
    if not range1 or not range2:
        return False
    return not (range1[1] < range2[0] or range2[1] < range1[0])


def normalize_quantity(qty: Dict[str, Any]) -> Optional[float]:
    """将用量转换为标准单位（克或毫升）"""
    if not qty or "value" not in qty or "unit" not in qty:
        return None
    
    val = qty["value"]
    unit = qty["unit"].lower()
    
    # 质量单位 -> 克
    if unit == "g":
        return val
    elif unit == "mg":
        return val / 1000
    elif unit == "kg":
        return val * 1000
    
    # 物质的量 -> 假定为1mmol = 100mg (平均分子量100)
    elif unit == "mmol":
        return val / 1000 * 100  # 100mg/mmol
    elif unit == "mol":
        return val * 100
    
    # 体积单位 -> 毫升
    elif unit in ["ml", "ul", "μl", "l"]:
        if unit == "ml":
            return val
        elif unit in ["ul", "μl"]:
            return val / 1000
        elif unit == "l":
            return val * 1000
    
    return None


def compute_quantity_similarity(qty_list1: List[Dict], qty_list2: List[Dict]) -> float:
    """计算两组用量的相似度，基于相对误差"""
    if not qty_list1 or not qty_list2:
        return 0.0
    
    # 标准化用量
    norm1 = [normalize_quantity(q) for q in qty_list1 if normalize_quantity(q) is not None]
    norm2 = [normalize_quantity(q) for q in qty_list2 if normalize_quantity(q) is not None]
    
    if not norm1 or not norm2:
        return 0.0
    
    # 找到最佳匹配的相对误差
    min_error = float('inf')
    for q1 in norm1:
        for q2 in norm2:
            if q2 > 0:
                error = abs(q1 - q2) / q2
                min_error = min(min_error, error)
    
    if min_error == float('inf'):
        return 0.0
    
    # 指数衰减得分
    return math.exp(-ALPHA_QTY * min_error)


# ================== R_format: 格式与守恒惩罚 ==================

def compute_format_reward(solution_str: str, prompt: str = "") -> float:
    """
    格式得分计算
    
    初始分1.0，根据违规情况扣分
    """
    score = 1.0
    
    # 1. 检查XML标签平衡
    tags_to_check = ['think', 'answer', 'procedure', 'time', 'temp']
    for tag in tags_to_check:
        open_count = len(re.findall(rf'<{tag}[^>]*>', solution_str, re.IGNORECASE))
        close_count = len(re.findall(rf'</{tag}>', solution_str, re.IGNORECASE))
        if open_count != close_count:
            score -= 0.1
            if score < 0:
                return 0.0
    
    # 2. 检查必须有think和answer标签
    if not re.search(r'<think>.*?</think>', solution_str, re.DOTALL | re.IGNORECASE):
        score -= 0.3
    
    if not re.search(r'<answer>.*?</answer>', solution_str, re.DOTALL | re.IGNORECASE):
        score -= 0.3
    
    if score <= 0:
        return max(score, 0.0)
    
    # 提取answer部分
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL | re.IGNORECASE)
    if not answer_match:
        return max(score, 0.0)
    
    answer_part = answer_match.group(1)
    
    # 3. 检查后向断言：动作必须在<procedure>内
    # 找到所有裸露的动作词（不在<procedure>内的）
    for action in ALLOWED_ACTIONS:
        # 使用负向回顾后发查找裸露的动作
        # 简化检查：移除所有<procedure>...</procedure>后的文本中不应有动作词
        pass
    
    # 更简单的检查：提取所有procedure，检查内部动作
    procedures = re.findall(r'<procedure>(.*?)</procedure>', answer_part, re.DOTALL | re.IGNORECASE)
    
    # 检查非原子性：一个procedure内有多个动作
    for proc in procedures:
        proc_clean = re.sub(r'<[^>]+>', ' ', proc)
        action_count = 0
        for action in ALLOWED_ACTIONS:
            if re.search(rf'\b{action}\b', proc_clean, re.IGNORECASE):
                action_count += 1
        if action_count > 1:
            score -= 0.05
    
    # 4. 检查SMILES格式
    # 4.1 检查是否有裸露的SMILES（疑似）
    # 移除已正确包裹的部分
    stripped = re.sub(r'\[MOL\].*?\[/MOL\]', ' ', answer_part, flags=re.DOTALL)
    stripped = re.sub(r'<[^>]+>', ' ', stripped)
    
    # 检查疑似SMILES的裸露文本
    words = stripped.split()
    naked_smiles_count = 0
    for word in words:
        # 疑似SMILES的特征
        if len(word) >= 5 and any(c in word for c in '()[]=#@+-'):
            has_chem = any(c.lower() in 'cnopsfbclbrimghk' for c in word)
            if has_chem:
                naked_smiles_count += 1
    
    if naked_smiles_count > 0:
        score -= min(naked_smiles_count * 0.03, 0.15)
    
    # 4.2 检查prompt中的分子是否都被使用
    if prompt:
        prompt_mols = extract_molecules_from_prompt(prompt)
        answer_mols = extract_smiles_from_text(answer_part)
        
        # 检查覆盖度
        if prompt_mols:
            covered = sum(1 for m in prompt_mols if m.lower() in answer_mols or 
                         any(m.lower() == smiles_to_canonical(a) for a in answer_mols))
            coverage = covered / len(prompt_mols)
            if coverage < 0.5:  # 如果覆盖度低于50%
                score -= 0.1
    
    # 5. 检查时间和温度格式
    # 5.1 检查裸露的时间
    temp_stripped = re.sub(r'<temp>.*?</temp>', '', answer_part, flags=re.DOTALL | re.IGNORECASE)
    time_stripped = re.sub(r'<time>.*?</time>', '', answer_part, flags=re.DOTALL | re.IGNORECASE)
    
    # 检查是否有时间关键词但没有<time>标签
    naked_time_patterns = [
        r'\b\d+\s*(min|minutes?|h|hours?|hr|hrs|overnight)\b',
        r'\b(overnight|immediately)\b',
    ]
    for pattern in naked_time_patterns:
        if re.search(pattern, time_stripped, re.IGNORECASE):
            score -= 0.05
            break
    
    # 检查是否有温度关键词但没有<temp>标签
    naked_temp_patterns = [
        r'\b\d+\s*°?c\b',
        r'\b(room temperature|rt|r\.t\.|ambient|reflux|ice bath)\b',
    ]
    for pattern in naked_temp_patterns:
        if re.search(pattern, temp_stripped, re.IGNORECASE):
            score -= 0.05
            break
    
    # 6. 检查最后一步必须是YIELD
    if procedures:
        last_proc = procedures[-1]
        if not re.search(r'^\s*YIELD\b', last_proc, re.IGNORECASE):
            score -= 0.1
    
    # 7. 检查非法动作
    for proc in procedures:
        action_match = re.match(r'\s*(\w+)', proc)
        if action_match:
            action = action_match.group(1).upper()
            if action not in ALLOWED_ACTIONS:
                score -= 0.1
    
    return max(score, 0.0)


# ================== R_lcs: 宏观动作序列对齐 ==================

def compute_lcs(seq1: List[str], seq2: List[str]) -> List[Tuple[int, int]]:
    """
    计算最长公共子序列(LCS)
    
    Returns:
        对齐的索引对列表 [(i1, j1), (i2, j2), ...]
    """
    m, n = len(seq1), len(seq2)
    
    # 动态规划表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # 回溯获取对齐结果
    alignments = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i-1] == seq2[j-1]:
            alignments.append((i-1, j-1))
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1
    
    return alignments[::-1]


def compute_lcs_reward(pred_procedures: List[str], gt_procedures: List[str]) -> Tuple[float, List[Tuple[int, int]]]:
    """
    基于LCS的动作序列对齐得分
    
    R_lcs = |LCS| / max(N_pred, N_gt)
    """
    # 提取动作序列
    pred_actions = []
    for proc in pred_procedures:
        action_match = re.match(r'\s*(\w+)', proc)
        if action_match:
            pred_actions.append(action_match.group(1).upper())
        else:
            pred_actions.append("UNKNOWN")
    
    gt_actions = []
    for proc in gt_procedures:
        action_match = re.match(r'\s*(\w+)', proc)
        if action_match:
            gt_actions.append(action_match.group(1).upper())
        else:
            gt_actions.append("UNKNOWN")
    
    if not gt_actions:
        return 0.0, []
    
    # 计算LCS
    lcs_alignments = compute_lcs(pred_actions, gt_actions)
    
    # 计算得分
    lcs_len = len(lcs_alignments)
    max_len = max(len(pred_actions), len(gt_actions))
    
    if max_len == 0:
        return 0.0, lcs_alignments
    
    score = lcs_len / max_len
    
    return score, lcs_alignments


# ================== R_step: 步骤得分 ==================

def count_gt_signals(gt_step: Dict) -> Tuple[int, int, int, int]:
    """
    统计GT步骤拥有的信号数量
    
    Returns:
        (has_mol, has_qty, has_condition, total_signals)
        每个为0或1
    """
    # 是否有分子
    has_mol = 1 if gt_step.get("smiles") else 0
    
    # 是否有用量
    has_qty = 1 if gt_step.get("quantities") else 0
    
    # 是否有条件（时间或温度）
    has_condition = 1 if (gt_step.get("time") or gt_step.get("temp")) else 0
    
    # 总信号数
    total = has_mol + has_qty + has_condition
    
    return has_mol, has_qty, has_condition, total


def compute_step_reward(pred_step: Dict, gt_step: Dict) -> Dict[str, Any]:
    """
    计算单个对齐步骤的得分
    
    对于LCS对齐的步骤，动作必然匹配。
    如果GT步骤没有任何信号（分子、用量、条件），则动作对了就得满分（1分）。
    如果GT步骤有信号，则计算各项信号的匹配情况（每项0或1分），总分 = 各项之和 / 信号数。
    
    Returns:
        {
            "mol": float,  # 0或1
            "quantity": float,  # 0或1
            "condition": float,  # 0或1
            "gt_signals": (int, int, int, int),  # GT拥有的信号
            "step_max_score": int,  # 该步骤的理论最大分值
            "total": float  # 该步骤实际得分
        }
    """
    result = {
        "mol": 0.0,
        "quantity": 0.0,
        "condition": 0.0,
        "gt_signals": (0, 0, 0, 0),
        "step_max_score": 1,  # 默认至少1分（动作分）
        "total": 0.0
    }
    
    # 统计GT拥有的信号
    has_mol, has_qty, has_cond, total_signals = count_gt_signals(gt_step)
    result["gt_signals"] = (has_mol, has_qty, has_cond, total_signals)
    
    # 如果没有信号，动作对了就得满分
    if total_signals == 0:
        result["total"] = 1.0
        return result
    
    # 有信号的情况，理论最大分值 = 信号数
    result["step_max_score"] = total_signals
    
    # 1. 分子匹配得分
    if has_mol:
        pred_mols = pred_step.get("smiles", set())
        gt_mols = gt_step.get("smiles", set())
        
        if not pred_mols:
            result["mol"] = 0.0
        else:
            intersection = len(pred_mols & gt_mols)
            union = len(pred_mols | gt_mols)
            jaccard = intersection / union if union > 0 else 0.0
            # Jaccard >= 0.6 算匹配
            result["mol"] = 1.0 if jaccard >= 0.6 else 0.0
    
    # 2. 用量匹配得分
    if has_qty:
        pred_qtys = pred_step.get("quantities", [])
        gt_qtys = gt_step.get("quantities", [])
        
        if not pred_qtys:
            result["quantity"] = 0.0
        else:
            qty_sim = compute_quantity_similarity(pred_qtys, gt_qtys)
            # 相似度 >= 0.75 算匹配
            result["quantity"] = 1.0 if qty_sim >= 0.75 else 0.0
    
    # 3. 条件匹配得分（时间和温度）
    if has_cond:
        pred_time = pred_step.get("time")
        gt_time = gt_step.get("time")
        pred_temp = pred_step.get("temp")
        gt_temp = gt_step.get("temp")
        
        time_ok = True
        temp_ok = True
        
        # 检查时间
        if gt_time:
            if not pred_time:
                time_ok = False
            else:
                time_ok = check_range_overlap(pred_time.get("range"), gt_time.get("range"))
        
        # 检查温度
        if gt_temp:
            if not pred_temp:
                temp_ok = False
            else:
                temp_ok = check_range_overlap(pred_temp.get("range"), gt_temp.get("range"))
        
        # 时间和温度都满足才算1分
        result["condition"] = 1.0 if (time_ok and temp_ok) else 0.0
    
    # 计算总分 = 各项之和
    result["total"] = result["mol"] + result["quantity"] + result["condition"]
    
    return result


def compute_all_steps_reward(
    pred_procedures: List[str], 
    gt_procedures: List[str],
    lcs_alignments: List[Tuple[int, int]]
) -> Tuple[float, List[Dict], int]:
    """
    计算所有对齐步骤的奖励
    
    理论最大值 = 所有GT步骤的信号数总和（每个无信号的步骤计1分）
    实际得分 = 所有对齐步骤的得分总和
    R_step = 实际得分 / 理论最大值
    
    Returns:
        (R_step得分, 详细得分列表, 理论最大总分)
    """
    if not gt_procedures:
        return 0.0, [], 0
    
    # 解析所有procedure
    pred_parsed = [parse_procedure(p) for p in pred_procedures]
    gt_parsed = [parse_procedure(g) for g in gt_procedures]
    
    # 计算理论最大值：所有GT步骤的分值总和
    # 无信号的步骤计1分，有信号的步骤计信号数分
    total_max_score = 0
    for gt_step in gt_parsed:
        _, _, _, total_signals = count_gt_signals(gt_step)
        if total_signals == 0:
            total_max_score += 1  # 只有动作的步骤，满分1分
        else:
            total_max_score += total_signals
    
    step_details = []
    actual_score = 0.0
    
    for pred_idx, gt_idx in lcs_alignments:
        pred_step = pred_parsed[pred_idx]
        gt_step = gt_parsed[gt_idx]
        
        step_scores = compute_step_reward(pred_step, gt_step)
        step_details.append({
            "pred_idx": pred_idx,
            "gt_idx": gt_idx,
            "pred_action": pred_step["action"],
            "gt_action": gt_step["action"],
            "scores": step_scores
        })
        actual_score += step_scores["total"]
    
    # 归一化：实际得分 / 理论最大分值
    if total_max_score == 0:
        r_step = 1.0 if actual_score == 0 else 0.0
    else:
        r_step = actual_score / total_max_score
    
    return r_step, step_details, total_max_score


# ================== 主函数 ==================

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict] = None
) -> float:
    """
    完整的奖励计算函数
    
    Reward = 0.2 * R_format + 0.2 * R_lcs + 0.6 * R_step
    
    Args:
        data_source: 数据集名称
        solution_str: 模型生成的回答
        ground_truth: 标准答案，包含:
            - 'actions': List[str] - GT procedure列表
            - 'prompt': str (optional) - 原始prompt，用于格式检查
    
    Returns:
        float: 最终奖励分数 [0, 1]
    """
    # 提取prompt（如果可用）- 确保是字符串
    prompt = ground_truth.get('prompt', '')
    if hasattr(prompt, '__len__') and not isinstance(prompt, str):
        prompt = ''
    if not prompt and extra_info:
        prompt = extra_info.get('prompt', '')
        if hasattr(prompt, '__len__') and not isinstance(prompt, str):
            prompt = ''
    
    # 1. 计算格式得分 R_format
    r_format = compute_format_reward(solution_str, prompt)
    
    # 2. 提取procedure列表
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL | re.IGNORECASE)
    if not answer_match:
        # 没有answer标签，只有格式得分
        return r_format * WEIGHT_FORMAT
    
    pred_procs = re.findall(r'<procedure>(.*?)</procedure>', answer_match.group(1), re.DOTALL | re.IGNORECASE)
    
    # 获取GT procedures
    gt_procs = ground_truth.get('actions', [])
    if isinstance(gt_procs, str):
        # 如果是字符串，尝试解析
        gt_answer_match = re.search(r'<answer>(.*?)</answer>', gt_procs, re.DOTALL | re.IGNORECASE)
        if gt_answer_match:
            gt_procs = re.findall(r'<procedure>(.*?)</procedure>', gt_answer_match.group(1), re.DOTALL | re.IGNORECASE)
    
    if not gt_procs:
        # 没有GT数据，只返回格式得分
        return r_format * WEIGHT_FORMAT
    
    # 3. 计算LCS得分 R_lcs
    r_lcs, alignments = compute_lcs_reward(pred_procs, gt_procs)
    
    # 4. 计算步骤得分 R_step
    r_step, step_details, total_max_score = compute_all_steps_reward(pred_procs, gt_procs, alignments)
    
    # 5. 计算最终奖励
    reward = WEIGHT_FORMAT * r_format + WEIGHT_LCS * r_lcs + WEIGHT_STEP * r_step
    
    return round(reward, 4)


def compute_score_with_details(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    带详细信息的奖励计算函数（用于调试）
    """
    prompt = ground_truth.get('prompt', '')
    if hasattr(prompt, '__len__') and not isinstance(prompt, str):
        prompt = ''
    if not prompt and extra_info:
        prompt = extra_info.get('prompt', '')
        if hasattr(prompt, '__len__') and not isinstance(prompt, str):
            prompt = ''
    
    # 1. 格式得分
    r_format = compute_format_reward(solution_str, prompt)
    
    # 2. 提取procedures
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL | re.IGNORECASE)
    if not answer_match:
        return {
            "score": r_format * WEIGHT_FORMAT,
            "r_format": r_format,
            "r_lcs": 0.0,
            "r_step": 0.0,
            "total_max_score": 0,
            "error": "No answer tag found"
        }
    
    pred_procs = re.findall(r'<procedure>(.*?)</procedure>', answer_match.group(1), re.DOTALL | re.IGNORECASE)
    
    gt_procs = ground_truth.get('actions', [])
    if isinstance(gt_procs, str):
        gt_answer_match = re.search(r'<answer>(.*?)</answer>', gt_procs, re.DOTALL | re.IGNORECASE)
        if gt_answer_match:
            gt_procs = re.findall(r'<procedure>(.*?)</procedure>', gt_answer_match.group(1), re.DOTALL | re.IGNORECASE)
    
    if not gt_procs:
        return {
            "score": r_format * WEIGHT_FORMAT,
            "r_format": r_format,
            "r_lcs": 0.0,
            "r_step": 0.0,
            "total_max_score": 0,
            "error": "No ground truth actions"
        }
    
    # 3. LCS得分
    r_lcs, alignments = compute_lcs_reward(pred_procs, gt_procs)
    
    # 4. 步骤得分
    r_step, step_details, total_max_score = compute_all_steps_reward(pred_procs, gt_procs, alignments)
    
    # 5. 最终奖励
    reward = WEIGHT_FORMAT * r_format + WEIGHT_LCS * r_lcs + WEIGHT_STEP * r_step
    
    return {
        "score": round(reward, 4),
        "r_format": round(r_format, 4),
        "r_lcs": round(r_lcs, 4),
        "r_step": round(r_step, 4),
        "format_contrib": round(WEIGHT_FORMAT * r_format, 4),
        "lcs_contrib": round(WEIGHT_LCS * r_lcs, 4),
        "step_contrib": round(WEIGHT_STEP * r_step, 4),
        "pred_procedure_count": len(pred_procs),
        "gt_procedure_count": len(gt_procs),
        "lcs_length": len(alignments),
        "total_max_score": total_max_score,
        "step_details": step_details
    }
