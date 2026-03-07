"""
reward/reward_function.py - 化学实验步骤奖励函数
针对GRPO训练，简洁优雅的设计
"""
import re
import json
import re
import json
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from dataclasses import dataclass


@dataclass
class StepReward:
    """
    Step-level reward for a single procedure.

    用于追踪每个步骤的奖励组成部分：
    - action_score: 动作类型是否正确 (0 或 1)
    - mol_score: 分子匹配度 (0-1 Jaccard相似度)
    - qty_score: 用量匹配度 (0 或 1)
    - step_total: 该步骤的加权总分
    - gt_action: 对应的GT动作（用于调试）
    - pred_action: 预测的动作（用于调试）
    """
    action_score: float = 0.0
    mol_score: float = 0.0
    qty_score: float = 0.0
    step_total: float = 0.0
    gt_action: Optional[str] = None
    pred_action: Optional[str] = None

# RDKit支持
try:
    from rdkit import Chem, DataStructs, rdBase
    rdBase.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

# 允许的动作
ALLOWED_ACTIONS = {
    'ADD', 'STIR', 'WAIT', 'CONCENTRATE', 'YIELD', 'MAKESOLUTION',
    'FILTER', 'WASH', 'DRYSOLUTION', 'COLLECTLAYER', 'EXTRACT',
    'SETTEMP', 'REFLUX', 'RECRYSTAL', 'PHASESEPA', 'PH', 'PURIFY',
    'QUENCH', 'PARTITION', 'TRITURATE', 'DRYSOLID', 'DEGAS',
    'MICROWAVE', 'SONICATE'
}

# 常用化学品映射：canonical_smiles -> [名称列表]
COMMON_CHEMICALS = {
    "c1ccoc1": ["thf", "tetrahydrofuran"],
    "cn(c)c=o": ["dmf", "dimethylformamide"],
    "clcccl": ["dcm", "dichloromethane"],
    "co": ["meoh", "methanol"],
    "cco": ["etoh", "ethanol"],
    "ccocc": ["diethyl ether", "ether", "ethyl ether"],
    "ccoc(c)=o": ["etoac", "ethyl acetate"],
    "o=c([o-])[o-].[k+].[k+]": ["k2co3", "potassium carbonate"],
    "[na+].[oh-]": ["naoh", "sodium hydroxide"],
    "ccn(cc)cc": ["tea", "triethylamine"],
    # DIPEA (N,N-diisopropylethylamine) - CORRECT SMILES
    "ccn(cc)c(c)(c)c": ["dipea", "diisopropylethylamine", "n,n-diisopropylethylamine"],
    "c1ccccn1": ["pyridine"],
    "cc(=o)o[na+]": ["sodium acetate", "naco3"],
    "[nh4+][cl-]": ["ammonium chloride", "nh4cl"],
    "[mg+2].2[cl-]": ["magnesium chloride", "mgcl2"],
    "o=c(o)oc": ["acetic acid", "hoac"],
    "n": ["ammonia", "nh3"],
    "o": ["water", "h2o"],
    # ...可根据需要继续添加
}


# ================== SMILES 工具函数 ==================

def canonicalize_smiles(smiles: str) -> str:
    """
    RDKit标准化SMILES

    注意：RDKit输出的canonical SMILES已经正确使用小写表示芳香原子，
    不要再强制lower()，否则会将脂肪族原子也变成小写导致无效SMILES。
    """
    if not HAS_RDKIT:
        return smiles.strip().lower()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    except:
        pass
    return smiles.strip().lower()


def is_valid_smiles(smiles: str) -> bool:
    """检查是否是有效SMILES"""
    if not HAS_RDKIT:
        # 简单启发式检查
        if len(smiles) < 3 or len(smiles) > 500:
            return False
        # 必须包含至少一个化学元素符号
        chem_elements = set('cnopsfbclbrimghk')
        return any(c.lower() in chem_elements for c in smiles)
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def is_likely_naked_smiles(text: str) -> bool:
    """
    判断文本是否可能是裸露的SMILES

    简化策略：不需要和molecules匹配，只要疑似就判断
    """
    text = text.strip()

    # 长度检查
    if len(text) < 5 or len(text) > 200:
        return False

    # 排除明显的非SMILES
    if re.match(r'^[a-zA-Z]+$', text):  # 纯单词
        return False
    if re.match(r'^\d+$', text):  # 纯数字
        return False
    if text.lower() in ['procedure', 'time', 'temp', 'answer', 'think', 'mol']:
        return False

    # SMILES特征：包含化学元素和括号
    chem_elements = set('cnopsfbclbrimghk[]()=#@+-')
    has_chem_char = any(c.lower() in chem_elements for c in text)
    has_structure = any(c in '()[]=#@+-' for c in text)

    return has_chem_char and (has_structure or len(re.findall(r'[0-9]', text)) > 0)


# ================== 格式检查 ==================

def check_format(output: str, molecules_json: str = "{}") -> float:
    """
    格式检查，从0.5开始扣分

    Args:
        output: 模型输出
        molecules_json: ground_truth中的molecules映射表（用于验证）
    """
    score = 0.5

    # 解析molecules映射表
    try:
        molecules_map = json.loads(molecules_json)
        gt_smiles_set = set(molecules_map.values())
    except:
        gt_smiles_set = set()

    # 1. 检查标签完整性
    if not re.search(r'<think>.*?</think>', output, re.DOTALL):
        score -= 0.5
        return max(score, 0.0)

    if not re.search(r'<answer>.*?</answer>', output, re.DOTALL):
        score -= 0.5
        return max(score, 0.0)

    # 2. 提取answer部分
    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    if not answer_match:
        return 0.0
    answer_part = answer_match.group(1)

    # 3. 检查procedure数量（至少2个，防止钻空子）
    proc_count = len(re.findall(r'<procedure>', answer_part))
    if proc_count < 2:
        score -= 0.2

    # 4. 检查最后一步是YIELD
    # 匹配完整的procedure标签获取最后一个动作
    all_procedures = re.findall(r'<procedure>\s*(\w+)', answer_part, re.IGNORECASE)
    if all_procedures:
        last_action = all_procedures[-1].upper()
        if last_action != 'YIELD':
            score -= 0.15

    # 5. 检查SMILES格式：必须是 [MOL] ```SMILES``` [/MOL]
    # 5.1 先找出被正确包裹的SMILES数量（支持```或```xml）
    wrapped_smiles = set()
    wrapped_patterns = re.findall(r'\[MOL\]\s*```(?:xml)?(.*?)```\s*\[/MOL\]', answer_part, re.DOTALL)
    for sm in wrapped_patterns:
        clean_sm = sm.strip()
        if is_valid_smiles(clean_sm):
            wrapped_smiles.add(canonicalize_smiles(clean_sm))

    # 5.2 检查裸露SMILES（疑似就扣分，不需要和molecules匹配）
    # 只检查answer部分的裸露SMILES，think部分放宽检查
    naked_smiles_count = 0
    # 移除已正确包裹的部分和XML标签
    answer_stripped = re.sub(r'\[MOL\].*?\[/MOL\]', ' ', answer_part, re.DOTALL)
    answer_stripped = re.sub(r'<[^>]+>', ' ', answer_stripped)
    # 提取所有疑似裸露SMILES
    words = answer_stripped.split()
    for word in words:
        # 更严格的判断：必须包含括号或数字，长度>=6
        if is_likely_naked_smiles(word) and len(word) >= 6:
            naked_smiles_count += 1

    # 只在answer部分检查裸露SMILES，且最多扣0.15分
    if naked_smiles_count > 0:
        score -= min(naked_smiles_count * 0.05, 0.15)

    # 5.3 鼓励使用SMILES：如果完全没有SMILES，额外扣分
    if len(wrapped_smiles) == 0:
        score -= 0.1

    # 6. 检查time格式：必须是 <time>...</time>
    naked_times = re.findall(
        r'\b(\d+\s*(?:min|h|hour|sec|overnight))\b',
        re.sub(r'<time>.*?</time>', '', answer_part, re.IGNORECASE)
    )
    if naked_times:
        score -= 0.1

    # 7. 检查temp格式：必须是 <temp>...</temp>
    naked_temps = re.findall(
        r'\b(\d+\s*°?C|room\s*temperature|rt|r\.t\.|ambient)\b',
        re.sub(r'<temp>.*?</temp>', '', answer_part, re.IGNORECASE)
    )
    if naked_temps:
        score -= 0.1

    # 8. 检查标签闭合性
    for tag in ['procedure', 'time', 'temp']:
        open_count = len(re.findall(f'<{tag}>', answer_part))
        close_count = len(re.findall(f'</{tag}>', answer_part))
        if open_count != close_count:
            score -= 0.1
            break

    return max(score, 0.0)


# ================== Procedure 解析 ==================

def parse_procedure(proc_text: str) -> Dict:
    """
    解析单个procedure，MAKESOLUTION作为整体

    返回: {
        "action": "ADD" | "STIR" | "MAKESOLUTION" | ...
        "smiles": set,        # 物质的SMILES集合（标准化后）
        "quantities": list,    # 用量列表
        "time": str|None,
        "temp": str|None
    }
    """
    # 提取动作 - 支持两种格式：
    # 1. <procedure>ACTION ...</procedure>
    # 2. ACTION ... （已经提取的内容）
    action_match = re.match(r'<procedure>\s*(\w+)', proc_text, re.IGNORECASE)
    if action_match:
        action = action_match.group(1).upper()
    else:
        # 直接匹配开头的动作名称
        direct_match = re.match(r'^(\w+)', proc_text.strip())
        action = direct_match.group(1).upper() if direct_match else "UNKNOWN"

    # 提取SMILES并标准化（支持```或```xml）
    smiles_raw = re.findall(r'\[MOL\]\s*```(?:xml)?(.*?)```\s*\[/MOL\]', proc_text, re.DOTALL)
    smiles_canon = {
        canonicalize_smiles(s.strip())
        for s in smiles_raw
        if s.strip() and is_valid_smiles(s.strip())
    }

    # 提取用量
    quantities = re.findall(r'\(([\d.]+\s*(?:g|mmol|ml|L|M|uL))\)', proc_text, re.IGNORECASE)

    # 提取time
    time_match = re.search(r'<time>(.*?)</time>', proc_text, re.IGNORECASE)
    time_val = time_match.group(1).strip() if time_match else None

    # 提取temp
    temp_match = re.search(r'<temp>(.*?)</temp>', proc_text, re.IGNORECASE)
    temp_val = temp_match.group(1).strip() if temp_match else None

    return {
        "action": action,
        "smiles": smiles_canon,
        "quantities": quantities,
        "time": time_val,
        "temp": temp_val
    }


# ================== Step-Level Score ==================

def compute_step_level_reward(
    pred_procs: List[str],
    gt_procs: List[str],
    molecules_json: str = "{}"
) -> tuple[List[StepReward], float]:
    """
    Step-level reward computation（逐步累加，不平均）

    Returns:
        List[StepReward]: 每个步骤的详细奖励
        float: 总化学奖励（0-1尺度）
    """
    # 解析ground truth molecules
    try:
        molecules_map = json.loads(molecules_json)
    except:
        molecules_map = {}

    # 解析所有procedure
    pred_steps = [parse_procedure(p) for p in pred_procs]
    gt_steps = [parse_procedure(g) for g in gt_procs]

    if not gt_steps:
        return [], 0.0

    # 使用动作类型对齐（比LCS更稳定）
    step_rewards = []

    # 简单的贪婪匹配：将pred步骤与GT步骤对齐
    for i, pred_step in enumerate(pred_steps):
        if i < len(gt_steps):
            gt_step = gt_steps[i]

            # 动作必须匹配
            if pred_step["action"] == gt_step["action"]:
                action_score = 1.0
                mol_score = match_smiles_set(pred_step["smiles"], gt_step["smiles"])

                # YIELD：只检查分子，不检查用量
                if gt_step["action"] == "YIELD":
                    qty_score = 0.3  # YIELD步骤不扣用量分
                else:
                    qty_score = match_quantity(pred_step["quantities"], gt_step["quantities"])

                # 步骤总分 = 0.4 * action + 0.5 * mol + 0.1 * qty
                step_total = 0.4 * action_score + 0.5 * mol_score + 0.1 * qty_score
            else:
                # 动作错误
                action_score = 0.0
                mol_score = 0.0
                qty_score = 0.0
                step_total = 0.0

            step_rewards.append(StepReward(
                action_score=action_score,
                mol_score=mol_score,
                qty_score=qty_score,
                step_total=step_total,
                gt_action=gt_step["action"] if i < len(gt_steps) else None,
                pred_action=pred_step["action"]
            ))

    # 总化学奖励 = 步骤总分平均（0-1尺度）
    total_chemistry = sum(sr.step_total for sr in step_rewards) / len(gt_steps)

    return step_rewards, total_chemistry


# ================== SMILES 集合匹配 ==================

def match_smiles_set(pred_smiles: Set[str], gt_smiles: Set[str]) -> float:
    """
    SMILES集合匹配（用于MAKESOLUTION和其他ADD动作）

    完全匹配 → 1.0
    部分匹配（Jaccard相似度）→ 相似度
    名称/缩写匹配 → 0.8 * 相似度
    """
    if not gt_smiles:
        return 1.0 if not pred_smiles else 0.0

    if not pred_smiles:
        return 0.0

    # 1. 标准化pred_smiles（处理名称/缩写）
    pred_canon = set()
    for sm in pred_smiles:
        sm_low = sm.lower()
        # 检查是否是名称/缩写
        found = False
        for canon_smiles, aliases in COMMON_CHEMICALS.items():
            if any(a.lower() == sm_low or a.lower() in sm_low for a in aliases):
                pred_canon.add(canon_smiles)
                found = True
                break
        if not found:
            # 作为SMILES处理
            if is_valid_smiles(sm):
                pred_canon.add(canonicalize_smiles(sm))

    # 2. Jaccard相似度
    intersection = len(pred_canon & gt_smiles)
    union = len(pred_canon | gt_smiles)

    if union == 0:
        return 1.0

    jaccard = intersection / union

    # 3. 检查是否有名称匹配（额外加分到0.8）
    if jaccard > 0:
        has_name_match = any(
            any(a.lower() in str(pred_smiles) for a in COMMON_CHEMICALS.get(gt_sm, []))
            for gt_sm in gt_smiles
        )
        if has_name_match:
            return min(jaccard / 0.8, 1.0)

    return jaccard


# ================== 用量匹配 ==================

# 单位转换映射（转换为克进行统一比较）
UNIT_CONVERSIONS = {
    'g': 1.0, 'mg': 0.001,
    'mol': 1.0, 'mmol': 0.001,
    'l': 1000.0, 'ml': 1.0, 'ul': 0.001
}


def normalize_quantity(value: float, unit: str) -> float:
    """
    将所有数量转换为克进行统一比较

    Args:
        value: 数值
        unit: 单位 (g, mg, mol, mmol, l, ml, ul)

    Returns:
        标准化为克的数值
    """
    return value * UNIT_CONVERSIONS.get(unit, 1.0)


def parse_qty(q: str) -> Optional[tuple[float, str]]:
    """
    解析数量字符串，返回 (数值, 单位) 元组

    Args:
        q: 数量字符串，如 "10 g", "0.55 mmol", "5 mL"

    Returns:
        (数值, 单位) 元组，如 (10.0, "g")
    """
    m = re.search(r'([\d.]+(?:\.\d+)?)\s*(g|mg|mmol|ml|ul|l|mol)', q.lower())
    if m:
        return (float(m.group(1)), m.group(2).lower())
    return None


def match_quantity(pred_qts: List[str], gt_qts: List[str]) -> float:
    """
    用量匹配：数值和单位都必须匹配，容差±20%

    重要：单位不同直接返回 0.0，不再允许 10 ml 匹配 10 g
    """
    if not pred_qts or not gt_qts:
        return 0.0

    pred_parsed = [parse_qty(q) for q in pred_qts if parse_qty(q)]
    gt_parsed = [parse_qty(q) for q in gt_qts if parse_qty(q)]

    for pv, pu in pred_parsed:
        for gv, gu in gt_parsed:
            # 单位必须完全匹配
            if pu != gu:
                continue
            # 标准化后比较，容差±20%
            pv_norm = normalize_quantity(pv, pu)
            gv_norm = normalize_quantity(gv, gu)
            if abs(pv_norm - gv_norm) <= max(0.2 * gv_norm, 0.01):
                return 1.0

    return 0.0

    pred_vals = [q for q in (parse_qty(q) for q in pred_qts) if q is not None]
    gt_vals = [q for q in (parse_qty(q) for q in gt_qts) if q is not None]

    for pv in pred_vals:
        for gv in gt_vals:
            if abs(pv - gv) <= max(0.2 * gv, 0.01):
                return 1.0

    return 0.0


# ================== Sequence Score ==================

import difflib

def compute_sequence_score(pred_procs: List[str], gt_procs: List[str]) -> float:
    """
    步骤序列得分

    1. 解析每个procedure
    2. 提取动作序列进行LCS对齐
    3. 对每个匹配步骤计算内容得分
    4. 平均得分 * 0.5
    """
    # 1. 解析
    pred_steps = [parse_procedure(p) for p in pred_procs]
    gt_steps = [parse_procedure(g) for g in gt_procs]

    if not gt_steps:
        return 0.0

    # 2. 提取动作序列
    pred_actions = [s["action"] for s in pred_steps]
    gt_actions = [s["action"] for s in gt_steps]

    # 3. LCS对齐
    matcher = difflib.SequenceMatcher(None, pred_actions, gt_actions)
    matches = list(matcher.get_matching_blocks())

    # 4. 计算匹配步骤得分
    total_score = 0.0

    for match in matches:
        i, j, size = match.a, match.b, match.size
        for k in range(size):
            p_step = pred_steps[i + k]
            g_step = gt_steps[j + k]

            # YIELD：只检查SMILES，不检查用量
            if g_step["action"] == "YIELD":
                mol_score = match_smiles_set(p_step["smiles"], g_step["smiles"])
                step_score = 0.7 * mol_score + 0.3
            else:
                # 其他动作
                mol_score = match_smiles_set(p_step["smiles"], g_step["smiles"])
                qty_score = match_quantity(p_step["quantities"], g_step["quantities"])
                step_score = 0.7 * mol_score + 0.3 * qty_score

            total_score += step_score

    # 5. 除以GT步骤数（未匹配的步骤得0分）
    return total_score / len(gt_steps)


# ================== Penalty ==================

def compute_penalty(output: str) -> float:
    """违规惩罚，累加"""
    penalty = 0.0

    answer_match = re.search(r'<answer>(.*?)</answer>', output, re.DOTALL)
    if not answer_match:
        return 0.0
    answer_part = answer_match.group(1)

    # 1. 非法动作
    procs = re.findall(r'<procedure>(\w+)', answer_part, re.IGNORECASE)
    for action in procs:
        if action.upper() not in ALLOWED_ACTIONS:
            penalty += 0.1

    # 2. 非原子性（一个procedure有多个动作）
    for proc_text in re.findall(r'<procedure>(.*?)</procedure>', answer_part, re.DOTALL):
        clean = re.sub(r'<[^>]+>', ' ', proc_text).strip()
        words = [w.upper() for w in clean.split()]
        illegal_count = sum(1 for w in words if w in ALLOWED_ACTIONS)
        if illegal_count > 1:
            penalty += 0.05

    # 3. think为空或太短
    think_match = re.search(r'<think>(.*?)</think>', output, re.DOTALL)
    if think_match:
        think_content = re.sub(r'\s+', '', think_match.group(1))
        if len(think_content) < 20:
            penalty += 0.15

    return penalty


# ================== 主函数 ==================

def compute_score(data_source: str, solution_str: str, ground_truth: Dict, extra_info: Optional[Dict] = None) -> float:
    """
    完整的reward计算函数，兼容verl接口

    Args:
        data_source: 数据源名称
        solution_str: 模型输出
        ground_truth: 包含{'thinking': str, 'actions': list, 'molecules': str}
        extra_info: 额外信息

    Returns:
        reward: float in [-0.5, 0.5]
    """
    # 1. 格式检查（传入molecules_json）
    format_score = check_format(solution_str, ground_truth.get('molecules', '{}'))

    # 2. 提取procedure列表
    answer_match = re.search(r'<answer>(.*?)</answer>', solution_str, re.DOTALL)
    if not answer_match:
        return -0.1  # 格式错误

    pred_procs = re.findall(r'<procedure>(.*?)</procedure>', answer_match.group(1), re.DOTALL)

    # 处理gt_actions
    gt_procs_list = ground_truth.get('actions', [])
    if isinstance(gt_procs_list, list):
        gt_procs = gt_procs_list
    else:
        import numpy as np
        gt_procs = gt_procs_list.tolist()

    # 3. Step-level化学奖励（90%权重）
    step_rewards, step_reward = compute_step_level_reward(
        pred_procs, gt_procs, ground_truth.get('molecules', '{}')
    )

    # 4. 惩罚
    penalty = compute_penalty(solution_str)

    # 5. 最终得分：0.1 * format + 0.9 * step_reward - penalty
    reward = 0.1 * format_score + 0.9 * step_reward - penalty

    return round(max(-1.0, min(1.0, reward)), 4)
