"""
reward/reward_function.py
"""

import re
import math
import json
import difflib
from typing import List, Dict, Tuple, Optional, Set

# 1. 强制静默 RDKit 日志
try:
    from rdkit import Chem, rdBase
    rdBase.DisableLog('rdApp.*') 
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

try:
    from .utils import ChemParser, ExperimentStep, ALLOWED_ACTIONS, BARRIER_ACTIONS, BAG_ACTIONS
except ImportError:
    from utils import ChemParser, ExperimentStep, ALLOWED_ACTIONS, BARRIER_ACTIONS, BAG_ACTIONS

WEIGHT_CONFIG = {'skeleton': 0.90, 'flesh': 0.10}
DECAY_GAMMA = 0.85

# ============================================================================
# 1. 化学物质解析器 (增强模糊匹配)
# ============================================================================

class MoleculeResolver:
    def __init__(self, molecules_map_str: str):
        self.name_to_smiles = {}
        if molecules_map_str:
            try:
                raw_map = json.loads(molecules_map_str)
                for k, v in raw_map.items():
                    self.name_to_smiles[k.strip().lower()] = v
            except:
                pass
        
        # 基础映射
        basic_map = {
            "water": "O", "methanol": "CO", "ethanol": "CCO", 
            "dichloromethane": "ClCCl", "dcm": "ClCCl",
            "ethyl acetate": "CCOC(C)=O", "etOAc": "CCOC(C)=O",
            "dmso": "CS(C)=O", "thf": "C1CCOC1", "ether": "CCOCC"
        }
        for k, v in basic_map.items():
            if k not in self.name_to_smiles:
                self.name_to_smiles[k] = v

    def canonicalize(self, smiles: str) -> str:
        if not HAS_RDKIT: return smiles.strip()
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return smiles.strip()
            return Chem.MolToSmiles(mol, isomericSmiles=False)
        except:
            return smiles.strip()

    def resolve(self, raw_chems: Set[str]) -> Set[str]:
        resolved_set = set()
        for item in raw_chems:
            item_clean = item.strip()
            item_lower = item_clean.lower()
            
            # 路径A: 精确匹配名称
            if item_lower in self.name_to_smiles:
                resolved_set.add(self.canonicalize(self.name_to_smiles[item_lower]))
                continue
            
            # 路径B: 模糊包含匹配 (解决 "ethyl acetate twice" 这种问题)
            found_fuzzy = False
            # 按长度倒序排列关键词，优先匹配长词（如 "diethyl ether" 优于 "ether"）
            sorted_keys = sorted(self.name_to_smiles.keys(), key=len, reverse=True)
            for key in sorted_keys:
                if key in item_lower and len(key) > 2: # 长度限制防止误伤
                    resolved_set.add(self.canonicalize(self.name_to_smiles[key]))
                    found_fuzzy = True
                    break
            if found_fuzzy: continue

            # 路径C: 尝试作为 SMILES 解析
            if HAS_RDKIT and len(item_clean) > 0:
                try:
                    mol = Chem.MolFromSmiles(item_clean)
                    if mol:
                        resolved_set.add(Chem.MolToSmiles(mol, isomericSmiles=False))
                        continue
                except:
                    pass
            
            # 路径D: 最终回退
            resolved_set.add(item_clean)
            
        return resolved_set


# ============================================================================
# 2. 评分核心
# ============================================================================

class ChemScorer:
    @staticmethod
    def _clean_text(text: str) -> str:
        """移除 Markdown 噪音"""
        return re.sub(r'```[a-zA-Z]*\n?|```', '', text)

    @staticmethod
    def check_fatal_errors(text: str, pred_steps: List[ExperimentStep], gt_len: int) -> Tuple[bool, float, str]:
        if not re.search(r'<answer>.*?</answer>', text, re.DOTALL):
            return True, 0.0, "Missing <answer>"
        
        think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
        if not think_match:
            return True, 0.0, "Missing <think>"
        
        if think_match and len(think_match.group(1).strip()) < 10:
            return True, 0.0, "Think too short"

        if not pred_steps:
            return True, 0.0, "No valid steps"

        # 闭合性
        open_count = len(re.findall(r'<procedure>', text))
        close_count = len(re.findall(r'</procedure>', text))
        if open_count > 0 and (open_count - close_count) / open_count > 0.5:
            return True, 0.0, "Unclosed procedures"

        # 裸露实体检测
        ans_content = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL).group(1)
        stripped = re.sub(r'<(time|temp)>.*?</\1>', ' ', ans_content, flags=re.IGNORECASE)
        stripped = re.sub(r'\[MOL\].*?\[/MOL\]', ' ', stripped, flags=re.IGNORECASE)
        
        naked_times = re.findall(r'\b(?:\d+\s*(?:min|h|hour|sec)|overnight)\b', stripped, re.IGNORECASE)
        naked_temps = re.findall(r'\b(?:\d+\s*(?:C|°C)|room\s*temperature|rt|r\.t\.)\b', stripped, re.IGNORECASE)
        
        expected = max(1, len(pred_steps) * 0.5)
        if len(naked_times) / expected > 0.5 or len(naked_temps) / expected > 0.5:
            return True, 0.0, "Naked entities"

        if gt_len > 0:
            p_len = len(pred_steps)
            if p_len > 2.0 * gt_len: return True, -0.5, "Loop"
            if p_len < 0.2 * gt_len: return True, 0.0, "Too short"

        return False, 0.0, ""

    @staticmethod
    def _check_step_accuracy(p: ExperimentStep, g: ExperimentStep, resolver: MoleculeResolver) -> float:
        if p.action != g.action: return 0.0
        score = 1.0
        
        p_res = resolver.resolve(p.chemicals)
        g_res = resolver.resolve(g.chemicals)
        if p_res != g_res:
            if p_res & g_res: score *= 0.5
            else: score *= 0.3
        
        # 时温
        for attr in ['time', 'temp']:
            g_type = getattr(g, f'{attr}_type')
            p_type = getattr(p, f'{attr}_type')
            if g_type != 'none':
                match = False
                if g_type == 'numeric' and p_type == 'numeric':
                    g_val = getattr(g, f'{attr}_val')
                    p_val = getattr(p, f'{attr}_val')
                    if attr == 'time':
                        if p_val and abs(p_val - g_val) <= (g_val * 0.1 + 1.0): match = True
                    else: # temp
                        if p_val[0] <= g_val[1] + 5 and p_val[1] >= g_val[0] - 5: match = True
                elif p_type != 'none': match = True
                if not match: score *= 0.5
        return score

    @staticmethod
    def _score_step_flesh(p: ExperimentStep, g: ExperimentStep) -> float:
        # 确保完全一致性
        if p.clean_text == g.clean_text: return 1.0
        return difflib.SequenceMatcher(None, p.clean_text, g.clean_text).ratio()

def compute_score(data_source: str, solution_str: str, ground_truth: Dict, extra_info: Optional[Dict] = None) -> float:
    resolver = MoleculeResolver(ground_truth.get('molecules', '{}'))
    
    # 清洗
    clean_sol = ChemScorer._clean_text(solution_str)
    gt_actions = ground_truth.get('actions', [])
    clean_gt_actions = [ChemScorer._clean_text(str(a)) for a in gt_actions]
    
    # 解析
    gt_steps = [ChemParser.parse(t) for t in clean_gt_actions]
    pred_raw = ChemParser.extract_procedures(clean_sol)
    pred_steps = [ChemParser.parse(t) for t in pred_raw]

    # 致命错误
    is_fatal, fatal_score, _ = ChemScorer.check_fatal_errors(clean_sol, pred_steps, len(gt_steps))
    if is_fatal: return fatal_score

    # 匹配
    p_ptr, g_ptr = 0, 0
    total_sk, total_fl = 0.0, 0.0
    error_count, consecutive = 0, 0

    while g_ptr < len(gt_steps):
        g_s = gt_steps[g_ptr]
        matched = False
        
        # Bag Logic
        if g_s.action in BAG_ACTIONS and p_ptr < len(pred_steps) and pred_steps[p_ptr].action in BAG_ACTIONS:
            g_bag, g_end = ChemScorer._get_bag_info(gt_steps, g_ptr, resolver)
            p_bag, p_end = ChemScorer._get_bag_info(pred_steps, p_ptr, resolver)
            if g_bag == p_bag and g_bag:
                decay = pow(DECAY_GAMMA, error_count)
                size = float(g_end - g_ptr)
                total_sk += size * decay
                total_fl += size * decay
                g_ptr, p_ptr = g_end, p_end
                consecutive += int(size)
                error_count = max(0, error_count - 2)
                continue

        # Window Search
        best_idx, best_sk = -1, 0.0
        for k in range(p_ptr, min(p_ptr + 3, len(pred_steps))):
            sk = ChemScorer._check_step_accuracy(pred_steps[k], g_s, resolver)
            if sk > best_sk:
                best_sk, best_idx = sk, k
                if sk == 1.0: break
        
        if best_idx != -1:
            decay = pow(DECAY_GAMMA, error_count)
            total_sk += best_sk * decay
            if best_sk >= 0.9:
                total_fl += ChemScorer._score_step_flesh(pred_steps[best_idx], g_s) * decay
            p_ptr, matched = best_idx + 1, True
        
        if matched:
            consecutive += 1
            if consecutive >= 2: error_count = max(0, error_count - 1); consecutive = 0
        else:
            error_count = min(error_count + 1, 2); consecutive = 0
        g_ptr += 1

    content_score = (WEIGHT_CONFIG['skeleton'] * (total_sk/len(gt_steps)) + 
                     WEIGHT_CONFIG['flesh'] * (total_fl/len(gt_steps)))
    
    ratio = len(pred_steps) / len(gt_steps) if len(gt_steps) > 0 else 0
    return round(float(max(-1.0, min(1.0, content_score * math.exp(-1.5 * abs(1.0 - ratio))))), 4)

# 补全辅助函数
def _get_bag_info_wrapper(steps, start, resolver):
    return ChemScorer._get_bag_info(steps, start, resolver)
ChemScorer._get_bag_info = staticmethod(lambda steps, start, res: (
    MoleculeResolver.resolve(res, {s.chemicals.pop() if s.chemicals else '' for s in steps[start:] if s.action in BAG_ACTIONS}) # 简化处理
    if False else reward_function_bag_helper(steps, start, res)
))

def reward_function_bag_helper(steps, start, resolver):
    chems = set()
    idx = start
    while idx < len(steps) and steps[idx].action in BAG_ACTIONS and steps[idx].action not in BARRIER_ACTIONS:
        chems.update(resolver.resolve(steps[idx].chemicals))
        idx += 1
    return chems, idx
ChemScorer._get_bag_info = staticmethod(reward_function_bag_helper)