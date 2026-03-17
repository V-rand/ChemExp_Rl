import json
import re
import asyncio
import numpy as np
from openai import AsyncOpenAI
from typing import List, Dict, Tuple, Optional
from tqdm.asyncio import tqdm
from rdkit import Chem

# ======================= CONFIG 配置区 =======================
CONFIG = {
    "api_url": "http://localhost:8000/v1",
    "model_name": "chem_expert_v1",
    "test_data_path": "/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed/chem_test_inference.jsonl",
    
    "num_samples": 1000,       # 需要测试的样本量 (例如 100, 6000)
    "k_attempts": 1,          # 每个样本推理 K 次取平均
    
    "temperature": 0.9,       # RL模型评测建议 0.7-1.0
    "top_p": 0.9,
    "max_tokens": 1024,
    "concurrency": 70,        # 并发请求数，vLLM 性能强可调高

    "save_success_cases": True,       # 是否保存对齐正确的样本
    "success_threshold": 0.05,         # 只有指标 A 分数 >= 该阈值时才保存 (1.0 代表 100% 对齐)
    "output_json_path": "correct_results.json"
}

SYSTEM_PROMPT = """You are a chemical synthesis expert. Design a precise experimental procedure from the provided reaction information.
#INPUT
Reactants, product, solvent, catalyst and target yield will be given.
#OUTPUT
Return exactly two sections and nothing else:
<think>
Chemical reasoning.
Analyze reaction mechanism and how to design experiment.
</think>

<answer>
Robot-executable procedure.
</answer>

#EXECUTION RULES
• Each step must start with one action from:
ADD, STIR, WAIT, CONCENTRATE, YIELD, MAKESOLUTION, FILTER, WASH, DRYSOLUTION, COLLECTLAYER, EXTRACT, SETTEMP, REFLUX, RECRYSTAL, PHASESEPA, PH, PURIFY, QUENCH, PARTITION, TRITURATE, DRYSOLID, DEGAS, MICROWAVE, SONICATE
• Each step must be wrapped in <procedure>...</procedure>
• Time format: <time>...</time>
• Temperature format: <temp>...</temp>
• Use [MOL] ```SMILES``` [/MOL](Quantity) for chemicals.
• Only one physical action per <procedure>.
- When preparing multiple solutions separately in an experiment, name them as (Solution A), (Solution B), etc.Clearly specify when referencing them later. Example: MAKESOLUTION (Solution A) with ...; MAKESOLUTION (Solution B) with ...; ADD Solution B to Solution A; 
• Final step must be YIELD.

#Example format:
<think>
Reasoning...
</think>

<answer>
<procedure>MAKESOLUTION(Solution A) with [MOL] ```SMILES``` [/MOL] (x g, x mol) in [MOL] ```SMILES``` [/MOL] (x mL)</procedure>
<procedure>ADD [MOL] ```SMILES``` [/MOL] (x g, x mmol)</procedure>
<procedure>STIR for <time>2 h</time> at <temp>room temperature</temp></procedure>
<procedure>MAKESOLUTION(Solution B) with [MOL] ```SMILES``` [/MOL] (x mg,x mmol) and [MOL] ```SMILES``` [/MOL](x ul) in [MOL] ```SMILES``` [/MOL] (x mL)</procedure>
<procedure>ADD Solution B to Solution A at <temp>20 °C</temp></procedure>
<procedure>STIR for <time>1 hour</time></procedure>
<procedure>EXTRACT with [MOL] ```CCOC(C)=O``` [/MOL]</procedure>
<procedure>WASH with [MOL] ```O.[Cl-].[Na+]``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```SMILES``` [/MOL] (0.8 g, 80%)</procedure>
</answer>"""

# ======================= 工具函数 =======================
# ================== 1. 扩展的化学品映射表 ==================
# 建立从 别名 -> 标准SMILES 的反向映射
CHEMICAL_SYNONYMS = {
    # --- 溶剂 ---
    "c1ccoc1": ["thf", "tetrahydrofuran", "四氢呋喃"],
    "cn(c)c=o": ["dmf", "dimethylformamide", "n,n-dimethylformamide"],
    "clcccl": ["dcm", "dichloromethane", "methylene chloride", "ch2cl2"],
    "co": ["meoh", "methanol", "methyl alcohol"],
    "cco": ["etoh", "ethanol", "ethyl alcohol"],
    "ccocc": ["diethyl ether", "ether", "ethyl ether", "乙醚"],
    "ccoc(c)=o": ["etoac", "ethyl acetate", "ea"],
    "c1ccccc1": ["benzene"],
    "cl1ccccc1": ["chlorobenzene", "phcl"],
    "c1cocco1": ["dioxane", "1,4-dioxane"],
    
    # --- 无机试剂 & 盐类 ---
    "o=c([o-])[o-].[k+].[k+]": ["k2co3", "potassium carbonate", "碳酸钾"],
    "[na+].[oh-]": ["naoh", "sodium hydroxide"],
    "[k+].[oh-]": ["koh", "potassium hydroxide"],
    "[nh4+].[cl-]": ["ammonium chloride", "nh4cl"],
    "[mg+2].[cl-].[cl-]": ["mgcl2", "magnesium chloride"],
    "[mg+2].[br-].[br-]": ["mgbr2", "magnesium bromide"],
    "[na+].[cl-]": ["nacl", "sodium chloride"],
    "[k+].[cl-]": ["kcl", "potassium chloride"],
    "o=c([o-])[o-].[na+].[na+]": ["na2co3", "sodium carbonate"],
    "o=s(=o)([o-])[o-].[na+].[na+]": ["na2so4", "sodium sulfate"],
    "[na+].[hco3-]": ["nahco3", "sodium bicarbonate"],
    "o=c([o-])o.[na+]": ["sodium bicarbonate", "nahco3"],
    "cc(=o)o[na+]": ["sodium acetate", "naoac"],
    "[na+].[o-]s(=o)(=o)c1ccc(cc1)cc": ["sodium tosylate"],
    
    # --- 碱和催化剂 ---
    "ccn(cc)cc": ["tea", "triethylamine", "et3n", "三乙胺"],
    "ccn(cc)c(c)(c)c": ["dipea", "diisopropylethylamine", "n,n-diisopropylethylamine", "diea"],
    "c1ccccn1": ["pyridine", "py"],
    "c1ccncc1": ["pyridine"],
    "[pd]": ["palladium", "pd"],
    "[pt]": ["platinum", "pt"],
    
    # --- 酸 ---
    "cc(=o)o": ["acetic acid", "hoac", "aacoh"],
    "o=c(o)c(f)(f)f": ["tfa", "trifluoroacetic acid"],
    "o=s(=o)(o)o": ["h2so4", "sulfuric acid"],
    "cl": ["hcl", "hydrochloric acid", "chloride"],
    "[cl-]": ["hcl", "chloride", "hydrochloric acid"],
    
    # --- 其他常见试剂 ---
    "o": ["water", "h2o", "水"],
    "n": ["ammonia", "nh3", "氨"],
    "c": ["carbon"],
    "[c-]#[n+]": ["cyanide", "cn-"],
    "[n+]#[c-]": ["isocyanide", "nc-"]
}

# 在脚本初始化部分执行一次
ALIAS_TO_SMILES = {}
for smiles, aliases in CHEMICAL_SYNONYMS.items():
    for alias in aliases:
        # 存入小写版本，确保模型输出 EtOAc 或 etoac 都能匹配
        ALIAS_TO_SMILES[alias.lower()] = smiles

def get_canon(text: str) -> str:
    if not text: return ""
    # 移除末尾可能的状态标注如 (aq)
    clean_text = re.sub(r'\(.*?\)$', '', text.strip()).lower()
    
    # 1. 别名表优先
    if clean_text in ALIAS_TO_SMILES:
        return ALIAS_TO_SMILES[clean_text]
    
    # 2. RDKit 标准化
    mol = Chem.MolFromSmiles(text.strip())
    if mol: return Chem.MolToSmiles(mol, canonical=True)
        
    return clean_text

def parse_steps(text: str) -> List[Dict]:
    steps = []
    ans_content = re.search(r'<answer>(.*?)</answer>', text, re.S)
    content = ans_content.group(1) if ans_content else text
    procs = re.findall(r'<procedure>(.*?)</procedure>', content, re.S)
    for p in procs:
        action = (re.search(r'^\s*(\w+)', p.strip()) or [None, ""])[1].upper()
        mols = {get_canon(m) for m in re.findall(r'\[MOL\]\s*```(.*?)```\s*\[/MOL\]', p)}
        qtys = []
        for v, u in re.findall(r'(\d+\.?\d*)\s*([a-zA-Zμ]+)', p):
            u_l = u.lower()
            conv = {"g":1.0, "mg":0.001, "kg":1000.0, "ml":1.0, "l":1000.0, "ul":0.001, "mmol":1.0, "mol":1000.0}
            if u_l in conv:
                dim = "mass" if u_l in ["g","mg","kg"] else ("vol" if u_l in ["ml","l","ul"] else "amt")
                qtys.append((float(v)*conv[u_l], dim))
        steps.append({"action": action, "mols": mols, "qtys": qtys, "has_cond": "<time>" in p or "<temp>" in p})
    return steps

def evaluate_single_step(pred: Dict, gt: Dict, check_qty: bool) -> bool:
    if pred['action'] != gt['action'] or pred['mols'] != gt['mols'] or pred['has_cond'] != gt['has_cond']:
        return False
    if check_qty:
        for g_v, g_d in gt['qtys']:
            matched = any(abs(p_v - g_v)/(g_v+1e-6) < 0.15 for p_v, p_d in pred['qtys'] if p_d == g_d)
            if not matched: return False
    return True

# ======================= 异步评测核心 =======================
async def fetch_task(client, model, messages, semaphore):
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages, 
                temperature=CONFIG["temperature"], top_p=CONFIG["top_p"], max_tokens=CONFIG["max_tokens"]
            )
            return resp.choices[0].message.content
        except Exception as e: return f"Error: {e}"

async def run_eval():
    client = AsyncOpenAI(api_key="EMPTY", base_url=CONFIG["api_url"])
    semaphore = asyncio.Semaphore(CONFIG["concurrency"])
    
    with open(CONFIG["test_data_path"], 'r') as f:
        all_data = [json.loads(line) for line in f][:CONFIG["num_samples"]]

    print(f"开始评测: {len(all_data)} 样本 x {CONFIG['k_attempts']} 尝试")
    
    tasks = []
    for item in all_data:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + item["prompt"]
        for _ in range(CONFIG["k_attempts"]):
            tasks.append(fetch_task(client, CONFIG["model_name"], msgs, semaphore))
    
    responses = await tqdm.gather(*tasks, desc="vLLM 推理中")
    
    scores_A, scores_B = [], []
    success_logs = []  # 用于保存符合阈值的结果

    for i, item in enumerate(all_data):
        gt_steps = parse_steps(item["gt"])
        # 获取该样本对应的所有 K 次尝试结果
        sample_res = responses[i*CONFIG["k_attempts"] : (i+1)*CONFIG["k_attempts"]]
        
        sample_scores_A, sample_scores_B = [], []
        for pred_text in sample_res:
            pred_steps = parse_steps(pred_text)
            correct_A, correct_B = 0, 0
            
            # 指标 A 评估
            for j in range(len(gt_steps)):
                if j < len(pred_steps) and evaluate_single_step(pred_steps[j], gt_steps[j], False): correct_A += 1
                else: break
            
            # 指标 B 评估
            for j in range(len(gt_steps)):
                if j < len(pred_steps) and evaluate_single_step(pred_steps[j], gt_steps[j], True): correct_B += 1
                else: break
            
            score_a = correct_A / len(gt_steps) if len(gt_steps) > 0 else 0
            score_b = correct_B / len(gt_steps) if len(gt_steps) > 0 else 0
            sample_scores_A.append(score_a)
            sample_scores_B.append(score_b)

            # --- 核心逻辑：判断是否达到保存阈值 ---
            if CONFIG["save_success_cases"] and score_a >= CONFIG["success_threshold"]:
                success_logs.append({
                    "score_A": score_a,
                    "score_B": score_b,
                    "model_output": pred_text,
                    "ground_truth": item["gt"]
                })
            
        scores_A.append(np.mean(sample_scores_A))
        scores_B.append(np.mean(sample_scores_B))

    # 保存文件
    if CONFIG["save_success_cases"]:
        with open(CONFIG["output_json_path"], 'w', encoding='utf-8') as f:
            json.dump(success_logs, f, ensure_ascii=False, indent=2)
        print(f"\n已保存 {len(success_logs)} 条符合条件的记录至: {CONFIG['output_json_path']}")

    print(f"\n" + "="*50)
    print(f"评测报告 (K={CONFIG['k_attempts']})")
    print(f"指标 A (动作+物质+条件) 平均对齐率: {np.mean(scores_A):.2%}")
    print(f"指标 B (动作+物质+条件+用量) 平均对齐率: {np.mean(scores_B):.2%}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_eval())