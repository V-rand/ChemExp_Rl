import json
import requests

# ======================= CONFIG 配置区 =======================
CONFIG = {
    "api_url": "http://localhost:8002/v1/chat/completions",
    "model_name": "chem_expert_v1",
    "test_data_path": "/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed/chem_test_inference.jsonl",
    "temperature": 0.7,
    "target_index": 1,    # 修改此处查看不同的样本
}

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

def inspect():
    target = None
    target_index = CONFIG["target_index"]
    with open(CONFIG["test_data_path"], 'r') as f:
        for idx,line in enumerate(f,0):
            if idx == target_index:
                item = json.loads(line)
                target = item
                break
    
    if not target:
        print(f"未找到 index: {CONFIG['target_index']}"); return

    print(f"正在请求模型对 index {CONFIG['target_index']} 的回答...")
    payload = {
        "model": CONFIG["model_name"],
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + target["prompt"],
        "extra_body": {
            "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        "temperature": CONFIG["temperature"]
    }
    
    try:
        response = requests.post(CONFIG["api_url"], json=payload)
        resp = response.json()
        
        # 检查是否有 choices 键
        if 'choices' in resp:
            model_content = resp['choices'][0]['message']['content']
            model_output = model_content.strip()  # 去除首尾空白字符
            print("\n" + " 模型回答 (THINK & ANSWER) ".center(80, "="))
            print(model_output)
            print("\n" + " 原始 Ground Truth ".center(80, "="))
            print(target["gt"])
            print("="*80)
        else:
            # 如果没有 choices，打印 vLLM 返回的完整错误
            print("\n" + " vLLM 服务报错 ".center(80, "!"))
            print(json.dumps(resp, indent=2, ensure_ascii=False))
            print("!"*80)
            
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    inspect()