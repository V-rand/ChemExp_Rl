"""
使用真实数据测试奖励函数
从data/processed中加载数据
"""

import sys
sys.path.insert(0, '/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl')

import json
import pandas as pd
from reward.reward_function import compute_score_with_details, compute_score


def load_sample_data(n=3):
    """加载样本数据"""
    try:
        df = pd.read_parquet('/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed/train.parquet')
        return df.head(n)
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None


def get_ground_truth(row):
    """从row中提取ground truth"""
    reward_model = row.get('reward_model', {})
    gt_data = reward_model.get('ground_truth', {})
    
    actions = gt_data.get('actions', [])
    if hasattr(actions, 'tolist'):
        actions = actions.tolist()
    
    # 从prompt构建字符串
    prompt_arr = row.get('prompt', [])
    prompt_str = ''
    if hasattr(prompt_arr, 'tolist'):
        prompt_arr = prompt_arr.tolist()
        for msg in prompt_arr:
            if isinstance(msg, dict) and 'content' in msg:
                prompt_str += msg.get('content', '') + '\n'
    
    return {
        "actions": actions,
        "prompt": prompt_str,
        "thinking": gt_data.get('thinking', ''),
        "molecules": gt_data.get('molecules', '{}')
    }


def test_real_data():
    """测试真实数据"""
    print("=" * 70)
    print("真实数据测试")
    print("=" * 70)
    
    df = load_sample_data(5)
    if df is None:
        print("无法加载数据，跳过测试")
        return
    
    print(f"加载了 {len(df)} 条数据\n")
    
    for idx, row in df.iterrows():
        print(f"\n{'='*70}")
        print(f"样本 {idx + 1}:")
        print(f"{'='*70}")
        
        ground_truth = get_ground_truth(row)
        
        print(f"GT步骤数: {len(ground_truth['actions'])}")
        print(f"动作序列: {', '.join([a.split()[0] for a in ground_truth['actions'][:5]])}{'...' if len(ground_truth['actions']) > 5 else ''}")
        
        # 使用GT本身作为预测（完美匹配）
        gt_actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in ground_truth['actions']])
        perfect_solution = f"""<think>
{ground_truth['thinking'][:200]}...
Therefore, the validated operational sequence is: {', '.join([a.split()[0] for a in ground_truth['actions']])}
</think>
<answer>
{gt_actions_str}
</answer>"""
        
        result = compute_score_with_details("train", perfect_solution, ground_truth)
        print(f"\n完美匹配得分:")
        print(f"  总分: {result['score']}")
        print(f"  R_format: {result['r_format']}")
        print(f"  R_lcs: {result['r_lcs']}")
        print(f"  R_step: {result['r_step']}")
        print(f"  LCS长度: {result['lcs_length']}")
        
        # 测试缺少一个步骤
        if len(ground_truth['actions']) > 2:
            incomplete_actions = ground_truth['actions'][:-1]  # 去掉最后一步
            incomplete_str = '\n'.join([f'<procedure>{a}</procedure>' for a in incomplete_actions])
            incomplete_solution = f"""<think>
Thinking...
Therefore, the validated operational sequence is: {', '.join([a.split()[0] for a in incomplete_actions])}
</think>
<answer>
{incomplete_str}
</answer>"""
            
            result2 = compute_score_with_details("train", incomplete_solution, ground_truth)
            print(f"\n缺少最后一步得分:")
            print(f"  总分: {result2['score']}")
            print(f"  R_lcs: {result2['r_lcs']}")
            print(f"  R_step: {result2['r_step']}")
        
        # 只打印前2个样本的详细信息
        if idx >= 1:
            break


def test_various_predictions():
    """测试各种预测情况"""
    print("\n" + "=" * 70)
    print("各种预测情况测试")
    print("=" * 70)
    
    # 使用第一条真实数据作为GT
    df = load_sample_data(1)
    if df is None:
        return
    
    row = df.iloc[0]
    ground_truth = get_ground_truth(row)
    
    print(f"GT步骤数: {len(ground_truth['actions'])}")
    
    gt_actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in ground_truth['actions']])
    
    test_cases = [
        ("完全正确", f"""<think>
{ground_truth['thinking'][:200]}...
Therefore, the validated operational sequence is: {', '.join([a.split()[0] for a in ground_truth['actions']])}
</think>
<answer>
{gt_actions_str}
</answer>"""),
        
        ("缺少think", f"""<answer>
{gt_actions_str}
</answer>"""),
        
        ("缺少最后一步", f"""<think>Thinking...</think>
<answer>
{chr(10).join([f'<procedure>{a}</procedure>' for a in ground_truth['actions'][:-1]])}
</answer>"""),
        
        ("动作正确但第一个分子错误", f"""<think>Wrong molecules...</think>
<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CCCCC``` [/MOL] (1.0 g) in [MOL] ```C1CCOC1``` [/MOL]</procedure>
{chr(10).join([f'<procedure>{a}</procedure>' for a in ground_truth['actions'][1:]])}
</answer>"""),
        
        ("完全乱写", """<think>Blah blah...</think>
<answer>
<procedure>MIX everything together</procedure>
<procedure>COOK for 5 minutes</procedure>
<procedure>EAT the product</procedure>
</answer>"""),
    ]
    
    for name, solution in test_cases:
        result = compute_score_with_details("train", solution, ground_truth)
        print(f"\n{name}:")
        print(f"  总分: {result['score']}")
        print(f"  R_format: {result.get('r_format', 'N/A')}")
        print(f"  R_lcs: {result.get('r_lcs', 'N/A')}")
        print(f"  R_step: {result.get('r_step', 'N/A')}")


def test_edge_cases_real():
    """测试真实数据上的边界情况"""
    print("\n" + "=" * 70)
    print("边界情况测试")
    print("=" * 70)
    
    df = load_sample_data(1)
    if df is None:
        return
    
    row = df.iloc[0]
    ground_truth = get_ground_truth(row)
    
    test_cases = [
        ("空字符串", ""),
        ("只有空格", "   "),
        ("无XML标签", "This is just plain text without any tags"),
        ("只有think", "<think>Thinking...</think>"),
        ("只有answer", "<answer><procedure>ADD</procedure></answer>"),
        ("错误的动作", "<think>...</think><answer><procedure>MIX [MOL] ```CC``` [/MOL]</procedure><procedure>YIELD [MOL] ```CC``` [/MOL]</procedure></answer>"),
        ("全部小写", "<think>...</think><answer><procedure>add [MOL] ```CC``` [/MOL]</procedure><procedure>yield [MOL] ```CC``` [/MOL]</procedure></answer>"),
    ]
    
    for name, solution in test_cases:
        try:
            result = compute_score_with_details("train", solution, ground_truth)
            print(f"\n{name}: score = {result['score']}")
        except Exception as e:
            print(f"\n{name}: 错误 - {e}")


if __name__ == "__main__":
    test_real_data()
    test_various_predictions()
    test_edge_cases_real()
