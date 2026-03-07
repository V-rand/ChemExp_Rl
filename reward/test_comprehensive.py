"""
全面严苛测试 - 覆盖各种真实场景和模型错误情况
"""

import sys
sys.path.insert(0, '/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl')

import json
import random
from reward.reward_function import (
    compute_score, 
    compute_score_with_details,
    compute_format_reward,
    compute_lcs_reward,
    parse_procedure,
    ALLOWED_ACTIONS,
    smiles_to_canonical
)


# ================== 测试数据 ==================

# 真实GT数据 - 不同类型
REAL_GT_CASES = [
    {
        "name": "简单合成",
        "actions": [
            "MAKESOLUTION with [MOL] ```CCO``` [/MOL] (10 mL)",
            "ADD [MOL] ```CC(=O)O``` [/MOL] (1.0 g, 10 mmol)",
            "STIR for <time>2 h</time> at <temp>80 C</temp>",
            "CONCENTRATE",
            "YIELD [MOL] ```CC(=O)OCC``` [/MOL] (1.1 g, 85%)"
        ]
    },
    {
        "name": "多步纯化",
        "actions": [
            "MAKESOLUTION (Solution A) with [MOL] ```c1ccccc1``` [/MOL] (5.0 g) in [MOL] ```C1CCOC1``` [/MOL] (50 mL)",
            "ADD [MOL] ```[H-].[Na+]``` [/MOL] (60%, 2.5 g)",
            "STIR for <time>30 min</time> at <temp>0 C</temp>",
            "ADD [MOL] ```CC(C)I``` [/MOL] (6.0 g)",
            "STIR for <time>overnight</time> at <temp>room temperature</temp>",
            "QUENCH with [MOL] ```O``` [/MOL]",
            "EXTRACT with [MOL] ```ClCCl``` [/MOL]",
            "COLLECTLAYER organic",
            "WASH with [MOL] ```O.[Cl-].[Na+]``` [/MOL]",
            "DRYSOLUTION over [MOL] ```O=S(=O)([O-])[O-].[Mg+2]``` [/MOL]",
            "FILTER keep filtrate",
            "CONCENTRATE",
            "PURIFY by silica gel chromatography",
            "YIELD [MOL] ```CC(C)c1ccccc1``` [/MOL] (4.5 g, 75%)"
        ]
    },
    {
        "name": "无水反应",
        "actions": [
            "MAKESOLUTION with [MOL] ```C1CCOC1``` [/MOL] (100 mL) and [MOL] ```CC(C)(C)OC(=O)N1CCCCC1=O``` [/MOL] (10.0 g, 44.4 mmol)",
            "SETTEMP to <temp>-78 C</temp>",
            "ADD [MOL] ```[Li]CCCC``` [/MOL] (1.6 M, 30 mL) dropwise",
            "STIR for <time>1 h</time>",
            "ADD [MOL] ```CI``` [/MOL] (6.3 g, 44.4 mmol)",
            "WARM to <temp>room temperature</temp>",
            "QUENCH with [MOL] ```O``` [/MOL] (50 mL)",
            "EXTRACT with [MOL] ```CCOCC``` [/MOL]",
            "COLLECTLAYER organic",
            "DRYSOLUTION over [MOL] ```O=S(=O)([O-])[O-].[Mg+2]``` [/MOL]",
            "FILTER keep filtrate",
            "CONCENTRATE",
            "YIELD [MOL] ```CC1CCCN(C(=O)OC(C)(C)C)C1``` [/MOL] (8.5 g, 80%)"
        ]
    },
    {
        "name": "微波反应",
        "actions": [
            "MAKESOLUTION with [MOL] ```Nc1ccccc1``` [/MOL] (2.0 g, 21.5 mmol) and [MOL] ```C1CCOC1``` [/MOL] (20 mL)",
            "ADD [MOL] ```CC(=O)Cl``` [/MOL] (2.5 g, 32 mmol)",
            "MICROWAVE for <time>10 min</time> at <temp>120 C</temp>",
            "CONCENTRATE",
            "YIELD [MOL] ```CC(=O)Nc1ccccc1``` [/MOL] (2.8 g, 95%)"
        ]
    }
]


# ================== 模拟模型输出生成器 ==================

def generate_perfect_match(gt_actions):
    """生成完美匹配的输出"""
    actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in gt_actions])
    return f"""<think>
Analyzing the reaction mechanism and conditions...
Therefore, the validated operational sequence is: {', '.join([a.split()[0] for a in gt_actions])}
</think>
<answer>
{actions_str}
</answer>"""

def generate_missing_think(gt_actions):
    """缺少think标签"""
    actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in gt_actions])
    return f"""<answer>
{actions_str}
</answer>"""

def generate_missing_answer(gt_actions):
    """缺少answer标签"""
    return f"""<think>
Analyzing the reaction...
</think>
Some text here"""

def generate_naked_smiles(gt_actions):
    """裸露的SMILES"""
    return f"""<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION with CCO (10 mL)</procedure>
<procedure>ADD CC(=O)O (1.0 g)</procedure>
<procedure>STIR for <time>2 h</time> at <temp>80 C</temp></procedure>
<procedure>YIELD CC(=O)OCC</procedure>
</answer>"""

def generate_naked_conditions(gt_actions):
    """裸露的时间和温度"""
    return f"""<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```CCO``` [/MOL] (10 mL)</procedure>
<procedure>ADD [MOL] ```CC(=O)O``` [/MOL] (1.0 g)</procedure>
<procedure>STIR for 2 hours at 80°C</procedure>
<procedure>YIELD [MOL] ```CC(=O)OCC``` [/MOL]</procedure>
</answer>"""

def generate_wrong_order(gt_actions):
    """动作顺序错误"""
    if len(gt_actions) >= 3:
        wrong_order = [gt_actions[0], gt_actions[2], gt_actions[1]] + gt_actions[3:]
        actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in wrong_order])
        return f"""<think>Thinking...</think>
<answer>
{actions_str}
</answer>"""
    return generate_perfect_match(gt_actions)

def generate_missing_steps(gt_actions, n_missing=1):
    """缺少n个步骤"""
    if len(gt_actions) > n_missing:
        incomplete = gt_actions[:-n_missing]
        actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in incomplete])
        return f"""<think>Thinking...</think>
<answer>
{actions_str}
</answer>"""
    return generate_perfect_match(gt_actions)

def generate_extra_steps(gt_actions, n_extra=2):
    """添加额外步骤"""
    extra = gt_actions.copy()
    extra.insert(-1, "WAIT for <time>5 min</time>")
    extra.insert(-1, "STIR")
    actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in extra])
    return f"""<think>Thinking...</think>
<answer>
{actions_str}
</answer>"""

def generate_repeated_actions(gt_actions):
    """重复输出同一动作"""
    repeated = gt_actions[:2] + [gt_actions[2]] * 3 + gt_actions[3:]
    actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in repeated])
    return f"""<think>Thinking...</think>
<answer>
{actions_str}
</answer>"""

def generate_gibberish(gt_actions=None):
    """完全胡言乱语"""
    return """<think>Blah blah blah...</think>
<answer>
<procedure>MIX everything together</procedure>
<procedure>COOK for 5 minutes</procedure>
<procedure>EAT the product</procedure>
<procedure>DRINK the solution</procedure>
</answer>"""

def generate_partial_gibberish(gt_actions):
    """部分胡言乱语"""
    actions_str = '\n'.join([
        f'<procedure>{gt_actions[0]}</procedure>',
        '<procedure>DO something magical</procedure>',
        f'<procedure>{gt_actions[2] if len(gt_actions) > 2 else gt_actions[-1]}</procedure>',
        '<procedure>blah blah blah</procedure>',
        f'<procedure>{gt_actions[-1]}</procedure>'
    ])
    return f"""<think>Thinking...</think>
<answer>
{actions_str}
</answer>"""

def generate_wrong_molecules(gt_actions):
    """分子错误"""
    return f"""<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```CCCC``` [/MOL] (10 mL)</procedure>
<procedure>ADD [MOL] ```CC(=O)O``` [/MOL] (1.0 g)</procedure>
<procedure>STIR for <time>2 h</time> at <temp>80 C</temp></procedure>
<procedure>YIELD [MOL] ```CC(=O)OCC``` [/MOL]</procedure>
</answer>"""

def generate_wrong_quantities(gt_actions):
    """用量错误"""
    return f"""<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```CCO``` [/MOL] (100 mL)</procedure>
<procedure>ADD [MOL] ```CC(=O)O``` [/MOL] (10.0 g, 100 mmol)</procedure>
<procedure>STIR for <time>2 h</time> at <temp>80 C</temp></procedure>
<procedure>YIELD [MOL] ```CC(=O)OCC``` [/MOL] (0.1 g, 1%)</procedure>
</answer>"""

def generate_wrong_conditions(gt_actions):
    """条件错误"""
    return f"""<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```CCO``` [/MOL] (10 mL)</procedure>
<procedure>ADD [MOL] ```CC(=O)O``` [/MOL] (1.0 g)</procedure>
<procedure>STIR for <time>24 h</time> at <temp>150 C</temp></procedure>
<procedure>YIELD [MOL] ```CC(=O)OCC``` [/MOL]</procedure>
</answer>"""

def generate_english_names(gt_actions):
    """使用英文名而非SMILES"""
    return f"""<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION with ethanol (10 mL)</procedure>
<procedure>ADD acetic acid (1.0 g)</procedure>
<procedure>STIR for <time>2 h</time> at <temp>80 C</temp></procedure>
<procedure>YIELD ethyl acetate</procedure>
</answer>"""

def generate_illegal_actions(gt_actions=None):
    """非法动作"""
    return f"""<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```CCO``` [/MOL] (10 mL)</procedure>
<procedure>MIX [MOL] ```CC(=O)O``` [/MOL] (1.0 g)</procedure>
<procedure>COOK for <time>2 h</time></procedure>
<procedure>FRY at <temp>80 C</temp></procedure>
<procedure>YIELD [MOL] ```CC(=O)OCC``` [/MOL]</procedure>
</answer>"""

def generate_non_atomic(gt_actions):
    """非原子性操作"""
    return f"""<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```CCO``` [/MOL] and ADD [MOL] ```CC(=O)O``` [/MOL]</procedure>
<procedure>STIR and CONCENTRATE</procedure>
<procedure>YIELD [MOL] ```CC(=O)OCC``` [/MOL]</procedure>
</answer>"""

def generate_missing_yield(gt_actions):
    """缺少YIELD步骤"""
    actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in gt_actions[:-1]])
    return f"""<think>Thinking...</think>
<answer>
{actions_str}
</answer>"""

def generate_wrong_yield_mol(gt_actions):
    """YIELD分子错误"""
    if len(gt_actions) >= 1:
        wrong_yield = gt_actions[:-1] + ["YIELD [MOL] ```CCCC``` [/MOL] (1.0 g)"]
        actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in wrong_yield])
        return f"""<think>Thinking...</think>
<answer>
{actions_str}
</answer>"""
    return generate_perfect_match(gt_actions)

def generate_empty_think(gt_actions):
    """空的think"""
    actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in gt_actions])
    return f"""<think>

</think>
<answer>
{actions_str}
</answer>"""

def generate_case_mismatch(gt_actions):
    """大小写不匹配"""
    return f"""<THINK>Thinking...</THINK>
<ANSWER>
<PROCEDURE>add [MOL] ```CCO``` [/MOL] (10 mL)</PROCEDURE>
<PROCEDURE>stir for <TIME>2 h</TIME> at <TEMP>80 C</TEMP></PROCEDURE>
<PROCEDURE>yield [MOL] ```CC(=O)OCC``` [/MOL]</PROCEDURE>
</ANSWER>"""

def generate_truncated_output(gt_actions):
    """截断输出"""
    return f"""<think>
Analyzing the reaction...
Therefore, the validated operational sequence is: MAKESOLUTION, ADD, STIR"""

def generate_xml_injection(gt_actions=None):
    """XML注入攻击尝试"""
    return """<think>Thinking...</think>
<answer>
<procedure>ADD [MOL] ```CCO``` [/MOL]</procedure>
</answer>
<procedure>EXTRACTION</procedure>
<answer>
<procedure>YIELD [MOL] ```CC``` [/MOL]</procedure>
</answer>"""

def generate_unicode_smiles(gt_actions=None):
    """Unicode字符"""
    return """<think>Thinking...</think>
<answer>
<procedure>ADD [MOL] ```CCO🧪``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CC💊``` [/MOL]</procedure>
</answer>"""

def generate_very_long_output(gt_actions):
    """超长输出（啰嗦）"""
    base = ["STIR for <time>1 h</time>"] * 50
    actions_str = '\n'.join([f'<procedure>{a}</procedure>' for a in base])
    return f"""<think>Thinking...</think>
<answer>
{actions_str}
</answer>"""

def generate_missing_tags(gt_actions):
    """缺失标签"""
    return """Thinking...
MAKESOLUTION with CCO
ADD CC(=O)O
STIR for 2 h
YIELD CC(=O)OCC"""


# ================== 测试执行 ==================

def run_test_case(gt_case, output_generator, test_name):
    """运行单个测试用例"""
    gt_actions = gt_case["actions"]
    output = output_generator(gt_actions)
    
    ground_truth = {"actions": gt_actions}
    result = compute_score_with_details("test", output, ground_truth)
    
    return {
        "case_name": gt_case["name"],
        "test_name": test_name,
        "score": result["score"],
        "r_format": result["r_format"],
        "r_lcs": result["r_lcs"],
        "r_step": result["r_step"],
        "total_gt_signals": result.get("total_gt_signals", 0),
        "lcs_length": result.get("lcs_length", 0)
    }


def print_result(result):
    """打印测试结果"""
    print(f"  {result['case_name'][:20]:20s} | {result['test_name'][:25]:25s} | "
          f"Score: {result['score']:.3f} | R_f: {result['r_format']:.2f} | "
          f"R_lcs: {result['r_lcs']:.2f} | R_step: {result['r_step']:.2f}")


def run_all_tests():
    """运行所有测试"""
    print("=" * 120)
    print("全面严苛测试套件")
    print("=" * 120)
    print()
    
    all_results = []
    
    # 1. 完美匹配测试
    print("【1. 完美匹配测试】")
    print("-" * 120)
    for gt_case in REAL_GT_CASES:
        result = run_test_case(gt_case, generate_perfect_match, "完美匹配")
        all_results.append(result)
        print_result(result)
    print()
    
    # 2. 格式错误测试
    print("【2. 格式错误测试】")
    print("-" * 120)
    format_tests = [
        (generate_missing_think, "缺少think"),
        (generate_missing_answer, "缺少answer"),
        (generate_naked_smiles, "裸露SMILES"),
        (generate_naked_conditions, "裸露条件"),
        (generate_missing_tags, "无XML标签"),
        (generate_case_mismatch, "大小写错误"),
        (generate_empty_think, "空think"),
    ]
    for gt_case in REAL_GT_CASES[:2]:
        for generator, test_name in format_tests:
            result = run_test_case(gt_case, generator, test_name)
            all_results.append(result)
            print_result(result)
    print()
    
    # 3. 序列错误测试
    print("【3. 序列错误测试】")
    print("-" * 120)
    sequence_tests = [
        (generate_wrong_order, "顺序错误"),
        (generate_missing_steps, "缺少步骤"),
        (generate_extra_steps, "额外步骤"),
        (generate_repeated_actions, "重复动作"),
    ]
    for gt_case in REAL_GT_CASES[:2]:
        for generator, test_name in sequence_tests:
            if test_name == "缺少步骤":
                result = run_test_case(gt_case, lambda x: generate_missing_steps(x, 2), test_name)
            elif test_name == "额外步骤":
                result = run_test_case(gt_case, lambda x: generate_extra_steps(x, 2), test_name)
            else:
                result = run_test_case(gt_case, generator, test_name)
            all_results.append(result)
            print_result(result)
    print()
    
    # 4. 内容错误测试
    print("【4. 内容错误测试】")
    print("-" * 120)
    content_tests = [
        (generate_wrong_molecules, "分子错误"),
        (generate_wrong_quantities, "用量错误"),
        (generate_wrong_conditions, "条件错误"),
        (generate_wrong_yield_mol, "YIELD分子错误"),
    ]
    for gt_case in REAL_GT_CASES[:2]:
        for generator, test_name in content_tests:
            result = run_test_case(gt_case, generator, test_name)
            all_results.append(result)
            print_result(result)
    print()
    
    # 5. 极端错误测试
    print("【5. 极端错误测试】")
    print("-" * 120)
    extreme_tests = [
        (generate_gibberish, "完全胡言乱语"),
        (generate_partial_gibberish, "部分胡言乱语"),
        (generate_english_names, "英文名替代"),
        (generate_illegal_actions, "非法动作"),
        (generate_non_atomic, "非原子操作"),
        (generate_missing_yield, "缺少YIELD"),
        (generate_truncated_output, "截断输出"),
        (generate_xml_injection, "XML注入"),
        (generate_unicode_smiles, "Unicode字符"),
        (generate_very_long_output, "超长啰嗦"),
    ]
    for gt_case in REAL_GT_CASES[:1]:
        for generator, test_name in extreme_tests:
            result = run_test_case(gt_case, generator, test_name)
            all_results.append(result)
            print_result(result)
    print()
    
    # 汇总统计
    print("=" * 120)
    print("测试汇总")
    print("=" * 120)
    
    scores = [r["score"] for r in all_results]
    print(f"总测试数: {len(scores)}")
    print(f"平均分: {sum(scores)/len(scores):.3f}")
    print(f"最高分: {max(scores):.3f}")
    print(f"最低分: {min(scores):.3f}")
    print(f"满分(>0.95): {sum(1 for s in scores if s > 0.95)} 个")
    print(f"高分(0.8-0.95): {sum(1 for s in scores if 0.8 <= s <= 0.95)} 个")
    print(f"中分(0.5-0.8): {sum(1 for s in scores if 0.5 <= s < 0.8)} 个")
    print(f"低分(0.2-0.5): {sum(1 for s in scores if 0.2 <= s < 0.5)} 个")
    print(f"极低分(<0.2): {sum(1 for s in scores if s < 0.2)} 个")
    
    return all_results


if __name__ == "__main__":
    run_all_tests()
