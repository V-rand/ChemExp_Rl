"""
测试 reward_function.py
覆盖各种真实场景：正常情况、错误格式、缺失步骤等
"""

import sys
sys.path.insert(0, '/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl')

from reward.reward_function import (
    compute_score,
    compute_score_with_details,
    compute_format_reward,
    compute_lcs_reward,
    parse_procedure,
    parse_time_range,
    parse_temp_range,
    extract_smiles_from_text,
    ALLOWED_ACTIONS
)


# ================== 测试数据 ==================

# 标准GT数据
SAMPLE_GT = """<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol) in [MOL] ```C1CCOC1``` [/MOL] (25 mL)</procedure>
<procedure>ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)</procedure>
<procedure>STIR for <time>15 h</time> at <temp>room temperature</temp></procedure>
<procedure>CONCENTRATE</procedure>
<procedure>PURIFY by silica gel flash chromatography</procedure>
<procedure>ADD [MOL] ```CCOCC``` [/MOL]</procedure>
<procedure>CONCENTRATE</procedure>
<procedure>DRYSOLID under vacuum</procedure>
<procedure>YIELD [MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL] (0.095 g, 17%)</procedure>
</answer>"""

SAMPLE_PROMPT = '''Please design a chemical experiment based on the following requirements:
  "REACTANT": ["[MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol)", "[MOL] ```NC=S``` [/MOL] (25 mL)"],
  "PRODUCT": ["[MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL]"],
  "YIELD_TARGET": "17%",
  "CATALYST": [],
  "SOLVENT": ["[MOL] ```C1CCOC1``` [/MOL]"]'''


def test_format_check():
    """测试格式检查"""
    print("=" * 60)
    print("测试: 格式检查 (R_format)")
    print("=" * 60)
    
    # 1. 完全正确的格式
    correct_output = """<think>
This is a valid thinking process.
Therefore, the validated operational sequence is: MAKESOLUTION, ADD, STIR, CONCENTRATE, PURIFY, ADD, CONCENTRATE, DRYSOLID, YIELD
</think>
<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol) in [MOL] ```C1CCOC1``` [/MOL] (25 mL)</procedure>
<procedure>ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)</procedure>
<procedure>STIR for <time>15 h</time> at <temp>room temperature</temp></procedure>
<procedure>YIELD [MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL] (0.095 g, 17%)</procedure>
</answer>"""
    
    score = compute_format_reward(correct_output, SAMPLE_PROMPT)
    print(f"1. 完全正确的格式: R_format = {score:.4f}")
    assert score > 0.8, f"期望高分，实际 {score}"
    
    # 2. 缺少think标签
    no_think = """<answer>
<procedure>ADD [MOL] ```CC``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL]</procedure>
</answer>"""
    score = compute_format_reward(no_think)
    print(f"2. 缺少think标签: R_format = {score:.4f}")
    
    # 3. 缺少answer标签
    no_answer = """<think>Some thinking</think>
Some text"""
    score = compute_format_reward(no_answer)
    print(f"3. 缺少answer标签: R_format = {score:.4f}")
    
    # 4. 裸露的SMILES
    naked_smiles = """<think>Thinking</think>
<answer>
<procedure>ADD CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12</procedure>
<procedure>YIELD CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12</procedure>
</answer>"""
    score = compute_format_reward(naked_smiles)
    print(f"4. 裸露的SMILES: R_format = {score:.4f}")
    
    # 5. 裸露的时间和温度
    naked_conditions = """<think>Thinking</think>
<answer>
<procedure>STIR for 2 hours at room temperature</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL]</procedure>
</answer>"""
    score = compute_format_reward(naked_conditions)
    print(f"5. 裸露的时间和温度: R_format = {score:.4f}")
    
    # 6. 最后一步不是YIELD
    no_yield_last = """<think>Thinking</think>
<answer>
<procedure>ADD [MOL] ```CC``` [/MOL]</procedure>
<procedure>STIR for <time>2 h</time></procedure>
</answer>"""
    score = compute_format_reward(no_yield_last)
    print(f"6. 最后一步不是YIELD: R_format = {score:.4f}")
    
    # 7. 非法动作
    illegal_action = """<think>Thinking</think>
<answer>
<procedure>MIX [MOL] ```CC``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL]</procedure>
</answer>"""
    score = compute_format_reward(illegal_action)
    print(f"7. 非法动作MIX: R_format = {score:.4f}")
    
    # 8. 非原子性操作
    non_atomic = """<think>Thinking</think>
<answer>
<procedure>ADD and STIR [MOL] ```CC``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL]</procedure>
</answer>"""
    score = compute_format_reward(non_atomic)
    print(f"8. 非原子性操作: R_format = {score:.4f}")
    
    print()


def test_time_parsing():
    """测试时间解析"""
    print("=" * 60)
    print("测试: 时间解析")
    print("=" * 60)
    
    test_cases = [
        ("15 h", (15.0, 15.0)),
        ("2-3 hours", (2.0, 3.0)),
        ("30 min", (0.5, 0.5)),
        ("overnight", (8, 16)),
        ("room temperature", None),  # 不是时间
        ("2 hr", (2.0, 2.0)),
    ]
    
    for time_str, expected in test_cases:
        result = parse_time_range(time_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{time_str}' -> {result} (期望: {expected})")
    
    print()


def test_temp_parsing():
    """测试温度解析"""
    print("=" * 60)
    print("测试: 温度解析")
    print("=" * 60)
    
    test_cases = [
        ("room temperature", (20, 25)),
        ("rt", (20, 25)),
        ("25°C", (25.0, 25.0)),
        ("25 C", (25.0, 25.0)),
        ("20-25°C", (20.0, 25.0)),
        ("below 40°C", (-100, 40.0)),
        ("-10 C", (-10.0, -10.0)),
        ("reflux", (50, 150)),
        ("ice bath", (0, 5)),
    ]
    
    for temp_str, expected in test_cases:
        result = parse_temp_range(temp_str)
        status = "✓" if result == expected else "✗"
        print(f"  {status} '{temp_str}' -> {result} (期望: {expected})")
    
    print()


def test_lcs_alignment():
    """测试LCS对齐"""
    print("=" * 60)
    print("测试: LCS序列对齐")
    print("=" * 60)
    
    pred = [
        "MAKESOLUTION with A",
        "ADD B",
        "STIR for 2h",
        "CONCENTRATE",
        "YIELD C"
    ]
    
    # 完全匹配
    gt_match = [
        "MAKESOLUTION with A",
        "ADD B",
        "STIR for 2h",
        "CONCENTRATE",
        "YIELD C"
    ]
    score, align = compute_lcs_reward(pred, gt_match)
    print(f"1. 完全匹配: R_lcs = {score:.4f}, 对齐: {align}")
    assert score == 1.0, f"期望1.0，实际 {score}"
    
    # 缺少一个步骤
    gt_missing = [
        "MAKESOLUTION with A",
        "ADD B",
        "CONCENTRATE",
        "YIELD C"
    ]
    score, align = compute_lcs_reward(pred, gt_missing)
    print(f"2. 预测多一步: R_lcs = {score:.4f}, 对齐: {align}")
    
    # 预测缺少步骤
    pred_missing = [
        "MAKESOLUTION with A",
        "ADD B",
        "YIELD C"
    ]
    gt_full = [
        "MAKESOLUTION with A",
        "ADD B",
        "STIR for 2h",
        "CONCENTRATE",
        "YIELD C"
    ]
    score, align = compute_lcs_reward(pred_missing, gt_full)
    print(f"3. 预测少步骤: R_lcs = {score:.4f}, 对齐: {align}")
    
    # 顺序错误
    pred_wrong = [
        "MAKESOLUTION with A",
        "STIR for 2h",  # 提前了
        "ADD B",
        "CONCENTRATE",
        "YIELD C"
    ]
    score, align = compute_lcs_reward(pred_wrong, gt_full)
    print(f"4. 顺序错误: R_lcs = {score:.4f}, 对齐: {align}")
    
    # 无限长序列（钻空子测试）
    pred_long = ["MAKESOLUTION with A"] + ["WAIT"] * 100 + ["YIELD C"]
    gt_short = [
        "MAKESOLUTION with A",
        "ADD B",
        "YIELD C"
    ]
    score, align = compute_lcs_reward(pred_long, gt_short)
    print(f"5. 长序列惩罚: R_lcs = {score:.4f} (分母变大，得分降低)")
    assert score < 0.1, f"长序列应得低分，实际 {score}"
    
    print()


def test_procedure_parsing():
    """测试procedure解析"""
    print("=" * 60)
    print("测试: Procedure解析")
    print("=" * 60)
    
    # 标准MAKESOLUTION
    proc1 = "MAKESOLUTION (Solution A) with [MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol) in [MOL] ```C1CCOC1``` [/MOL] (25 mL)"
    result1 = parse_procedure(proc1)
    print(f"1. MAKESOLUTION:")
    print(f"   动作: {result1['action']}")
    print(f"   SMILES数量: {len(result1['smiles'])}")
    print(f"   用量: {result1['quantities']}")
    
    # STIR带时间和温度
    proc2 = "STIR for <time>15 h</time> at <temp>room temperature</temp>"
    result2 = parse_procedure(proc2)
    print(f"\n2. STIR:")
    print(f"   动作: {result2['action']}")
    print(f"   时间: {result2['time']}")
    print(f"   温度: {result2['temp']}")
    
    # YIELD
    proc3 = "YIELD [MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL] (0.095 g, 17%)"
    result3 = parse_procedure(proc3)
    print(f"\n3. YIELD:")
    print(f"   动作: {result3['action']}")
    print(f"   SMILES数量: {len(result3['smiles'])}")
    print(f"   用量: {result3['quantities']}")
    
    print()


def test_complete_scoring():
    """测试完整打分"""
    print("=" * 60)
    print("测试: 完整打分流程")
    print("=" * 60)
    
    ground_truth = {
        "actions": [
            "MAKESOLUTION (Solution A) with [MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol) in [MOL] ```C1CCOC1``` [/MOL] (25 mL)",
            "ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)",
            "STIR for <time>15 h</time> at <temp>room temperature</temp>",
            "CONCENTRATE",
            "YIELD [MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL] (0.095 g, 17%)"
        ],
        "prompt": SAMPLE_PROMPT
    }
    
    # 1. 完美匹配
    perfect = """<think>
Analyzing this reaction...
Therefore, the validated operational sequence is: MAKESOLUTION, ADD, STIR, CONCENTRATE, YIELD
</think>
<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol) in [MOL] ```C1CCOC1``` [/MOL] (25 mL)</procedure>
<procedure>ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)</procedure>
<procedure>STIR for <time>15 h</time> at <temp>room temperature</temp></procedure>
<procedure>CONCENTRATE</procedure>
<procedure>YIELD [MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL] (0.095 g, 17%)</procedure>
</answer>"""
    
    result = compute_score_with_details("test", perfect, ground_truth)
    print(f"1. 完美匹配:")
    print(f"   总分: {result['score']}")
    print(f"   R_format: {result['r_format']}, R_lcs: {result['r_lcs']}, R_step: {result['r_step']}")
    
    # 2. 格式正确但少了一步
    missing_step = """<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol) in [MOL] ```C1CCOC1``` [/MOL] (25 mL)</procedure>
<procedure>ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)</procedure>
<procedure>CONCENTRATE</procedure>
<procedure>YIELD [MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL] (0.095 g, 17%)</procedure>
</answer>"""
    
    result = compute_score_with_details("test", missing_step, ground_truth)
    print(f"\n2. 缺少STIR步骤:")
    print(f"   总分: {result['score']}")
    print(f"   R_format: {result['r_format']}, R_lcs: {result['r_lcs']}, R_step: {result['r_step']}")
    
    # 3. 动作正确但分子错误
    wrong_mol = """<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CCCC``` [/MOL] (0.60 g, 1.5 mmol) in [MOL] ```C1CCOC1``` [/MOL] (25 mL)</procedure>
<procedure>ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)</procedure>
<procedure>STIR for <time>15 h</time> at <temp>room temperature</temp></procedure>
<procedure>CONCENTRATE</procedure>
<procedure>YIELD [MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL] (0.095 g, 17%)</procedure>
</answer>"""
    
    result = compute_score_with_details("test", wrong_mol, ground_truth)
    print(f"\n3. 第一步分子错误:")
    print(f"   总分: {result['score']}")
    print(f"   R_format: {result['r_format']}, R_lcs: {result['r_lcs']}, R_step: {result['r_step']}")
    
    # 4. 完全不按格式
    garbage = "This is complete garbage output with no proper format"
    result = compute_score_with_details("test", garbage, ground_truth)
    print(f"\n4. 完全无格式:")
    print(f"   总分: {result['score']}")
    print(f"   R_format: {result['r_format']}")
    
    # 5. 重复输出（模型常见错误）
    repeated = """<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol) in [MOL] ```C1CCOC1``` [/MOL] (25 mL)</procedure>
<procedure>ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)</procedure>
<procedure>ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)</procedure>
<procedure>ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)</procedure>
<procedure>STIR for <time>15 h</time> at <temp>room temperature</temp></procedure>
<procedure>CONCENTRATE</procedure>
<procedure>YIELD [MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL] (0.095 g, 17%)</procedure>
</answer>"""
    
    result = compute_score_with_details("test", repeated, ground_truth)
    print(f"\n5. 重复输出ADD:")
    print(f"   总分: {result['score']}")
    print(f"   R_lcs: {result['r_lcs']}, R_step: {result['r_step']}")
    
    # 6. 用英文名而非SMILES
    name_instead = """<think>Thinking...</think>
<answer>
<procedure>MAKESOLUTION (Solution A) with THF (25 mL)</procedure>
<procedure>ADD reagent</procedure>
<procedure>STIR for <time>15 h</time> at <temp>room temperature</temp></procedure>
<procedure>YIELD product</procedure>
</answer>"""
    
    result = compute_score_with_details("test", name_instead, ground_truth)
    print(f"\n6. 用英文名而非SMILES:")
    print(f"   总分: {result['score']}")
    print(f"   R_format: {result['r_format']}, R_step: {result['r_step']}")
    
    print()


def test_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试: 边界情况")
    print("=" * 60)
    
    ground_truth = {
        "actions": [
            "MAKESOLUTION with [MOL] ```CC``` [/MOL]",
            "YIELD [MOL] ```CC``` [/MOL]"
        ]
    }
    
    # 1. 空输出
    result = compute_score("test", "", ground_truth)
    print(f"1. 空输出: score = {result}")
    
    # 2. 只有think没有answer
    only_think = "<think>Thinking...</think>"
    result = compute_score("test", only_think, ground_truth)
    print(f"2. 只有think: score = {result}")
    
    # 3. 空的answer
    empty_answer = "<think>Thinking...</think><answer></answer>"
    result = compute_score("test", empty_answer, ground_truth)
    print(f"3. 空的answer: score = {result}")
    
    # 4. 超长的胡言乱语
    long_gibberish = "<think>" + "blah " * 1000 + "</think><answer>" + "<procedure>ADD</procedure>" * 100 + "</answer>"
    result = compute_score("test", long_gibberish, ground_truth)
    print(f"4. 超长胡言乱语: score = {result}")
    
    # 5. 大小写混合
    mixed_case = """<THINK>Thinking...</THINK>
<ANSWER>
<PROCEDURE>add [MOL] ```CC``` [/MOL]</PROCEDURE>
<PROCEDURE>YIELD [MOL] ```CC``` [/MOL]</PROCEDURE>
</ANSWER>"""
    result = compute_score("test", mixed_case, ground_truth)
    print(f"5. 大小写混合: score = {result}")
    
    print()


def test_quantity_matching():
    """测试用量匹配"""
    print("=" * 60)
    print("测试: 用量匹配")
    print("=" * 60)
    
    from reward.reward_function import compute_quantity_similarity
    
    # 精确匹配
    qty1 = [{"value": 10.0, "unit": "g", "raw": "10 g"}]
    qty2 = [{"value": 10.0, "unit": "g", "raw": "10 g"}]
    score = compute_quantity_similarity(qty1, qty2)
    print(f"1. 精确匹配 (10g vs 10g): {score:.4f}")
    
    # 小误差 (10%)
    qty3 = [{"value": 11.0, "unit": "g", "raw": "11 g"}]
    score = compute_quantity_similarity(qty1, qty3)
    print(f"2. 10%误差 (10g vs 11g): {score:.4f}")
    
    # 大误差 (50%)
    qty4 = [{"value": 15.0, "unit": "g", "raw": "15 g"}]
    score = compute_quantity_similarity(qty1, qty4)
    print(f"3. 50%误差 (10g vs 15g): {score:.4f}")
    
    # 单位不同 (同一量纲)
    qty5 = [{"value": 10000.0, "unit": "mg", "raw": "10000 mg"}]
    score = compute_quantity_similarity(qty1, qty5)
    print(f"4. 单位换算 (10g vs 10000mg): {score:.4f}")
    
    print()


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("奖励函数测试套件")
    print("=" * 60 + "\n")
    
    test_format_check()
    test_time_parsing()
    test_temp_parsing()
    test_lcs_alignment()
    test_procedure_parsing()
    test_quantity_matching()
    test_complete_scoring()
    test_edge_cases()
    
    print("=" * 60)
    print("所有测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
