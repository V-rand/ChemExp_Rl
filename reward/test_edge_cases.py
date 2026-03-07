"""
边界情况和极端场景测试
"""

import sys
sys.path.insert(0, '/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl')

from reward.reward_function import (
    compute_score_with_details,
    parse_procedure,
    parse_time_range,
    parse_temp_range,
    compute_format_reward
)


def test_case(name, solution, ground_truth, expected_score_range=None):
    """运行单个测试用例"""
    result = compute_score_with_details("test", solution, ground_truth)
    score = result["score"]
    
    status = "✓"
    if expected_score_range:
        min_score, max_score = expected_score_range
        if not (min_score <= score <= max_score):
            status = "✗"
    
    print(f"  {status} {name[:50]:50s} | Score: {score:.3f}")
    return result


def test_special_conditions():
    """测试特殊条件格式"""
    print("=" * 80)
    print("特殊条件格式测试")
    print("=" * 80)
    
    gt = {
        "actions": [
            "STIR for <time>2-3 h</time> at <temp>3-5 C</temp>",
            "REFLUX for <time>1.5 hours</time>",
            "WAIT for <time>below 5 min</time>",
            "HEAT to <temp>above 80 C</temp>",
            "STIR for <time>2 day</time> at <temp>below 3 C</temp>"
        ]
    }
    
    # 完美匹配
    solution = """<think>
Analysis...
</think>
<answer>
<procedure>STIR for <time>2-3 h</time> at <temp>3-5 C</temp></procedure>
<procedure>REFLUX for <time>1.5 hours</time></procedure>
<procedure>WAIT for <time>below 5 min</time></procedure>
<procedure>HEAT to <temp>above 80 C</temp></procedure>
<procedure>STIR for <time>2 day</time> at <temp>below 3 C</temp></procedure>
</answer>"""
    
    test_case("特殊条件格式-完美匹配", solution, gt, (0.95, 1.0))
    
    # 时间范围不匹配
    solution2 = """<think>
Analysis...
</think>
<answer>
<procedure>STIR for <time>4-5 h</time> at <temp>3-5 C</temp></procedure>
<procedure>REFLUX for <time>1.5 hours</time></procedure>
<procedure>WAIT for <time>below 5 min</time></procedure>
<procedure>HEAT to <temp>above 80 C</temp></procedure>
<procedure>STIR for <time>2 day</time> at <temp>below 3 C</temp></procedure>
</answer>"""
    test_case("时间范围不匹配", solution2, gt, (0.7, 0.95))
    print()


def test_molecule_variations():
    """测试分子匹配的各种情况"""
    print("=" * 80)
    print("分子匹配变体测试")
    print("=" * 80)
    
    gt = {
        "actions": [
            "ADD [MOL] ```c1ccccc1``` [/MOL] (1.0 g)",
            "ADD [MOL] ```CCO``` [/MOL] and [MOL] ```CC(=O)O``` [/MOL]",
            "YIELD [MOL] ```CC(=O)OCC``` [/MOL]"
        ]
    }
    
    # 完全匹配
    solution = """<think>Analysis...</think>
<answer>
<procedure>ADD [MOL] ```c1ccccc1``` [/MOL] (1.0 g)</procedure>
<procedure>ADD [MOL] ```CCO``` [/MOL] and [MOL] ```CC(=O)O``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CC(=O)OCC``` [/MOL]</procedure>
</answer>"""
    test_case("完全匹配", solution, gt, (0.95, 1.0))
    
    # 少一个分子
    solution2 = """<think>Analysis...</think>
<answer>
<procedure>ADD [MOL] ```c1ccccc1``` [/MOL] (1.0 g)</procedure>
<procedure>ADD [MOL] ```CCO``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CC(=O)OCC``` [/MOL]</procedure>
</answer>"""
    test_case("少一个分子", solution2, gt, (0.7, 0.95))
    
    # 分子顺序不同
    solution3 = """<think>Analysis...</think>
<answer>
<procedure>ADD [MOL] ```c1ccccc1``` [/MOL] (1.0 g)</procedure>
<procedure>ADD [MOL] ```CC(=O)O``` [/MOL] and [MOL] ```CCO``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CC(=O)OCC``` [/MOL]</procedure>
</answer>"""
    test_case("分子顺序不同", solution3, gt, (0.95, 1.0))
    
    # 完全不同的分子
    solution4 = """<think>Analysis...</think>
<answer>
<procedure>ADD [MOL] ```CCCC``` [/MOL] (1.0 g)</procedure>
<procedure>ADD [MOL] ```CCN``` [/MOL] and [MOL] ```CCCN``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CCCCC``` [/MOL]</procedure>
</answer>"""
    test_case("完全不同分子", solution4, gt, (0.0, 0.5))
    print()


def test_quantity_edge_cases():
    """测试用量边界情况"""
    print("=" * 80)
    print("用量边界情况测试")
    print("=" * 80)
    
    gt = {
        "actions": [
            "ADD [MOL] ```CCO``` [/MOL] (1.0 g, 10 mmol)",
            "ADD [MOL] ```C1CCOC1``` [/MOL] (50 mL)",
            "YIELD [MOL] ```CC``` [/MOL] (0.8 g)"
        ]
    }
    
    # 精确匹配
    solution = """<think>Analysis...</think>
<answer>
<procedure>ADD [MOL] ```CCO``` [/MOL] (1.0 g, 10 mmol)</procedure>
<procedure>ADD [MOL] ```C1CCOC1``` [/MOL] (50 mL)</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL] (0.8 g)</procedure>
</answer>"""
    test_case("用量精确匹配", solution, gt, (0.95, 1.0))
    
    # 小误差（<10%）
    solution2 = """<think>Analysis...</think>
<answer>
<procedure>ADD [MOL] ```CCO``` [/MOL] (1.08 g, 10.5 mmol)</procedure>
<procedure>ADD [MOL] ```C1CCOC1``` [/MOL] (53 mL)</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL] (0.85 g)</procedure>
</answer>"""
    test_case("用量小误差(<10%)", solution2, gt, (0.9, 1.0))
    
    # 单位换算
    solution3 = """<think>Analysis...</think>
<answer>
<procedure>ADD [MOL] ```CCO``` [/MOL] (1000 mg, 10000 umol)</procedure>
<procedure>ADD [MOL] ```C1CCOC1``` [/MOL] (0.05 L)</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL] (800 mg)</procedure>
</answer>"""
    test_case("用量单位换算", solution3, gt, (0.9, 1.0))
    
    # 巨大误差
    solution4 = """<think>Analysis...</think>
<answer>
<procedure>ADD [MOL] ```CCO``` [/MOL] (100 g, 1000 mmol)</procedure>
<procedure>ADD [MOL] ```C1CCOC1``` [/MOL] (5000 mL)</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL] (0.01 g)</procedure>
</answer>"""
    test_case("用量巨大误差", solution4, gt, (0.0, 0.5))
    print()


def test_action_only_steps():
    """测试只有动作的步骤"""
    print("=" * 80)
    print("纯动作步骤测试")
    print("=" * 80)
    
    gt = {
        "actions": [
            "CONCENTRATE",
            "FILTER",
            "WASH",
            "DRYSOLUTION",
            "YIELD [MOL] ```CC``` [/MOL]"
        ]
    }
    
    # 完美匹配 - 纯动作步骤应该满分
    solution = """<think>Analysis...</think>
<answer>
<procedure>CONCENTRATE</procedure>
<procedure>FILTER</procedure>
<procedure>WASH</procedure>
<procedure>DRYSOLUTION</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL]</procedure>
</answer>"""
    
    result = test_case("纯动作步骤-完美匹配", solution, gt, (0.95, 1.0))
    
    # 验证R_step是否为1.0
    if result["r_step"] == 1.0:
        print("    ✓ R_step = 1.0 (纯动作步骤动作对了就满分)")
    else:
        print(f"    ✗ R_step = {result['r_step']} (应该为1.0)")
    
    # 纯动作步骤但顺序错误
    solution2 = """<think>Analysis...</think>
<answer>
<procedure>FILTER</procedure>
<procedure>CONCENTRATE</procedure>
<procedure>WASH</procedure>
<procedure>DRYSOLUTION</procedure>
<procedure>YIELD [MOL] ```CC``` [/MOL]</procedure>
</answer>"""
    test_case("纯动作步骤-顺序错误", solution2, gt, (0.7, 0.95))
    print()


def test_complex_reaction():
    """测试复杂反应场景"""
    print("=" * 80)
    print("复杂反应场景测试")
    print("=" * 80)
    
    # 复杂多步反应
    gt = {
        "actions": [
            "MAKESOLUTION (Solution A) with [MOL] ```Br[Mg]C1CCCCC1``` [/MOL] (10 mmol) in [MOL] ```C1CCOC1``` [/MOL] (20 mL)",
            "SETTEMP to <temp>-78 C</temp>",
            "MAKESOLUTION (Solution B) with [MOL] ```CC(C)=O``` [/MOL] (8 mmol) in [MOL] ```C1CCOC1``` [/MOL] (10 mL)",
            "ADD Solution B to Solution A dropwise",
            "STIR for <time>2 h</time> while warming to <temp>0 C</temp>",
            "QUENCH with [MOL] ```O.[Cl-].[NH4+]``` [/MOL] (sat. aq.)",
            "EXTRACT with [MOL] ```CCOCC``` [/MOL] (3 x 50 mL)",
            "COLLECTLAYER organic",
            "WASH with [MOL] ```O.[Cl-].[Na+]``` [/MOL] (brine)",
            "DRYSOLUTION over [MOL] ```O=S(=O)([O-])[O-].[Mg+2]``` [/MOL]",
            "FILTER keep filtrate",
            "CONCENTRATE",
            "PURIFY by column chromatography",
            "YIELD [MOL] ```CC(C)(O)C1CCCCC1``` [/MOL] (1.2 g, 75%)"
        ]
    }
    
    # 完美匹配
    solution = """<think>
This is a Grignard reaction between cyclohexylmagnesium bromide and acetone.
Therefore, the validated operational sequence is: MAKESOLUTION, SETTEMP, MAKESOLUTION, ADD, STIR, QUENCH, EXTRACT, COLLECTLAYER, WASH, DRYSOLUTION, FILTER, CONCENTRATE, PURIFY, YIELD
</think>
<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```Br[Mg]C1CCCCC1``` [/MOL] (10 mmol) in [MOL] ```C1CCOC1``` [/MOL] (20 mL)</procedure>
<procedure>SETTEMP to <temp>-78 C</temp></procedure>
<procedure>MAKESOLUTION (Solution B) with [MOL] ```CC(C)=O``` [/MOL] (8 mmol) in [MOL] ```C1CCOC1``` [/MOL] (10 mL)</procedure>
<procedure>ADD Solution B to Solution A dropwise</procedure>
<procedure>STIR for <time>2 h</time> while warming to <temp>0 C</temp></procedure>
<procedure>QUENCH with [MOL] ```O.[Cl-].[NH4+]``` [/MOL] (sat. aq.)</procedure>
<procedure>EXTRACT with [MOL] ```CCOCC``` [/MOL] (3 x 50 mL)</procedure>
<procedure>COLLECTLAYER organic</procedure>
<procedure>WASH with [MOL] ```O.[Cl-].[Na+]``` [/MOL] (brine)</procedure>
<procedure>DRYSOLUTION over [MOL] ```O=S(=O)([O-])[O-].[Mg+2]``` [/MOL]</procedure>
<procedure>FILTER keep filtrate</procedure>
<procedure>CONCENTRATE</procedure>
<procedure>PURIFY by column chromatography</procedure>
<procedure>YIELD [MOL] ```CC(C)(O)C1CCCCC1``` [/MOL] (1.2 g, 75%)</procedure>
</answer>"""
    
    test_case("复杂反应-完美匹配", solution, gt, (0.95, 1.0))
    
    # 跳过纯化步骤
    solution2 = """<think>Analysis...</think>
<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```Br[Mg]C1CCCCC1``` [/MOL] (10 mmol) in [MOL] ```C1CCOC1``` [/MOL] (20 mL)</procedure>
<procedure>SETTEMP to <temp>-78 C</temp></procedure>
<procedure>MAKESOLUTION (Solution B) with [MOL] ```CC(C)=O``` [/MOL] (8 mmol) in [MOL] ```C1CCOC1``` [/MOL] (10 mL)</procedure>
<procedure>ADD Solution B to Solution A dropwise</procedure>
<procedure>STIR for <time>2 h</time> while warming to <temp>0 C</temp></procedure>
<procedure>QUENCH with [MOL] ```O.[Cl-].[NH4+]``` [/MOL] (sat. aq.)</procedure>
<procedure>EXTRACT with [MOL] ```CCOCC``` [/MOL] (3 x 50 mL)</procedure>
<procedure>COLLECTLAYER organic</procedure>
<procedure>WASH with [MOL] ```O.[Cl-].[Na+]``` [/MOL] (brine)</procedure>
<procedure>DRYSOLUTION over [MOL] ```O=S(=O)([O-])[O-].[Mg+2]``` [/MOL]</procedure>
<procedure>FILTER keep filtrate</procedure>
<procedure>CONCENTRATE</procedure>
<procedure>YIELD [MOL] ```CC(C)(O)C1CCCCC1``` [/MOL] (1.2 g, 75%)</procedure>
</answer>"""
    test_case("复杂反应-跳过纯化", solution2, gt, (0.85, 0.95))
    print()


def test_model_specific_errors():
    """测试模型特有的错误模式"""
    print("=" * 80)
    print("模型特有错误模式测试")
    print("=" * 80)
    
    gt = {
        "actions": [
            "MAKESOLUTION with [MOL] ```c1ccccc1``` [/MOL] (5.0 g) in THF",
            "ADD n-butyllithium (1.6 M, 20 mL)",
            "STIR for <time>1 h</time> at <temp>-78 C</temp>",
            "YIELD [MOL] ```c1ccccc1[Li]``` [/MOL]"
        ]
    }
    
    # 模型使用常见名称而非SMILES
    solution = """<think>Analysis...</think>
<answer>
<procedure>MAKESOLUTION with benzene (5.0 g) in THF</procedure>
<procedure>ADD n-butyllithium (1.6 M, 20 mL)</procedure>
<procedure>STIR for <time>1 h</time> at <temp>-78 C</temp></procedure>
<procedure>YIELD phenyllithium</procedure>
</answer>"""
    test_case("使用常见名称", solution, gt, (0.0, 0.5))
    
    # 模型重复输出相同步骤
    solution2 = """<think>Analysis...</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```c1ccccc1``` [/MOL] (5.0 g)</procedure>
<procedure>MAKESOLUTION with [MOL] ```c1ccccc1``` [/MOL] (5.0 g)</procedure>
<procedure>ADD [MOL] ```[Li]CCCC``` [/MOL] (1.6 M, 20 mL)</procedure>
<procedure>ADD [MOL] ```[Li]CCCC``` [/MOL] (1.6 M, 20 mL)</procedure>
<procedure>STIR for <time>1 h</time> at <temp>-78 C</temp></procedure>
<procedure>STIR for <time>1 h</time> at <temp>-78 C</temp></procedure>
<procedure>YIELD [MOL] ```c1ccccc1[Li]``` [/MOL]</procedure>
</answer>"""
    test_case("重复输出", solution2, gt, (0.4, 0.7))
    
    # 模型漏掉关键步骤
    solution3 = """<think>Analysis...</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```c1ccccc1``` [/MOL] (5.0 g)</procedure>
<procedure>STIR for <time>1 h</time></procedure>
<procedure>YIELD [MOL] ```c1ccccc1[Li]``` [/MOL]</procedure>
</answer>"""
    test_case("漏掉关键步骤", solution3, gt, (0.4, 0.7))
    
    # 模型条件错误但其他正确
    solution4 = """<think>Analysis...</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```c1ccccc1``` [/MOL] (5.0 g)</procedure>
<procedure>ADD [MOL] ```[Li]CCCC``` [/MOL] (1.6 M, 20 mL)</procedure>
<procedure>STIR for <time>1 h</time> at <temp>25 C</temp></procedure>
<procedure>YIELD [MOL] ```c1ccccc1[Li]``` [/MOL]</procedure>
</answer>"""
    test_case("条件错误", solution4, gt, (0.7, 0.9))
    print()


def run_all_edge_tests():
    """运行所有边界测试"""
    test_special_conditions()
    test_molecule_variations()
    test_quantity_edge_cases()
    test_action_only_steps()
    test_complex_reaction()
    test_model_specific_errors()
    
    print("=" * 80)
    print("边界测试完成")
    print("=" * 80)


if __name__ == "__main__":
    run_all_edge_tests()
