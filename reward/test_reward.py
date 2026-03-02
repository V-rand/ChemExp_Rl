import unittest
import numpy as np
import re
try:
    from reward_function import compute_score
except ImportError:
    from .reward_function import compute_score

# ============================================================================
# 测试数据准备
# ============================================================================

# 1. 保真度测试数据 (Fidelity)
GT_FIDELITY = {
    "actions": np.array(["EXTRACT with ethyl acetate twice"], dtype=object), 
    "molecules": "{}"
}

# 2. RDKit 标准化测试数据 (Canonical)
GT_CANONICAL = {
    "actions": np.array(["ADD [MOL] CCOC(C)=O [/MOL]"], dtype=object), # Ethyl Acetate
    "molecules": "{}"
}

# 3. 复杂场景 raw XML (Complex 17 steps)
RAW_XML_COMPLEX = """<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```COC(=O)c1cccc(-c2cnc(C(=O)CCc3ccc(-c4ccc(CN5CCCCC5)cc4)cc3)o2)n1``` [/MOL] (11 mg, 0.022 mmol) in [MOL] ```C1CCOC1``` [/MOL] (1.2 ml) and [MOL] ```O``` [/MOL] (0.8 ml)</procedure>
<procedure>ADD [MOL] ```[Li+].[OH-]``` [/MOL] (1.5 mg, 0.065 mmol) to Solution A</procedure>
<procedure>STIR for <time>20 min</time> at <temp>room temperature</temp> under Ar</procedure>
<procedure>ADD [MOL] ```Cl``` [/MOL] (1 N aqueous solution) to adjust pH to 4</procedure>
<procedure>ADD [MOL] ```ClCCl``` [/MOL] (2 ml)</procedure>
<procedure>PHASESEPA</procedure>
<procedure>COLLECTLAYER organic</procedure>
<procedure>ADD saturated aqueous NaHCO3 to aqueous layer to adjust pH to 8</procedure>
<procedure>ADD [MOL] ```ClCCl``` [/MOL] (2 ml) to aqueous layer</procedure>
<procedure>EXTRACT</procedure>
<procedure>COLLECTLAYER organic</procedure>
<procedure>COMBINE organic layers</procedure>
<procedure>DRYSOLUTION over Na2SO4</procedure>
<procedure>FILTER keep filtrate</procedure>
<procedure>CONCENTRATE</procedure>
<procedure>TRITURATE in ether</procedure>
<procedure>YIELD [MOL] ```O=C(O)c1cccc(-c2cnc(C(=O)CCc3ccc(-c4ccc(CN5CCCCC5)cc4)cc3)o2)n1``` [/MOL] (4 mg, 38%)</procedure>
</answer>"""

GT_COMPLEX_ACTIONS = re.findall(r'<procedure>(.*?)</procedure>', RAW_XML_COMPLEX)
GT_COMPLEX = {"actions": np.array(GT_COMPLEX_ACTIONS, dtype=object), "molecules": "{}"}

class TestRewardFinal(unittest.TestCase):
    
    def wrap(self, content, think="Valid reasoning block."):
        # 如果 content 本身包含了 tags，就不再重复包裹
        if "<answer>" in content:
            if "<think>" not in content:
                return f"<think>{think}</think>\n{content}"
            return content
        return f"<think>{think}</think>\n<answer>\n{content}\n</answer>"

    def wrap_proc(self, proc_text):
        return self.wrap(f"<procedure>{proc_text}</procedure>")

    # ------------------------------------------------------------------------
    # 1. 保真度测试 (String Fidelity)
    # ------------------------------------------------------------------------
    def test_fidelity_twice(self):
        """测试 'twice' 是否能提高分数"""
        # 完美
        score_full = compute_score('chemexp', self.wrap_proc("EXTRACT with ethyl acetate twice"), GT_FIDELITY)
        # 缺失 twice
        score_miss = compute_score('chemexp', self.wrap_proc("EXTRACT with ethyl acetate"), GT_FIDELITY)
        
        print(f"\n[Fidelity] 'Twice': {score_full} vs {score_miss}")
        self.assertEqual(score_full, 1.0)
        self.assertLess(score_miss, 1.0)
        self.assertGreater(score_miss, 0.9) # 骨架对，细节扣

    # ------------------------------------------------------------------------
    # 2. RDKit 标准化测试 (Canonical)
    # ------------------------------------------------------------------------
    def test_canonical_smiles(self):
        """测试不同 SMILES 写法是否匹配"""
        # GT: CCOC(C)=O
        # Model: CC(=O)OCC
        score = compute_score('chemexp', self.wrap_proc("ADD [MOL] CC(=O)OCC [/MOL]"), GT_CANONICAL)
        print(f"[Canonical] SMILES Equivalence: {score}")
        self.assertEqual(score, 1.0)

    # ------------------------------------------------------------------------
    # 3. 致命错误检查 (Fatal Veto)
    # ------------------------------------------------------------------------
    def test_fatal_missing_think(self):
        """测试缺失 <think>"""
        score = compute_score('chemexp', RAW_XML_COMPLEX, GT_COMPLEX) # RAW没有think
        print(f"[Fatal] Missing Think: {score}")
        self.assertEqual(score, 0.0)

    def test_fatal_naked_entities(self):
        """测试裸露实体 > 50%"""
        # 移除 STIR 的标签 (STIR for 20 min at room temperature)
        # 复杂例子有17步，移除1步不会触发 Fatal，我们需要移除很多
        # 构造简单裸奔例子
        gt_simple = {"actions": np.array(["STIR 10 min 100 C"], dtype=object)}
        naked_out = self.wrap("<procedure>STIR 10 min 100 C</procedure>")
        score = compute_score('chemexp', naked_out, gt_simple)
        print(f"[Fatal] Naked Entities: {score}")
        self.assertEqual(score, 0.0)

    # ------------------------------------------------------------------------
    # 4. 复杂场景综合 (Complex)
    # ------------------------------------------------------------------------
    def test_complex_perfect(self):
        """测试 17 步复杂反应完美匹配"""
        valid_out = f"<think>Reasoning...</think>\n{RAW_XML_COMPLEX}"
        score = compute_score('chemexp', valid_out, GT_COMPLEX)
        print(f"[Complex] Perfect Match: {score}")
        self.assertEqual(score, 1.0)

    def test_complex_skeleton_error(self):
        """测试骨架错误 (温度/时间)"""
        # 修改 STIR 步骤: 20 min -> 20 h
        broken_xml = RAW_XML_COMPLEX.replace("20 min", "20 h")
        valid_out = f"<think>Reasoning...</think>\n{broken_xml}"
        score = compute_score('chemexp', valid_out, GT_COMPLEX)
        print(f"[Complex] Time Mismatch: {score}")
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.85) # 仅错一步骨架

    def test_complex_fidelity_error(self):
        """测试细节缺失 (TRITURATE in ether -> TRITURATE)"""
        # 修改最后一步
        broken_xml = RAW_XML_COMPLEX.replace("TRITURATE in ether", "TRITURATE")
        valid_out = f"<think>Reasoning...</think>\n{broken_xml}"
        score = compute_score('chemexp', valid_out, GT_COMPLEX)
        print(f"[Complex] Detail Missing: {score}")
        self.assertLess(score, 1.0)
        self.assertGreater(score, 0.95) # 仅错一步细节，扣分很轻

if __name__ == '__main__':
    unittest.main()