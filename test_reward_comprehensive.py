"""
Comprehensive test for reward function with real data
Tests each scoring module independently
"""
import json
import re
import sys
from pathlib import Path

# Add verl to path
sys.path.insert(0, str(Path(__file__).parent / "verl"))

from reward.reward_function import (
    parse_qty, match_quantity, normalize_quantity, UNIT_CONVERSIONS,
    canonicalize_smiles, COMMON_CHEMICALS, is_valid_smiles,
    match_smiles_set, StepReward, compute_step_level_reward,
    check_format, compute_penalty, compute_score, ALLOWED_ACTIONS,
    parse_procedure
)

# ANSI colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_test(name, passed, details=""):
    """Print test result with color"""
    status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
    print(f"{status} | {name}")
    if details:
        print(f"      {details}")

# ==================== Test 1: DIPEA/TEA Separation ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST 1: DIPEA/TEA SMILES Separation{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

# Test canonicalization with valid SMILES
tea_smiles = canonicalize_smiles("ccn(cc)cc")
dipea_smiles = canonicalize_smiles("ccn(cc)c(c)(c)c")

print_test("DIPEA canonicalizes correctly",
          dipea_smiles == "ccn(cc)c(c)(c)c",
          f"Expected: ccn(cc)c(c)(c)c, Got: {dipea_smiles}")

print_test("TEA canonicalizes correctly",
          tea_smiles == "ccn(cc)cc",
          f"Expected: ccn(cc)cc, Got: {tea_smiles}")

print_test("DIPEA and TEA have different SMILES",
          dipea_smiles != tea_smiles,
          f"DIPEA: {dipea_smiles}, TEA: {tea_smiles}")

# Check COMMON_CHEMICALS mapping
tea_aliases = COMMON_CHEMICALS.get("ccn(cc)cc", [])
dipea_aliases = COMMON_CHEMICALS.get("ccn(cc)c(c)(c)c", [])

print_test("TEA only has TEA aliases (no DIPEA)",
          "dipea" not in tea_aliases and "diisopropylethylamine" not in tea_aliases,
          f"TEA aliases: {tea_aliases}")

print_test("DIPEA has correct aliases",
          "dipea" in dipea_aliases and "diisopropylethylamine" in dipea_aliases,
          f"DIPEA aliases: {dipea_aliases}")

# ==================== Test 2: Unit Parsing ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST 2: Quantity Parsing with Units{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

test_qty_inputs = [
    ("10 g", (10.0, "g")),
    ("10.5 ml", (10.5, "ml")),
    ("0.55 mmol", (0.55, "mmol")),
    ("261 uL", (261.0, "ul")),
    ("1.0 mol", (1.0, "mol")),
    ("5 mg", (5.0, "mg")),
]

for input_str, expected in test_qty_inputs:
    result = parse_qty(input_str)
    passed = result == expected
    print_test(f"parse_qty('{input_str}')", passed,
              f"Expected: {expected}, Got: {result}")

# ==================== Test 3: Unit Enforcement ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST 3: Unit Enforcement in Quantity Matching{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

print_test("Same unit and value: (10 ml) vs (10 ml) = 1.0",
          match_quantity(["(10 ml)"], ["(10 ml)"]) == 1.0)

print_test("Same unit, within 20% tolerance: (10.5 ml) vs (10 ml) = 1.0",
          match_quantity(["(10.5 ml)"], ["(10 ml)"]) == 1.0)

print_test("Different units (ml vs g) should NOT match",
          match_quantity(["(10 ml)"], ["(10 g)"]) == 0.0,
          "This is critical: 10 ml != 10 g")

print_test("Different units (g vs mg) should NOT match",
          match_quantity(["(10 g)"], ["(10 mg)"]) == 0.0)

print_test("Same unit, outside tolerance should NOT match",
          match_quantity(["(15 ml)"], ["(10 ml)"]) == 0.0,
          "15 ml vs 10 ml is > 20% tolerance")

# ==================== Test 4: SMILES Matching ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST 4: SMILES Set Matching{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

test_smiles_cases = [
    # Exact match (need to canonicalize first)
    (
        {canonicalize_smiles("CCO"), canonicalize_smiles("c1ccccc1")},
        {canonicalize_smiles("CCO"), canonicalize_smiles("c1ccccc1")},
        1.0,
        "Exact SMILES match"
    ),
    # Partial match (1/2) - Jaccard similarity
    (
        {canonicalize_smiles("CCO"), canonicalize_smiles("CN")},
        {canonicalize_smiles("CCO"), canonicalize_smiles("c1ccccc1")},
        0.5,
        "Partial SMILES match (1 of 2)"
    ),
    # Name match via COMMON_CHEMICALS
    (
        {"thf"},  # Name will be converted in match_smiles_set
        {canonicalize_smiles("c1ccoc1")},
        1.0,
        "Name 'thf' matches SMILES 'c1ccoc1'"
    ),
    # DIPEA name test
    (
        {"dipea"},  # Name will be converted in match_smiles_set
        {canonicalize_smiles("ccn(cc)c(c)(c)c")},
        1.0,
        "Name 'dipea' matches DIPEA SMILES"
    ),
    # TEA vs DIPEA should NOT match
    (
        {canonicalize_smiles("ccn(cc)cc")},  # TEA
        {canonicalize_smiles("ccn(cc)c(c)(c)c")},  # DIPEA
        0.0,
        "TEA and DIPEA should NOT match"
    ),
]

for pred_set, gt_set, expected_score, desc in test_smiles_cases:
    result = match_smiles_set(pred_set, gt_set)
    passed = result == expected_score
    print_test(f"{desc}", passed,
              f"Expected: {expected_score}, Got: {result}")

# ==================== Test 5: Procedure Parsing ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST 5: Procedure Parsing{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

test_procedures = [
    # MAKESOLUTION with SMILES and quantities
    '<procedure>MAKESOLUTION (Solution A) with [MOL] ```CCO``` [/MOL] (10 ml) and [MOL] ```c1ccccc1``` [/MOL] (5 g)</procedure>',
    # ADD with SMILES
    '<procedure>ADD [MOL] ```CN(C)C=O``` [/MOL] (25 mL)</procedure>',
    # STIR with time and temp
    '<procedure>STIR for <time>2 hours</time> at <temp>room temperature</temp></procedure>',
    # YIELD
    '<procedure>YIELD [MOL] ```CCCOc1c(CN(C)C(=O)/C=C/c2cnc3c(c2)OC(C)(C)C(=O)N3)cccc1OC``` [/MOL] (164 mg, 75%)</procedure>',
]

for proc_text in test_procedures:
    parsed = parse_procedure(proc_text)
    print_test(f"Parse procedure with action '{parsed['action']}'",
              parsed['action'] in ALLOWED_ACTIONS,
              f"SMILES count: {len(parsed['smiles'])}, Quantities: {parsed['quantities']}")

# ==================== Test 6: Real Data Test ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST 6: Real Data from example_data.jsonl{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

# Load example data
example_file = Path(__file__).parent / "data" / "raw" / "example_data.jsonl"
with open(example_file, 'r') as f:
    first_line = f.readline()
    gt_data = json.loads(first_line)

print(f"\nTesting with ground truth from example_data.jsonl (index: {gt_data['index']})")

# Extract ground truth actions and molecules
gt_action_text = gt_data.get("ACTION", "")
gt_procs = [a.strip() for a in gt_action_text.split('<procedure>')[1:] if a.strip()]

# Build molecules JSON for test
molecules_json = json.dumps(gt_data.get("molecules", {}))

print(f"Number of GT procedures: {len(gt_procs)}")

# Test 1: Perfect prediction (same as GT)
print(f"\n{YELLOW}--- Case 1: Perfect prediction (identical to GT) ---{RESET}")
pred_procs = gt_procs[:]  # Same as GT

step_rewards, total_chemistry = compute_step_level_reward(pred_procs, gt_procs, molecules_json)
print_test(f"Total chemistry score for perfect match",
          total_chemistry > 0.9,
          f"Got: {total_chemistry:.4f}")

# Step details
print(f"\nStep-by-step breakdown:")
for i, sr in enumerate(step_rewards):
    print(f"  Step {i+1}: action={sr.pred_action}, total={sr.step_total:.3f} "
          f"(act={sr.action_score:.1f}, mol={sr.mol_score:.3f}, qty={sr.qty_score:.1f})")

# Test 2: Wrong quantities (different units)
print(f"\n{YELLOW}--- Case 2: Wrong units (10 g vs 10 ml) ---{RESET}")

# Modify first procedure to have wrong unit
modified_procs = []
for proc in gt_procs:
    modified = proc.replace("(5 mL)", "(5 g)").replace("(10 mL)", "(10 g)")
    modified_procs.append(modified)

step_rewards_2, total_chemistry_2 = compute_step_level_reward(modified_procs, gt_procs, molecules_json)
print_test(f"Wrong unit should reduce chemistry score",
          total_chemistry_2 < total_chemistry,
          f"Perfect: {total_chemistry:.4f}, Wrong units: {total_chemistry_2:.4f}")

# Test 3: Wrong actions
print(f"\n{YELLOW}--- Case 3: Wrong actions (STIR instead of ADD) ---{RESET}")

modified_procs_3 = []
for proc in gt_procs:
    if "ADD" in proc:
        modified = proc.replace("<procedure>ADD", "<procedure>STIR")
    else:
        modified = proc
    modified_procs_3.append(modified)

step_rewards_3, total_chemistry_3 = compute_step_level_reward(modified_procs_3, gt_procs, molecules_json)
print_test(f"Wrong actions should significantly reduce score",
          total_chemistry_3 < 0.3,
          f"Got: {total_chemistry_3:.4f}")

# Test 4: Wrong molecule (DIPEA vs TEA)
print(f"\n{YELLOW}--- Case 4: DIPEA vs TEA (different chemicals) ---{RESET}")

# Create a test with DIPEA in prediction vs TEA in GT
test_pred = """
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CCO``` [/MOL] (10 ml) and [MOL] ```ccn(cc)c(c)(c)c``` [/MOL] (5 g)</procedure>
<procedure>ADD [MOL] ```c1ccccc1``` [/MOL]</procedure>
"""
test_gt = """
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CCO``` [/MOL] (10 ml) and [MOL] ```ccn(cc)cc``` [/MOL] (5 g)</procedure>
<procedure>ADD [MOL] ```c1ccccc1``` [/MOL]</procedure>
"""

test_molecules = {
    "ethanol": "CCO",
    "triethylamine": "ccn(cc)cc",
}

test_pred_procs = re.findall(r'<procedure>(.*?)</procedure>', test_pred, re.DOTALL)
test_gt_procs = re.findall(r'<procedure>(.*?)</procedure>', test_gt, re.DOTALL)

import re
step_rewards_4, total_chemistry_4 = compute_step_level_reward(
    test_pred_procs, test_gt_procs, json.dumps(test_molecules)
)
print_test(f"DIPEA vs TEA should NOT match",
          total_chemistry_4 < 0.8,
          f"Got: {total_chemistry_4:.4f} (should be < 0.8 due to mismatch)")

# ==================== Test 7: Format Check ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST 7: Format Check{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

good_format = """


<answer>
<procedure>MAKESOLUTION (Solution A) with [MOL] ```CCO``` [/MOL] (10 ml)</procedure>
<procedure>ADD [MOL] ```c1ccccc1``` [/MOL]</procedure>
<procedure>STIR for <time>2 hours</time> at <temp>room temperature</temp></procedure>
<procedure>YIELD [MOL] ```CCO``` [/MOL]</procedure>
</answer>
"""

bad_format_no_think = """
<answer>
<procedure>YIELD [MOL] ```CCO``` [/MOL]</procedure>
</answer>
"""

bad_format_no_answer = """
<think>This is thinking</think>
<procedure>ADD something</procedure>
"""

bad_format_naked_smiles = """
<think>Thinking</think>

<answer>
<procedure>ADD CCO c1ccccc1</procedure>
<procedure>YIELD product</procedure>
</answer>
"""

bad_format_last_not_yield = """
<think>Thinking</think>

<answer>
<procedure>ADD [MOL] ```CCO``` [/MOL]</procedure>
<procedure>STIR</procedure>
</answer>
"""

format_score_1 = check_format(good_format, '{}')
print_test("Good format gets high score",
          format_score_1 >= 0.4,
          f"Got: {format_score_1:.4f}")

format_score_2 = check_format(bad_format_no_think, '{}')
print_test("Missing think reduces score",
          format_score_2 < format_score_1,
          f"No think: {format_score_2:.4f}, Good: {format_score_1:.4f}")

format_score_3 = check_format(bad_format_naked_smiles, '{}')
print_test("Naked SMILES reduce score",
          format_score_3 < format_score_1,
          f"Naked SMILES: {format_score_3:.4f}, Good: {format_score_1:.4f}")

format_score_4 = check_format(bad_format_last_not_yield, '{}')
print_test("Last step not YIELD reduces score",
          format_score_4 < format_score_1,
          f"No YIELD: {format_score_4:.4f}, Good: {format_score_1:.4f}")

# ==================== Test 8: Full compute_score ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST 8: Full compute_score with weight distribution{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

# Build ground truth from example
ground_truth = {
    'thinking': gt_data.get('Thinking_text', ''),
    'actions': gt_procs,
    'molecules': molecules_json
}

# Test perfect format and perfect chemistry
perfect_solution = gt_data.get("ACTION", "")
perfect_score = compute_score("chemexp", perfect_solution, ground_truth)
print_test("Perfect solution gets high score (> 0.8)",
          perfect_score > 0.8,
          f"Got: {perfect_score:.4f}")

# Test good format, bad chemistry
bad_chemistry_solution = perfect_solution.replace("<procedure>ADD", "<procedure>STIR").replace("<procedure>MAKESOLUTION", "<procedure>STIR")
bad_chemistry_score = compute_score("chemexp", bad_chemistry_solution, ground_truth)
print_test("Good format, bad chemistry gets lower score",
          bad_chemistry_score < 0.5,
          f"Got: {bad_chemistry_score:.4f}")

# Test bad format, good chemistry (same actions but bad tags)
bad_format_solution = perfect_solution.replace("[MOL]", "").replace("[/MOL]", "")
bad_format_score = compute_score("chemexp", bad_format_solution, ground_truth)
print_test("Bad format reduces score",
          bad_format_score < perfect_score,
          f"Good: {perfect_score:.4f}, Bad format: {bad_format_score:.4f}")

# ==================== Test 9: Penalty ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST 9: Penalty Computation{RESET}")
print(f"{BLUE}{'='*60}{RESET}")

illegal_action = """
<think>Thinking</think>

<answer>
<procedure>INVALID_ACTION [MOL] ```CCO``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CCO``` [/MOL]</procedure>
</answer>
"""

non_atomic = """
<think>Thinking</think>

<answer>
<procedure>ADD and STIR [MOL] ```CCO``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CCO``` [/MOL]</procedure>
</answer>
"""

short_think = """
<think>xx</think>

<answer>
<procedure>ADD [MOL] ```CCO``` [/MOL]</procedure>
<procedure>YIELD [MOL] ```CCO``` [/MOL]</procedure>
</answer>
"""

penalty_1 = compute_penalty(illegal_action)
penalty_2 = compute_penalty(non_atomic)
penalty_3 = compute_penalty(short_think)

print_test("Illegal action incurs penalty",
          penalty_1 > 0,
          f"Penalty: {penalty_1:.4f}")

print_test("Non-atomic procedure incurs penalty",
          penalty_2 > 0,
          f"Penalty: {penalty_2:.4f}")

print_test("Short think incurs penalty",
          penalty_3 > 0,
          f"Penalty: {penalty_3:.4f}")

# ==================== Summary ====================
print(f"\n{BLUE}{'='*60}{RESET}")
print(f"{BLUE}TEST SUMMARY{RESET}")
print(f"{BLUE}{'='*60}{RESET}")
print(f"All tests completed. Review any FAIL results above.")
print(f"\nKey verification points:")
print(f"  1. DIPEA and TEA are correctly separated (different SMILES)")
print(f"  2. Unit matching enforces same unit (10 ml != 10 g)")
print(f"  3. Step-level reward gives independent scores per step")
print(f"  4. compute_score uses 0.1*format + 0.9*chemistry - penalty")
print(f"  5. Chemistry dominates (90% weight) vs format (10%)")
