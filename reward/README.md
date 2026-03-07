# Chemical Experiment Reward Function

化学实验步骤奖励函数，用于VERL框架中的GRPO训练。

## 设计架构

奖励函数设计解耦为三个维度：

```
Reward = 0.2 * R_format + 0.2 * R_lcs + 0.6 * R_step
```

### R_format: 格式与守恒惩罚模块

- **初始分**: 1.0
- **检查项**:
  - XML标签平衡性（`<think>`, `<answer>`, `<procedure>`, `<time>`, `<temp>`）
  - 动作必须被`<procedure>`包裹（后向断言检查）
  - 每个`<procedure>`只能包含一个动作（原子性）
  - SMILES必须被`[MOL] ```...``` [/MOL]`包裹
  - 时间必须被`<time>`包裹
  - 温度必须被`<temp>`包裹
  - 最后一步必须是YIELD
  - 动作必须是24个允许动作之一

### R_lcs: 宏观动作序列对齐模块

计算预测序列与GT序列的最长公共子序列（LCS）：

```
R_lcs = |LCS| / max(N_pred, N_gt)
```

- 双向惩罚：漏步和啰嗦都会降低得分
- 强制模型学习正确的动作拓扑顺序

### R_step: 微观步骤得分模块

基于GT实际信号数量的归一化设计：

1. **信号识别**：对于每个GT步骤，统计其拥有的信号：
   - `has_mol`: 是否有SMILES（0或1）
   - `has_qty`: 是否有用量（0或1）
   - `has_condition`: 是否有时间或温度（0或1）

2. **步骤满分计算**：
   - 如果GT步骤没有任何信号（只有动作，如CONCENTRATE），则该步骤满分 = 1分
   - 如果有信号，则该步骤满分 = 信号数（1-3分）

3. **得分计算**（针对LCS对齐的步骤，动作必然匹配）：
   - **S_mol**: Jaccard相似度 >= 0.5 得1分，否则0分
   - **S_qty**: 用量相似度 >= 0.8 得1分，否则0分（YIELD步骤放宽到0.5）
   - **S_condition**: 时间和温度都匹配得1分，否则0分
   - 总分 = S_mol + S_qty + S_condition

4. **归一化**：
   ```
   R_step = 所有对齐步骤得分总和 / 所有GT步骤满分总和
   ```

## 接口

### compute_score

```python
def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Dict[str, Any],
    extra_info: Optional[Dict] = None
) -> float:
    """
    计算奖励分数
    
    Args:
        data_source: 数据集名称
        solution_str: 模型生成的回答（包含<think>和<answer>）
        ground_truth: 标准答案，包含:
            - 'actions': List[str] - GT procedure列表
            - 'prompt': str (optional) - 原始prompt
            - 'molecules': str (optional) - 分子映射表JSON
        extra_info: 额外信息字典
    
    Returns:
        float: 最终奖励分数 [0, 1]
    """
```

### compute_score_with_details

调试版本，返回详细的得分信息。

## 24个允许动作（严格限定）

**警告**: 模型只允许使用以下24个标准动作，任何其他动作（包括变体）都会被判定为非法并扣分。

```python
ALLOWED_ACTIONS = {
    'ADD', 'STIR', 'WAIT', 'CONCENTRATE', 'YIELD', 'MAKESOLUTION',
    'FILTER', 'WASH', 'DRYSOLUTION', 'COLLECTLAYER', 'EXTRACT',
    'SETTEMP', 'REFLUX', 'RECRYSTAL', 'PHASESEPA', 'PH', 'PURIFY',
    'QUENCH', 'PARTITION', 'TRITURATE', 'DRYSOLID', 'DEGAS',
    'MICROWAVE', 'SONICATE'
}
```

**注意**: 
- 使用 `SETTEMP` 设置温度（包括降温场景）
- 不要使用 `COOLTEMP`, `FREEZE`, `DISSOLVE` 等非标准动作
- 非法动作会触发格式扣分（-0.1分/个）

## 化学同义词映射

支持常见化学品的英文名、缩写映射到标准SMILES：

```python
CHEMICAL_SYNONYMS = {
    "c1ccoc1": ["thf", "tetrahydrofuran", "四氢呋喃"],
    "cn(c)c=o": ["dmf", "dimethylformamide"],
    # ...
}
```

## 温度和时间解析

支持多种格式的自然语言解析：

**时间格式**：
- 单个值: "15 h", "30 min", "2 day"
- 范围: "2-3 h", "3-5 hours"
- below/above: "below 3 min", "above 2 h"
- 自然语言: "overnight"

**温度格式**：
- 单个值: "25°C", "25 C", "100C"
- 范围: "20-25°C", "3-5 C"
- below/above: "below 40°C", "above 0 C"
- 自然语言: "room temperature", "rt", "reflux", "ice bath"

## 测试

```bash
# 运行基础测试
python reward/test_reward_function.py

# 运行真实数据测试
python reward/test_real_data.py
```

## 使用示例

```python
from reward.reward_function import compute_score

solution_str = """<think>
Analyzing the reaction...
Therefore, the validated operational sequence is: MAKESOLUTION, ADD, STIR, YIELD
</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```CC``` [/MOL] (1.0 g)</procedure>
<procedure>ADD [MOL] ```O``` [/MOL]</procedure>
<procedure>STIR for <time>2 h</time> at <temp>room temperature</temp></procedure>
<procedure>YIELD [MOL] ```CCO``` [/MOL]</procedure>
</answer>"""

ground_truth = {
    "actions": [
        "MAKESOLUTION with [MOL] ```CC``` [/MOL] (1.0 g)",
        "ADD [MOL] ```O``` [/MOL]",
        "STIR for <time>2 h</time> at <temp>room temperature</temp>",
        "YIELD [MOL] ```CCO``` [/MOL]"
    ],
    "prompt": "..."
}

score = compute_score("test", solution_str, ground_truth)
print(f"Score: {score}")
```
