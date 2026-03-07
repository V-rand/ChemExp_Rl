#/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl项目目标
在虚拟环境中工作conda activate chemexp_verl
设计一个适合verl rl训练的reward model
###rl 训练任务为：对于Qwen-4B-instruct模型，为模型输入反应物（包含用量）生成物（包含参量）以及可能提供的必要溶剂和催化剂，模型输出think思考过程以及化学实验步骤
算法为GRPO。
输入示例："REACTANT": ["[MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol)", "[MOL] ```NC=S``` [/MOL] (25 mL)"],
  "PRODUCT": ["[MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL]"],
  "YIELD_TARGET": "17%",
  "CATALYST": [],
  "SOLVENT": ["[MOL] ```C1CCOC1``` [/MOL]"]
输出示例：<think>
[Reasoning process...]
Therefore, the validated operational sequence is: [summary]
</think>
<answer>
<procedure>MAKESOLUTION with [MOL] ```CC(C)c1nnc2ccc(C(Br)C(=O)c3ccc(F)cc3F)nn12``` [/MOL] (0.60 g, 1.5 mmol) in [MOL] ```C1CCOC1``` [/MOL] (25 mL)</procedure>
<procedure>ADD [MOL] ```NC=S``` [/MOL] (0.19 g, 3.0 mmol)</procedure>
<procedure>STIR for <time>15 h</time> at <temp>room temperature</temp></procedure>
<procedure>CONCENTRATE</procedure>
<procedure>PURIFY by silica gel flash chromatography</procedure>
<procedure>ADD [MOL] ```CCOCC``` [/MOL]</procedure>
<procedure>CONCENTRATE</procedure>
<procedure>DRYSOLID under vacuum</procedure>
<procedure>YIELD [MOL] ```CC(C)c1nnc2ccc(-c3scnc3-c3ccc(F)cc3F)nn12``` [/MOL] (0.095 g, 17%)</procedure>
</answer>
需要用<think></think>包裹思考过程
<answer></answer>包裹实验步骤回答
<procedure></procedure>包裹每一步动作
[MOL] ```smiles``` [/MOL]包裹化学品的smiles格式
<time>overnight</time>包裹时间
<temp>room temperature</temp>包裹温度
容许对于一些没有smiles格式的常见试剂模型使用缩写或者英文名，例如THF、K2CO3。你作为开发者但是要维护一个常见名称映射表防止匹配失误
模型仅容许使用24 actions: ADD, STIR, WAIT, CONCENTRATE, YIELD, MAKESOLUTION, FILTER, WASH, DRYSOLUTION, COLLECTLAYER, EXTRACT, SETTEMP, REFLUX, RECRYSTAL, PHASESEPA, PH, PURIFY, QUENCH, PARTITION, TRITURATE, DRYSOLID, DEGAS, MICROWAVE, SONICATE.
在添加试剂或者制备溶剂时，需要标明用量，类似ADD [MOL] ```SMILES``` [/MOL](0.60 g, 1.5 mmol)
完整的system prompt详见/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/configs/system_prompt.txt

###gt数据
对于数据你不要完整的将jsonl读完，因为非常大非常长，你最好只取10条或者你感兴趣的部分看。
原始数据通过hf下载，共2万数据，可以通过/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/download_dataset.py下载
示例数据为/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/raw/example_data.jsonl，共十条，你可以进去查看
为了适配verl的工作流，需要先转换格式/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/data_processing.py，并在这过程构建prompt和将smiles格式变为标准smiles
训练数据原始存到/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed
目前已经处理了一版1千条数据的高质量数据，这份数据已经处理了所有试剂的缩写或者英文名，将所有化学试剂转成了标准smiles格式，并且只使用规定的24个动作。
转成verl格式后存到了/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed中。

###奖励函数设计
目前有一个不太好的版本/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/reward
你可以按我下面的要求再构建一个reward

# 化学实验 RL 训练奖励模型 (Reward Model) 架构设计书

在当前强化学习微调阶段，我们的目标是引导 Qwen-4B-instruct 模型生成极其严谨的化学实验操作序列。目前的训练基座建立在 1000 条高质量、已清洗的规范化数据之上。在这批数据中，所有非标试剂俗名已被统一转换为规范的 SMILES 字符串，且所有操作均严格限定在 24 个基础原子动作（如 ADD, STIR, YIELD 等）之内。整体奖励函数的设计被优雅地解耦为三个维度，其总得分的计算公式为 $Reward = 0.2 \cdot R_{format} + 0.2 \cdot R_{lcs} + 0.6 \cdot R_{step}$。

### 格式与守恒惩罚模块设计 ($R_{format}$)

格式得分 $R_{format}$ 的初始基准值为 1.0，通过一系列基于惩罚的机制向 0 衰减。在该模块的工程实现中，首先需要进行宏观的 XML 标签平衡校验，确保完整的 `<think>` 思考过程和 `<answer>` 结论被正确输出。随后，需要利用正则表达式的向后断言（Lookbehind）技术，对文本中出现的所有 24 个合法动作词汇进行“裸奔”扫描。具体而言，算法必须检查任何出现的合法动作词汇（例如 ADD 或 CONCENTRATE）其前方是否紧密跟随了 `<procedure>` 标签。如果发现了未被包裹的动作指令，或者在一个 `<procedure>` 闭合区间内提取出了多个合法动作，则判定为非原子性操作并施加扣分惩罚。

防止smiles、temp、time裸露，SMILES 规范：如果出现疑似 SMILES 但没有被 [MOL] ```...``` [/MOL] 包裹（所谓的裸露 SMILES），进行扣分。时间与温度规范：如果文本中出现了诸如 "room temperature" 或 "2 hours" 却没有被 <temp> 或 <time> 包裹，进行扣分。系统在入口处需解析 prompt 中提供的 `REACTANT`、`PRODUCT`、`CATALYST` 和 `SOLVENT` 字段中的smiles，检查answer中是否完全覆盖，遗漏扣分。

构建全局化学同义词映射表,防止模型对于常用试剂只输出了同义的英文名、缩写而非smiles，例如water、THF导致的物质匹配失败。
### 宏观动作序列对齐模块 ($R_{lcs}$)

模型必须首先学会在不考虑具体试剂和用量的前提下，复刻正确的实验流程。我们从模型预测的完整输出和 Ground Truth (GT) 中按顺序提取出纯粹的动作词序列，分别记为 $A_{pred}$ 和 $A_{gt}$，其对应的序列长度为 $N_{pred}$ 和 $N_{gt}$。 为了评估序列的拓扑一致性，系统需要利用动态规划算法计算这两个序列的最长公共子序列（Longest Common Subsequence, LCS）。

为了实现针对模型“漏步”和“啰嗦”的双向惩罚，序列匹配得分的数学建模定义为 $R_{lcs} = \frac{|LCS|}{\max(N_{pred}, N_{gt})}$。在这一公式下，如果模型为了碰运气而生成了无限长的动作序列，分母 $N_{pred}$ 的急剧膨胀将使得该项得分趋近于零；反之，如果模型跳过了核心的纯化或搅拌步骤导致 $N_{pred}$ 小于 $N_{gt}$，得分同样无法满额。这种设计强迫模型在动作的先后逻辑和步骤的精简度上向 GT 完美收敛。

### 步骤得分模块 ($R_{step}$)

这是决定模型能否真正理解化学计量学的核心模块。微观步骤得分仅针对在 LCS 算法中被成功对齐的 $|LCS|$ 个步骤对进行计算。对于第 $k$ 个对齐的步骤对 $(p_k, g_k)$，其单步得分采取累加计算，对于匹配上的步骤，可以分三部分进行对比molecule、quantity、condition。S_step =S_action+S_mol+S_qty，这样对于复杂的步骤可以多给分。

对于molecule得分，解析第k个步骤中smiles的以及可能为缩写、英文名的试剂，放入集合中，计算S_mol =
|intersection(pred_mol, gt_mol)|/|union(pred_mol, gt_mol)|

对于quantity得分，在进行用量提取时，需要通过正则捕获所有的数值和单位组合。我们采用“单单位匹配即通关”的工程策略，只要在模型预测的用量体系中找到任意一个能与 GT 单位同量纲对齐的数值，即可计算相对误差 $E$。Score_qty = exp(-α * relative_error)

condition得分，负责处理 `<temp>` 和 `<time>`。在代码逻辑上，需要将自然语言或模糊数值映射为数学区间，注意常用自然语言ambient temperature、room temperature、overnight、−10° C、2-3 hours、45 min。注意如何提取<temp>below 40°C</temp>; STIR for <time>2-3 hours</time> at <temp>room temperature</temp>;这样的区间值。注意单数字、区间的匹配。匹配上得一分，有时只会出现温度、有时只会出现时间、有时两者都有，这是两者都匹配上才得一分。

最后进行归一化得分pred_signal_score / gt_signal_total（理论最大值）

#测试
你需要对你的奖励函数进行严苛的debug测试，验证打分符合预期，且能预判不同的真实状况，模型胡言论语、模型重复输出、模型投机取巧、模型完全不按格式走、模型smiles格式输出错误率高、模型大量输出英文名和缩写而非smiles等等。

###注意事项：
真实数据可以在/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed中读取，训练数据为900条数据，你可以从中抽取作为你测试的标准已经预判一些可能出现的意外，
不需要去翻阅verl文件夹里的代码
verl reward设计接口如下：
在verl框架中，你可以通过配置custom_reward_function来自定义奖励函数。主要有两种方式：基于规则的奖励函数和基于模型的奖励函数。

配置方式
在配置文件中设置custom_reward_function字段：

custom_reward_function:  
  path: /path/to/your/reward_function.py  # 你的奖励函数文件路径  
  name: compute_score  # 函数名称，默认为compute_score
_generated_ppo_trainer.yaml:512-514

奖励函数接口
你的奖励函数必须接受以下参数：
def compute_score(data_source, solution_str, ground_truth, extra_info=None):  
    """  
    Args:  
        data_source: 数据集名称  
        solution_str: 模型生成的回答  
        ground_truth: 标准答案  
        extra_info: 额外信息字典  
    Returns:  
        float 或 dict: 如果返回dict，必须包含"score"键  
    """  
    # 你的奖励计算逻辑  
    return score  # 或 {"score": score, "other_info": ...}
reward_function.rst:60-66

简单示例
def my_reward_function(data_source, solution_str, ground_truth, extra_info=None):  
    # 基于回答长度的简单奖励  
    return len(solution_str) / 100




