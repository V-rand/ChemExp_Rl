#/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl项目目标
在虚拟环境中工作conda activate chemexp_verl
设计一个适合verl rl训练的reward model
###rl 训练任务为：对于Qwen-4B-instruct模型，为模型输入反应物（包含用量）生成物（包含参量）以及可能提供的必要溶剂和催化剂，模型输出think思考过程以及化学实验步骤
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
原始数据通过hf下载，共2万数据，可以通过/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/download_dataset.py下载
示例数据为/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/raw/example_data.jsonl，共十条，你可以进去查看
为了适配verl的工作流，需要先转换格式/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/data_processing.py，并在这过程构建prompt和将smiles格式变为标准smiles
训练数据存到/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/data/processed

###奖励函数设计
目前有一个不太好的版本/home/xiemingjie/WorkSpace/Exp_project/ChemExp_Rl/reward
你可以按我下面的要求再构建一个reward




###注意事项：
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




