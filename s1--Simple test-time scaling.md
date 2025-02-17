- [s1：1000条数据SFT实现o1-like推理](#s11000条数据sft实现o1-like推理)
  - [主要内容](#主要内容)
    - [作者和团队信息](#作者和团队信息)
    - [背景和动机](#背景和动机)
  - [相关研究](#相关研究)
  - [核心思路](#核心思路)
  - [方案与技术](#方案与技术)
  - [实验与结论](#实验与结论)
  - [贡献](#贡献)
  - [不足](#不足)
  - [QA](#qa)
    - [Q1：s1K 数据集是如何保证「质量」、「难度」和「多样性」的？](#q1s1k-数据集是如何保证质量难度和多样性的)
    - [Q2：什么是「预算强制 (Budget Forcing)」？它是如何控制测试时计算量的？](#q2什么是预算强制-budget-forcing它是如何控制测试时计算量的)
    - [Q3：什么是Wait token？](#q3什么是wait-token)
    - [Q4：和多数表决（majority voting）相比，为什么作者更推荐 sequential scaling？](#q4和多数表决majority-voting相比为什么作者更推荐-sequential-scaling)
    - [Q5：budget forcing 具体是怎么操作的？](#q5budget-forcing-具体是怎么操作的)
    - [Q6：未来是否可以结合强化学习（RL）或搜索？](#q6未来是否可以结合强化学习rl或搜索)
    - [Q7：若想在自己项目复现该思路，核心步骤是什么？](#q7若想在自己项目复现该思路核心步骤是什么)
  - [伪代码](#伪代码)


# s1：1000条数据SFT实现o1-like推理

李飞飞团队最近（2025年1月）公布的一篇文章，通过在一个精心策划的小规模数据集上进行微调，并结合预算强制技术，实现了强大的推理能力和测试时计算扩展性。其实稍微有点夸张，s1k数据集的问题虽然是真实的，但答案和推理步骤是Gemini Flash Thinking API生成的，所以某种程度讲，这个推理能力其实可以看做是从gemini蒸馏而来的。不过这个工作也挺有意义，一定程度上验证了「浅层对齐假说」是有效的，只要数据够好，针对特定任务SFT就能涨点。

> **Gemini Flash Thinking API 介绍**
>
> Gemini Flash Thinking API是指谷歌开发的一款人工智能模型接口，用于生成具有详细推理路径的答案。这款API允许开发者通过接入Gemini 2.0 Flash Thinking这一实验性多模态推理模型来实现在多种应用场景下的智能化需求。Gemini Flash Thinking API的主要特点如下：
>
> - 快速生成答案：此API能够在短时间内解答复杂的数学、物理等问题，展示出高效的推理速度；
> - 显示完整思维流程：相比其他模型只提供最终答案的做法，Gemini Flash Thinking API特别之处在于它会展示其背后的推理逻辑及步骤，让用户了解它是如何得出结论的；
> - 广泛的应用场景：适用于包括但不限于教育辅助、科学研究、创意写作等领域的需求；
> - 技术和实践创新的价值：除了直接的功能之外，Gemini Flash Thinking API也为研究者提供了重要的资源库。例如，在论文《s1: simple test-time scaling》的研究中，研究人员便利用了该API生成的大约1000个样本的推理路径及其对应答案，以此为基础构建了小型数据集，进而探究了通过有限量级的数据训练达到接近顶级语言模型性能的可能性。这些样本的质量和多样性直接决定了模型微调后的泛化能力，从而成为一种新颖的深度学习研究范式——“浅层对齐假设”。
>
> 综上所述，Gemini Flash Thinking API不仅是连接理论与实际应用之间的桥梁，更是推动整个AI行业向着更加透明和可解释的方向发展的催化剂。
> 
> **浅层对齐假说**
> - 浅层对齐假说的基本观点是语言模型的深度学习能力已经在预训练期间获得了，对于特定的任务，只需通过有限的数据进行监督微调（SFT），就能够显著改善模型的表现。
> - 实验结果表明，在一个小规模数据集（如s1k数据集中仅有的1000个样本）上的微调可以增强模型的推理能力，这支持了浅层对齐假说的有效性。
> - 更广泛地说，浅层对齐假说提示了通过精心设计的微调而非依赖海量数据集，也可以大幅度提升大型语言模型的质量，尤其是它们的实用性和针对性。
>
> 这种假说不仅减少了传统RLHF（Reinforcement Learning from Human Feedback）中高昂的成本和技术复杂度，还可能促进更加高效且经济的模型部署方案。此外，它也促进了对现有大模型基础能力的理解，并激发了关于如何更好地激活和展示模型潜力的新研究方向。

## 主要内容

### 作者和团队信息

这篇论文由 Niklas Muennighoff、Zitong Yang、Weijia Shi、Xiang Lisa Li、Li Fei-Fei、Hannaneh Hajishirzi、Luke Zettlemoyer、Percy Liang、Emmanuel Candès 和 Tatsunori Hashimoto 共同完成。

- **主要贡献者**
  - **Niklas Muennighoff**：在多模态和多语言模型的结合方面有突出贡献，比如著名的多语言模型OLMo。
  - **Luke Zettlemoyer 和 Percy Liang**：在自然语言处理领域有很高的声誉，尤其在语义解析、问答系统和可解释性方面。他们领导的斯坦福大学NLP小组在相关方向有很深入的研究。
  - **Li Fei-Fei**：在计算机视觉领域做出了卓越贡献，尤其是在图像识别、物体检测和大规模数据集构建方面。
- **团队背景**
  - 该团队成员来自斯坦福大学、华盛顿大学等知名学府，具有强大的学术背景和研究实力。

### 背景和动机

- **发表时间**：2025年1月（arxiv预印版）。
- **研究问题**：如何在测试时有效地扩展语言模型的计算量，以提高其推理性能，同时保持方法和数据的简洁性。
- **问题背景**
  - 近年来，语言模型的能力提升主要依赖于训练时计算量的增加。
  - OpenAI 的 o1 模型展示了测试时计算扩展的潜力，但未公开具体方法，引发了大量的研究和复现尝试。
  - 之前的复现工作主要集中在使用大规模强化学习、蒙特卡洛树搜索等复杂技术，但这些方法通常需要大量的资源和数据。

## 相关研究

- **现有方法**
  - 大规模强化学习 (RL)：通过大量的训练数据和复杂的强化学习算法来优化模型的推理策略 (OpenAI, 2024; DeepSeek-AI et al., 2025)。
  - 蒙特卡洛树搜索 (Monte Carlo Tree Search, MCTS)：通过模拟多次推理过程，选择最优的推理路径 (Gao et al., 2024b; Zhang et al., 2024a)。
  - 多智能体方法 (Multi-agent Approaches)：使用多个智能体协同推理，提高模型的鲁棒性和准确性 (Qin et al., 2024)。
- **不足之处**
  - 这些方法通常需要大量的计算资源、数据和复杂的训练流程。
  - 难以复现 OpenAI o1 模型的测试时计算扩展能力。
  - 缺乏对测试时计算扩展方法的可控性和可解释性。

## 核心思路

- **核心观点**：通过在一个精心策划的小规模数据集上进行微调，并结合一种简单的测试时计算控制技术（预算强制），可以实现强大的推理能力和测试时计算扩展性。
- **灵感来源**
  - 对 OpenAI o1 模型的复现尝试的失败，促使作者思考是否可以通过更简单的方法实现相同的效果。
  - LIMA (Zhou et al., 2023) 提出的「浅层对齐假说 (Superficial Alignment Hypothesis)」，即少量的样本可以使模型与人类偏好对齐。

## 方案与技术

- **数据集构建**
  - 初始数据集：从 16 个不同的来源收集了 59,029 个问题，涵盖数学、科学、逻辑等领域。
  - 数据集筛选：通过三个标准（质量、难度和多样性）对初始数据集进行筛选，最终得到包含 1,000 个样本的 s1K 数据集。
    - 质量：排除包含格式错误、图像引用错误等低质量样本。
    - 难度：排除可以被 Qwen2.5-7B-Instruct 和 Qwen2.5-32B-Instruct 模型轻松解决的问题。
    - 多样性：使用 Claude 3.5 Sonnet 将问题分类到不同的领域，并确保最终数据集包含 50 个不同的领域。
- **模型训练**
  - 使用 s1K 数据集对 Qwen2.5-32B-Instruct 模型进行监督微调 (SFT)。
  - 训练时长仅为 26 分钟 (16 H100 GPUs)。
- **测试时计算扩展**
  - 预算强制 (Budget Forcing)：
    - 强制结束 (Early Exit)：如果模型生成的 token 数量超过预设的最大值，则强制结束思考过程，并输出答案。
    - 延长思考 (Wait)：如果模型尝试提前结束思考过程，则通过添加 Wait token 来鼓励模型进行更多的探索。
- **对比方法**
  - 条件长度控制 (Conditional Length-Control Methods)：
    - Token-conditional control：在prompt中指定最大token数量。
    - Step-conditional control：在prompt中指定最大步数 (每个step约100个token)。
    - Class-conditional control：使用不同的prompt来引导模型进行长/短时间的思考。
  - 拒绝采样 (Rejection Sampling)：对生成结果进行多次采样，直到生成结果满足预定的计算预算。
  
## 实验与结论

- **实验设计**
  - 数据集：AIME24、MATH500 和 GPQA Diamond。
  - 评估指标：准确率 (Performance)、可控性 (Control) 和缩放比例 (Scaling)。
    - 准确率 (Performance)：模型在基准测试中获得的最高准确率。
    - 可控性 (Control)：测试时计算控制方法能够多大程度上满足计算预算。
    - 缩放比例 (Scaling)：随着计算预算的增加，模型性能的提升程度。
- **实验结果**
  - s1-32B 模型在 AIME24 数据集上超过了 o1-preview 模型，并且在 MATH 和 AIME24 数据集上实现了高达 27% 的性能提升。
  - 通过预算强制，s1-32B 能够有效地扩展测试时计算，并在 AIME24 数据集上将性能从 50% 提升到 57%。
  - s1-32B 在样本效率方面优于其他模型，仅使用 1,000 个样本就达到了与使用更多数据训练的模型相媲美的性能。

## 贡献

- **提出了一种简单、高效的测试时计算扩展方法**：通过预算强制，可以有效地控制模型的推理过程，并在不增加模型大小的情况下提高性能。
- **构建了一个高质量的小规模数据集**：s1K 数据集包含了高质量、多样化和具有挑战性的问题，可以用于训练具有强大推理能力的语言模型。
- **验证了浅层对齐假说**：通过在小规模数据集上进行微调，可以有效地激活预训练模型中已有的推理能力。

## 不足

- 预算强制的局限性
  - 过度使用 Wait token 可能会导致模型陷入重复循环，从而限制性能提升。
  - 模型性能最终会受到上下文窗口大小的限制。
- 数据集的局限性
  - s1K 数据集主要集中在数学和科学领域，可能无法很好地泛化到其他领域。
  - 数据集的构建依赖于 Gemini Flash Thinking API，可能存在偏差

## QA

### Q1：s1K 数据集是如何保证「质量」、「难度」和「多样性」的？

- **质量**：
  - 人工检查：作者会检查原始数据集，排除格式错误、乱码等低质量样本。
  - API 错误过滤：排除通过 Gemini API 生成 reasoning trace 失败的样本。
- **难度**：
  - 模型评估：使用 Qwen2.5-7B-Instruct 和 Qwen2.5-32B-Instruct 模型评估问题的难度，排除可以被这些模型轻松解决的问题。
  - 推理轨迹长度：假设更难的问题需要更长的思考过程，因此选择具有更长推理轨迹的样本。
- **多样性**：
  - 领域分类：使用 Claude 3.5 Sonnet 将问题分类到不同的领域，并确保最终数据集包含 50 个不同的领域。
  - 随机抽样：在每个领域中随机抽样问题，以保证领域之间的平衡。

### Q2：什么是「预算强制 (Budget Forcing)」？它是如何控制测试时计算量的？

预算强制是一种在测试时控制语言模型计算量的方法。具体来说，它包含以下两种策略：

- 强制结束 (Early Exit)：如果模型生成的 token 数量超过预设的最大值，则强制结束思考过程，并输出答案。这可以防止模型过度思考，浪费计算资源。
- 延长思考 (Wait)：如果模型尝试提前结束思考过程，则通过添加 Wait token 来鼓励模型进行更多的探索。这可以促使模型进行更深入的思考，提高性能。
  
通过调整最大 token 数量和 Wait token 的使用频率，可以灵活地控制模型在测试时的计算量。

### Q3：什么是Wait token？

Wait token 其实指的是在推理过程中插入的一个触发词（例如直接使用单词「Wait」），用来告诉模型「不要急着结束思考阶段，请继续推理」。它并非某种特殊符号，而更像是一段能诱导模型继续输出思考内容的提示文字。

不过，作者发现当「Wait」次数过多，模型容易重复胡扯或陷入循环，收益边际递减。上下文窗口也会限制思考长度。实验显示到达一定次数后，准确率的提升趋于平稳。

### Q4：和多数表决（majority voting）相比，为什么作者更推荐 sequential scaling？

多数表决是一种「平行式扩展」，它一次性生成很多独立答案，再投票选最优。但这样做不能让后一次思考依据前一次调试或改进。sequential scaling（如 budget forcing）是一种「迭代式」的推理，可以在后面推理步骤里修正之前的错误，更有机会得到更高的上限。

### Q5：budget forcing 具体是怎么操作的？

在解码时，如果「思考阶段」的生成超过预算，就插入一个结束符让模型立即产出答案（相当于强制结束思考）。若想让模型思考更久，则抑制结束符并在思考文本后附加Wait等触发词，让模型继续写下思考过程，从而「延长」推理并可能纠正之前的错误推理。

### Q6：未来是否可以结合强化学习（RL）或搜索？

文章也提到如 MCTS、REBASE 等方法能在推理时引入搜索或过程奖励。budget forcing 是一种简单可控的启发式手段，未来可将其与 RL 搜索策略结合，以期在「深度搜索」和「可控延长推理」上得到更好的效果。

### Q7：若想在自己项目复现该思路，核心步骤是什么？

- 准备小而精的数据集（1000 条左右，高难度、多风格），包含详细推理过程；
用现有大模型进行短时间的监督微调；
- 在推理阶段实现类似 budget forcing 的机制（即在生成思考时，检测思考标记并加以控制或延长）；
- 在任务上验证不同预算时推理性能随「思考长度」的曲线是否真的上升，从而实现 test-time scaling。
  

## 伪代码

下面这段示例性代码演示了论文核心的两个阶段：
- 第一阶段：用一小批量带「思考过程（reasoning trace）」的数据来微调一个现成的大语言模型（示意性的s1K数据和Qwen2.5-32B-Instruct为例）。
- 第二阶段：在推理阶段实现「budget forcing」，即可控地截断或延长模型的思考过程。

```python
"""
以下示例使用了Hugging Face的transformers库，并模拟了论文中的核心思路：
1. 准备少量高质量的带“思考过程+答案”的数据（s1K）。
2. 用这些数据对一个已训练好的大模型（如Qwen2.5-32B-Instruct）进行再微调。
3. 实现'budget forcing'：在解码时可控地结束或延长思考过程。
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
import math
import random

#────────────────────────────────────────────────────────────────────────────
# 假设我们已经有一个小规模的、结构如下的训练数据：
# 每个样本有：
#   "question"：题目
#   "reasoning"：模型产生的思考过程文本
#   "answer"：最终答案
# 其中 reason + answer 会被拼接起来做微调。
# "
# 下面仅用一个列表进行示例，实际中应替换为真正的s1K数据。
#────────────────────────────────────────────────────────────────────────────

MOCK_S1K_DATA = [
    {
        "question": "用三角形面积公式推导以下函数在[0,1]区间的积分结果。",
        "reasoning": "首先，我们考察此函数......(此处省略大量中间思考过程)......所以积分值为1",
        "answer": "答案：1"
    },
    {
        "question": "若有长度为N的数列，如何求最大连续子段和？",
        "reasoning": "先用动态规划......(中间推理过程)......最终结果由dp数组求得。",
        "answer": "答案：dp数组的最大值"
    },
    # ...... 省略更多的题目与推理示例
]

#────────────────────────────────────────────────────────────────────────────
# 1. 构建一个Dataset，将上述示例数据处理成可微调形式
#   - 在真实场景下，我们会有1K条（s1K）或其他规模的数据。
#   - 我们需要将“question + reasoning + answer”拼接，但一般只对 reason+answer 部分算loss(因question只是上下文)。
#   - 这里做一个简单范例。
#────────────────────────────────────────────────────────────────────────────

class S1KDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # question 是上下文引导，reasoning+answer 作为需微调学习的目标
        question_prompt = f"问题：{sample['question']}\n思考："
        reasoning_part = sample['reasoning']
        answer_part = f"\n最终答案：{sample['answer']}"

        # 构造输入: [question_prompt + reasoning_part + answer_part]
        # 其中可以用特殊标记分割，以便模型学习何时 "思考" 何时 "回答"
        # 这里为了简单就串接在一起
        input_text = question_prompt + reasoning_part + answer_part

        # 注意：要确保最大长度截断策略
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=self.max_length
        )
        # 注意：在实际论文中，作者对question部分可能不计算loss，保证只在(思考 + 答案)上训练
        # 这里为了简化，直接对整段文本计算loss。
        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            # 简化：若需要更精细控制loss mask，可自行构造 labels
            "labels": torch.tensor(input_ids, dtype=torch.long)
        }

#────────────────────────────────────────────────────────────────────────────
# 2. 模型与微调设置
#   - 注：Qwen2.5-32B-Instruct可能需要额外安装企业版/特殊处理，这里以“AutoModelForCausalLM”示例
#   - 实际环境中，需确保你有权访问并安装相应的模型权重
#────────────────────────────────────────────────────────────────────────────

def train_s1_model():
    model_name_or_path = "Qwen/Qwen-2.5-7B-Example"  # 仅举例；真实场景替换成Qwen2.5-32B等
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

    # 构建数据集
    train_dataset = S1KDataset(MOCK_S1K_DATA, tokenizer)

    # 简单的训练参数
    training_args = TrainingArguments(
        output_dir="s1-model-output",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        logging_steps=1,
        # 仅演示，所以不做过多的参数调优
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 开始训练（演示）
    trainer.train()

    # 训练完成后保存
    trainer.save_model("s1-model-checkpoint")
    tokenizer.save_pretrained("s1-model-checkpoint")

    return "s1-model-checkpoint"

#────────────────────────────────────────────────────────────────────────────
# 3. 推理阶段（test-time scaling）：演示budget forcing
#   “budget forcing”主要指：可控地结束/延长模型的“思考过程”
#   - 如果到达思考token上限就结束思考并输出答案
#   - 如果希望延长，就抑制end-of-thinking并插入一个"Wait"来让模型继续
# 下面伪代码结构展示如何在解码时操作
#────────────────────────────────────────────────────────────────────────────

def budget_forcing_generate(
    model, tokenizer, prompt, max_thinking_tokens=256, force_extra_thoughts=0
):
    """
    :param model: 已微调好的模型
    :param tokenizer: 分词器
    :param prompt: 用户问题上下文
    :param max_thinking_tokens: 在'思考'阶段允许的最大token数
    :param force_extra_thoughts: 需要额外强行延长多少次"Wait"
    """
    # 先encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # 我们需要在生成时分两阶段：
    # 1) "思考阶段" 2) "答案阶段"
    # 这里做一个最简化：只展示如何在思考阶段插入 Wait
    # 实际可能要用一些特别的标记 <|im_start|>think ... <|im_start|>answer
    # 但这里是演示代码

    # 初始输出序列就先等于input_ids
    generated = input_ids.clone()

    # 假设我们定义"end-of-thinking"符号是一个特殊的token id, 这里伪造一下
    end_of_thinking_token_id = 999999  # 在真实应用中需替换实际ID

    # 步骤A: 先做"思考阶段"的生成
    # 做一个简单循环，边生成边检查
    model.eval()
    with torch.no_grad():
        for _ in range(max_thinking_tokens):
            outputs = model(input_ids=generated)
            logits = outputs.logits
            next_token_logits = logits[:, -1, :]

            # 可以在这里做一些logits处理：如温度/采样
            # 简单起见，取argmax
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # 如果下一个token是end-of-thinking，则停止
            if next_token_id.item() == end_of_thinking_token_id:
                # end-of-thinking 出现，思考结束
                break

            # 否则拼到generated里
            generated = torch.cat((generated, next_token_id), dim=1)

        # 若我们希望强行多想 force_extra_thoughts 次，每次就不让它用end-of-thinking
        # 而是自动加"Wait"。
        # 在实际操作上你需要写一个循环：抑制<end-of-thinking_token_id>后续，然后手动插入Wait之类
        # 这里只做一个简单演示：
        for _ in range(force_extra_thoughts):
            # 在真实模型里, "Wait" 也需转为token。我们就用tokenizer转换一下
            wait_ids = tokenizer.encode(" Wait", return_tensors="pt")
            generated = torch.cat((generated, wait_ids), dim=1)
            # 然后再继续若干步思考
            # (同上做法, 省略具体逻辑)

    # 步骤B: 进入答案阶段
    # 通常在CoT场景，我们用一种标记<|answer|>或者"答案："来提醒模型把最终答案输出
    # 假设思考阶段生成结束后，我们人为接一个"答案："标记
    answer_trigger_ids = tokenizer.encode("\n答案：", return_tensors="pt")
    generated = torch.cat((generated, answer_trigger_ids), dim=1)

    # 在答案阶段，我们可以再做若干token的自由生成
    answer_output = model.generate(
        generated,
        max_new_tokens=128,
        do_sample=False  # 简化: 直接贪心
    )

    # 解码
    full_text = tokenizer.decode(answer_output[0], skip_special_tokens=True)

    return full_text

#────────────────────────────────────────────────────────────────────────────
# 4. 综合示例：演示如何使用
#────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # (A) 演示微调
    checkpoint = train_s1_model()

    # (B) 加载并做推理
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint)

    test_prompt = "请计算以下函数在区间[0,1]的积分，并给出详细推理。"

    # (B1) 不做强制延长思考
    normal_result = budget_forcing_generate(
        model, tokenizer, test_prompt, max_thinking_tokens=50, force_extra_thoughts=0
    )
    print("普通推理结果：", normal_result)

    # (B2) 强制多增2次Wait，让模型更多思考
    extended_result = budget_forcing_generate(
        model, tokenizer, test_prompt, max_thinking_tokens=50, force_extra_thoughts=2
    )
    print("延长思考推理：", extended_result)

"""
总结：
1. 我们先构造一个小规模带“思考过程”的s1K样例（此处仅mock）。
2. 用该数据微调一个大语言模型，并将"question + reasoning + answer"打包成训练输入。
3. 在推理时，通过 budget_forcing_generate 实现“可控地终止或延长思考”。
   - 若到达思考token上限则结束
   - 若想多让模型思考，就抑制结束符、插入 Wait 来继续生成
以上流程对应论文内提出的关键点，可根据需求进行更精细化实现。
"""
```



>
> refer to: https://zhuanlan.zhihu.com/p/21602993558
>
