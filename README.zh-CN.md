# CALR: Guiding Reasoning Models toward Efficient Reasoning with Correctness-Aware Length Rewards

[English](README.md) | 简体中文

**CALR**是一种用于数学推理的长度奖励方法，旨在减少模型的输出Token，同时尽量保持正确率。该方法基于GRPO，以同题采样中的最短正确回答为长度参照，仅对正确回答施加有界的相对长度惩罚，错误回答的奖励保持为零，从而在优先保证答案正确性的基础上鼓励简洁推理。

实验基于Qwen3.5-2B，在MATH500上，CALR相比原始模型减少了**6.72%**的平均输出Token，准确率仅由70.8%降至70.0%，相比GRPO，准确率提高2.0%，平均输出Token减少10.85%。在AIME24上，CALR和原始模型保持相同的正确题数，并减少3.89%的平均输出Token。



## 方法

### 设计思路

GRPO的奖励函数为所有正确回答给1分、错误回答给0分，没有显式鼓励模型减少冗余输出。若直接在其基础上引入长度奖励，即直接对所有回答按长度扣分，又可能让短的错误回答优于长的正确回答。

CALR保留正确性优先的原始奖励排序：错误回答始终为0，正确回答再根据相对长度获得不同奖励。长度参照来自同一道题的正确回答，而不是所有题共用一个固定目标。

### 奖励计算

对同一道题采样8条回答。若其中有正确回答，取最短正确回答的Token数作为参照长度，计算每条正确回答相对参照长度的超额比例，再映射为有界惩罚：

$$
d_i=\frac{\max(0,L_i-L_{\min}^{+})}{\max(L_{\min}^{+},\tau)},\qquad
R_i=\begin{cases}
1-\alpha\dfrac{d_i}{1+d_i}, & c_i=1,\\
0, & c_i=0.
\end{cases}
$$

| 符号 | 含义 |
|---|---|
| c | 答案正确为1，错误为0 |
| L | 当前回答的输出Token数 |
| Lmin+ | 本组最短正确回答的输出Token数 |
| d | 相对参照长度的超额比例 |
| α = 0.1 | 正确回答的长度扣分上限 |
| τ = 256 | 归一化分母的下限，避免参照长度过短时放大长度差 |

最短正确回答获得1分，其他正确回答的奖励介于0.9与1之间；错误回答无论长短都为0。如果整组全部答错，则奖励全部为0。这个设计鼓励缩短正确解答，但不会仅因错误回答更短就给它更高的奖励。

训练保留组内标准化：奖励为正不等于优势为正，较长的正确回答仍可能得到负优势。



## 实验结果

实验基于Qwen3.5-2B。GRPO和CALR从原始模型开始训练，使用相同的1600道MATH题训练100步。SFT使用1887条Qwen3.5-27B教师轨迹进行QLoRA微调，作为独立对照。

<table>
  <thead>
    <tr>
      <th rowspan="2">Dataset</th>
      <th colspan="2">Base</th><th colspan="2">SFT</th>
      <th colspan="2">GRPO</th><th colspan="2">CALR</th>
    </tr>
    <tr>
      <th>Acc. ↑</th><th>Len. ↓</th><th>Acc. ↑</th><th>Len. ↓</th>
      <th>Acc. ↑</th><th>Len. ↓</th><th>Acc. ↑</th><th>Len. ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="9"><em>In-domain Tasks</em></td></tr>
    <tr>
      <td>MATH500</td>
      <td><strong>70.80</strong></td><td>1556.53</td><td>65.60</td><td>1719.06</td>
      <td>68.00</td><td>1628.61</td><td>70.00</td><td><strong>1451.96</strong></td>
    </tr>
    <tr><td colspan="9"><em>Out-of-domain Tasks</em></td></tr>
    <tr>
      <td>AIME24</td>
      <td>13.33</td><td>6768.37</td><td>10.00</td><td>6834.50</td>
      <td><strong>16.67</strong></td><td>7352.67</td><td>13.33</td><td><strong>6505.13</strong></td>
    </tr>
  </tbody>
</table>

Acc.为准确率（%），Len.为全部回答的平均输出Token。

CALR相比Base，在MATH500上减少6.72%的输出Token，而准确率仅下降0.8%；在AIME24上减少3.89%的Token，正确题数同为4/30。

完整结果见[results/summary.json](results/summary.json)；各方法目录保留逐题回答、评分与训练指标。



## 实验设置

| 项目 | 说明 |
|---|---|
| RL训练 | 基于TRL实现GRPO与CALR，采用LoRA微调和单卡BF16训练 |
| 训练硬件 | 单张NVIDIA RTX 6000D，84GB显存 |
| 回答生成 | 使用vLLM批量采样，每批最多生成16条回答，并与训练端同步模型权重 |



## 运行

项目使用Python3.12。

### 环境准备

```bash
uv sync --locked --extra gpu
export MODEL=/path/to/Qwen3.5-2B
```

需要将模型路径替换为实际位置。

### 训练

```bash
export METHOD=calr  # 训练GRPO时改为grpo
uv run --locked --extra gpu python main.py train --method "$METHOD" --model "$MODEL" \
  --export "runs/${METHOD}_model_copy" --output "runs/$METHOD" --run
```

### 测评

```bash
export DATASET=math500  # 测AIME24时改为aime

# 原始模型
uv run --locked --extra gpu python main.py evaluate --method base --model "$MODEL" \
  --dataset "$DATASET" --output "runs/base_$DATASET" --run

# 训练后的模型，METHOD与训练时一致
uv run --locked --extra gpu python main.py evaluate --method "$METHOD" --model "$MODEL" \
  --adapter "runs/$METHOD/final" --dataset "$DATASET" \
  --output "runs/${METHOD}_$DATASET" --run
```

测评自动评分，新实验结果保存在`runs/`，已有结果见`results/`。



## 目录

```text
main.py           # 训练、测评入口
pyproject.toml    # uv 依赖
uv.lock           # 固定依赖版本
data/             # 训练题、MATH500、AIME24
training/         # 奖励函数、LoRA 训练、权重同步与恢复
evaluation/       # 模型生成与答案判分
analysis/         # 评分及结果汇总
results/          # Base、SFT、GRPO、CALR 的结果与训练记录
```



## 致谢

本项目基于以下开源项目构建，并借鉴了相关研究的方法设计：

- [TRL](https://github.com/huggingface/trl)：提供GRPO训练框架。
- [vLLM](https://github.com/vllm-project/vllm)：用于训练采样和测评时的批量生成。
- [Training Language Models to Reason Efficiently](https://github.com/Zanette-Labs/efficient-reasoning)：CALR借鉴其正确性门控与有界长度惩罚设计。
- [ShorterBetter](https://github.com/JingyangYi/ShorterBetter)：CALR借鉴其以最短正确回答长度作为动态参照的思路。

感谢上述项目的作者、开发者与维护者对开源社区的贡献。

