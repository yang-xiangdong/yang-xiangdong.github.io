---
title: "大模型训练与推理加速技术"
date: 2026-03-07
description: "这份文档梳理了大模型从训练到推理的主要加速技术，涵盖数据/张量/流水线/序列并行、ZeRO、MoE、量化、KV Cache、FlashAttention 等核心方法。"
---

## 第零章  引言：为什么需要加速技术

### 0.1  大模型规模的爆炸式增长

自 2017 年 Transformer 架构（*Attention Is All You Need*）提出以来，语言模型的参数量以每年约 **10 倍**的速度增长，训练所需算力的增速更快。

```
模型            参数量 (B)     训练算力 (PetaFLOP-days)     GPU 数量(A100)
──────────────────────────────────────────────────────────────────────────
GPT-2            0.117              21                       1
GPT-3            175                3640                     1000
PaLM             540                29232                    6144
LLaMA-3          405                ---                      2048
GPT-4            1800 (est.)        ?                        20000+
```

### 0.2  三大核心挑战

训练与推理大模型面临三个正交的瓶颈，这也是所有加速技术的出发点：

| 挑战 | 根本原因 | 典型症状 |
|------|----------|----------|
| **显存（Memory）** | 参数 + 优化器状态 + 激活值超出单卡容量 | OOM，无法加载模型 |
| **算力（Compute）** | 矩阵乘法量级庞大 | 训练速度慢，吞吐低 |
| **通信（Communication）** | 多设备同步梯度/激活 | GPU 等待，利用率低 |

> **💡 核心思路**：所有并行策略本质上都是在"显存节省"、"算力扩展"与"通信开销"三者之间寻求最优权衡。

### 0.3  技术发展时间线

```
2016 : Ring All-Reduce (Baidu)
2017 : 数据并行 DDP / Horovod
2018 : 朴素模型并行 (GPipe / PipeDream)
2019 : Megatron-LM 张量并行 (NVIDIA)
2020 : ZeRO-1/2/3 (DeepSpeed, Microsoft)
2021 : 流水线并行优化 / 序列并行萌芽
2022 : ZeRO++ / MoE 规模化 / FlashAttention-1
2023 : FlashAttention-2 / AWQ / GPTQ / PagedAttention (vLLM)
2024 : FlashAttention-3 / FP8 训练 / 推测解码普及
```

---

## 第一章  数据并行（Data Parallelism）★

数据并行是工业界**使用最广泛**的并行策略，也是理解其他并行技术的基础。

### 1.1  基本思想

将一个 mini-batch 拆分成若干 micro-batch，分配给不同 GPU **各自独立计算**梯度，再通过 **All-Reduce** 汇总梯度后统一更新参数。每张 GPU 持有**完整的模型副本**。

```
         Input Batch (N samples)
                   |
        ┌──────────┼──────────┐
        │          │          │
      GPU-0      GPU-1      GPU-2      <-- 每个 GPU 持有完整模型副本，但是训练数据不同
    micro-b0   micro-b1   micro-b2
        │          │          │
        └─────────>│<─────────┘
              All-Reduce
           (梯度求和 / 平均)
                   │
              统一更新参数
```

### 1.2  Ring All-Reduce ★

Baidu 2017 年提出的 Ring All-Reduce 是现代数据并行的通信基础，将 All-Reduce 的通信量从 $O(N^2)$ 降至 $O(2 \cdot (N-1) \cdot \frac{\vert g \vert}{N})$，其中 $N$ 表示 GPU 数量，$\vert g \vert$ 表示完整梯度向量的维度，梯度向量会被切分为 $\frac{\vert g \vert}{N}$ 个 Chunk。通信效率在 $\lim_{N \rightarrow +\infty}$ 时达到最优。

```
Ring All-Reduce 两阶段示意（N = 3 GPU，梯度向量被切成 4 个 chunk：A, B, C, D）

// Step 0. 初始阶段

    GPU0: [A0, B0, C0, D0]
    GPU1: [A1, B1, C1, D1]
    GPU2: [A2, B2, C2, D2]
    GPU3: [A3, B3, C3, D3]

    >> 预期每个 GPU 都持有完整的梯度信息 [A0+A1+A2+A3, B0+B1+B2+B3, C0+C1+C2+C3, D0+D1+D2+D3]

// Step 1. Scatter-Reduce 阶段（每个 GPU 发送自己的梯度段给下一个）

    // Round 1
    GPU0 → GPU1 : A0 => GPU1: A1+A0
    GPU1 → GPU2 : B1 => GPU2: B2+B1
    GPU2 → GPU3 : C2 => GPU3: C3+C2
    GPU3 → GPU0 : D3 => GPU0: D0+D3

    // Round 2
    GPU0 → GPU1 : D0+D3 => GPU1: D1+D0+D3
    GPU1 → GPU2 : A0+A1 => GPU2: A2+A0+A1
    GPU2 → GPU3 : B1+B2 => GPU3: B3+B1+B2
    GPU3 → GPU0 : C2+C3 => GPU0: C0+C2+C3

    // Round 3
    GPU0 → GPU1 : C0+C2+C3 => GPU1: C1+C0+C2+C3
    GPU1 → GPU2 : D0+D1+D3 => GPU2: D2+D0+D1+D3
    GPU2 → GPU3 : A0+A1+A2 => GPU3: A3+A0+A1+A2
    GPU3 → GPU0 : B1+B2+B3 => GPU0: B0+B1+B2+B3

    >> 经过 N-1 轮后，每个 GPU 有且仅有一个梯度 Chunk 的完整信息。

// Step 2. All-Gather 阶段（广播完整段给所有 GPU）

    // Round 1
    GPU0 → GPU1 : B
    GPU1 → GPU2 : C
    GPU2 → GPU3 : D
    GPU3 → GPU0 : A

    // Round 2
    GPU0 → GPU1 : A
    GPU1 → GPU2 : B
    GPU2 → GPU3 : C
    GPU3 → GPU0 : D

    // Round 3
    GPU0 → GPU1 : D
    GPU1 → GPU2 : A
    GPU2 → GPU3 : B
    GPU3 → GPU0 : C

    >> 经过 N-1 轮后，每个 GPU 都持有了完整的梯度信息。
```

### 1.3  PyTorch DDP 实现

```python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# 初始化进程组（NCCL 后端适用于 GPU）
dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

model = MyModel().cuda(local_rank)
# DDP 包装：自动在 backward 后插入 All-Reduce hook
model = DDP(model, device_ids=[local_rank])

# 数据需要用 DistributedSampler 保证不同 GPU 拿到不同数据
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()   # <-- 梯度 All-Reduce 在这里自动触发
    optimizer.step()
```

> **⚙️ DDP 优化细节**：DDP 会将梯度 All-Reduce 与 backward 计算**重叠执行**（overlap），即某一层的梯度计算完毕后立刻开始通信，无需等待整个 backward 完成，大幅隐藏通信延迟。

### 1.4  梯度累积（Gradient Accumulation）

当单卡显存不够放入目标 batch size 时，可以通过多次小 batch 累积梯度再更新来**模拟大 batch**：

```python
accumulation_steps = 8   # 等效 batch size = 实际 batch × 8

for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 1.5  数据并行的显存瓶颈

数据并行**无法解决显存问题**：每张 GPU 仍需存储完整的：

```
显存占用 = 参数 + 梯度 + 优化器状态 + 激活值

以 7B 模型（FP16 混合精度）为例：
  参数        = 7B × 2 bytes = 14 GB
  梯度        = 7B × 2 bytes = 14 GB
  优化器状态  = 7B × 8 bytes = 56 GB  (Adam: m, v, FP32 param)
  激活值      ≈ batch-dependent
  ─────────────────────────────────────
  合计        ≈ 84 GB（不含激活）
```

单张 A100 80GB 显存已接近极限，这正是模型并行技术的动机。

---

## 第二章  模型并行（Model Parallelism）★

### 2.1  朴素模型并行（Naive Model Parallelism）

最简单的模型并行：将模型**按层**切分到不同 GPU。

```
         单机 4 GPU 朴素模型并行示意

  Layer 0-5   Layer 6-11  Layer 12-17  Layer 18-23
  ┌────────┐  ┌────────┐  ┌────────┐   ┌────────┐
  │ GPU-0  │─>│ GPU-1  │─>│ GPU-2  │──>│ GPU-3  │
  │(前向)  │   │(前向)  │  │(前向)  │   │(前向)  │
  └────────┘  └────────┘  └────────┘   └────────┘
       ↑            ↑           ↑            ↑
  (反向传播方向：梯度从 GPU-3 反向传回 GPU-0)
```

```python
# PyTorch 朴素模型并行示例
class NaiveModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1024, 1024).to('cuda:0')
        self.layer2 = nn.Linear(1024, 1024).to('cuda:1')
        self.layer3 = nn.Linear(1024, 512).to('cuda:2')

    def forward(self, x):
        x = self.layer1(x.to('cuda:0'))
        x = self.layer2(x.to('cuda:1'))   # 自动触发 GPU 间数据传输
        x = self.layer3(x.to('cuda:2'))
        return x
```

### 2.2  朴素模型并行的致命缺陷：GPU 空闲问题

```
  时间轴 ──────────────────────────────────────────────>

  GPU-0  [前向 B1][          空闲          ][反向 B1]
  GPU-1           [前向 B1][     空闲      ][反向 B1]
  GPU-2                    [前向 B1][空闲  ][反向 B1]
  GPU-3                             [前向/反向 B1]

  GPU 利用率：约 25%（N=4 时为 1/N）
  这种空闲气泡被称为 "Pipeline Bubble"
```

朴素模型并行同一时刻只有一张 GPU 在工作，利用率极低。**流水线并行**正是为解决这个问题而生（见第四章）。

---

## 第三章  张量并行（Tensor Parallelism）★★

> **参考论文**：Megatron-LM: *Efficient Large-Scale Language Model Training on GPU Clusters*（NVIDIA，2021）

### 3.1  核心思想

张量并行在**算子（Operation）内部**对参数矩阵进行切分，多个 GPU **同时计算**同一层的不同部分，然后通过 All-Reduce 或 All-Gather 合并结果。

这与模型并行（按层切分）的本质区别在于：**张量并行让多 GPU 协同完成同一个矩阵乘法**。

### 3.2  线性层的列切分与行切分

Transformer 中最核心的算子是线性层 `Y = XA`，Megatron-LM 提出了两种切分方式：

#### 列切分（Column Parallel Linear）

将权重矩阵 A 按列切分：

```
  A = [A1 | A2]    （A1, A2 分别在 GPU-0, GPU-1）

  GPU-0: Y1 = X · A1
  GPU-1: Y2 = X · A2

  结果：Y = [Y1 | Y2]  （沿列拼接）

  ┌───────────┐         ┌────────┐   ┌────────┐
  │    X      │ ──────> │ X · A1 │   │ X · A2 │
  │(broadcast)│         │  GPU-0 │   │  GPU-1 │
  └───────────┘         └────────┘   └────────┘
  通信：需要在输入端 All-Gather（或 broadcast）X
```

#### 行切分（Row Parallel Linear）

将权重矩阵 A 按行切分：

```
  A = [A1]   （A1 在 GPU-0，A2 在 GPU-1）
      [A2]

  GPU-0: Y_partial_0 = X1 · A1
  GPU-1: Y_partial_1 = X2 · A2

  结果：Y = Y_partial_0 + Y_partial_1  （需要 All-Reduce）

  ┌────┐        ┌──────────┐
  │ X1 │──────> │ X1 · A1  │──┐
  │GPU0│        └──────────┘  ├─ All-Reduce ──> Y
  │ X2 │──────> │ X2 · A2  │──┘
  │GPU1│        └──────────┘
  └────┘
```

### 3.3  Transformer MLP 层的张量并行

MLP 层结构：`Y = GeLU(XA) · B`，Megatron-LM 将其组合为"列切分 + 行切分"的无缝对接：

```
  ┌─────────────────────────────────────────────────────┐
  │                   MLP 张量并行                       │
  │                                                     │
  │  输入 X            第一个线性层（列切分）             │
  │  ┌───┐  f        ┌──────┐   ┌──────┐                │
  │  │   │──────────>│ XA1  │   │ XA2  │   (GeLU)       │
  │  │ X │  (all-    │ GPU0 │   │ GPU1 │                │
  │  │   │  gather)  └──────┘   └──────┘                │
  │  └───┘                                              │
  │                  第二个线性层（行切分）               │
  │            ┌──────────┐   ┌──────────┐              │
  │            │Y1·B1 GPU0│   │Y2·B2 GPU1│              │
  │            └────┬─────┘   └─────┬────┘              │
  │                 └───── g ───────┘                   │
  │                    (All-Reduce)                     │
  │                       │                             │
  │                    输出 Z                           │
  └─────────────────────────────────────────────────────┘

  注：f = 前向 identity / 反向 All-Reduce
      g = 前向 All-Reduce / 反向 identity
  这样整个 MLP 只需 2 次 All-Reduce（前向 1 次，反向 1 次）
```

### 3.4  Self-Attention 层的张量并行

Multi-Head Attention 天然适合按"头（Head）"切分：

```
  Multi-Head Attention 张量并行

  Q, K, V 投影（列切分，按 head 分配）：
  ┌─────────────────────────────────────┐
  │  GPU-0：Head 0,1,...,H/2-1          │
  │  GPU-1：Head H/2,...,H-1            │
  └─────────────────────────────────────┘

  各 GPU 独立计算自己负责的 head：
    Attention_i = softmax(Q_i K_i^T / √d_k) V_i

  输出投影（行切分 + All-Reduce）：
    Output = AllReduce(O_local)
```

### 3.5  张量并行的通信分析

```
  每个 Transformer 层的通信次数：
    MLP：前向 1 次 All-Reduce，反向 1 次 All-Reduce
    Attention：前向 1 次 All-Reduce，反向 1 次 All-Reduce
    合计：每层 4 次 All-Reduce

  通信量（All-Reduce per step）：
    = 2 × (tp_size - 1) / tp_size × 激活值大小
    当 tp_size=8 时，额外通信量 ≈ 1.75 × 激活值大小
```

> **⚠️ 张量并行的局限**：All-Reduce 要求 GPU 间**高速互联**（NVLink），通常只在同一台机器的 GPU 间使用（tp_size ≤ 8）。跨节点张量并行受限于 InfiniBand 带宽，效率急剧下降。

### 3.6  Megatron-LM 代码示例

```python
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear

class TensorParallelMLP(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super().__init__()
        # 列切分：每个 GPU 只存 ffn_size // tp_size 列
        self.dense_h_to_4h = ColumnParallelLinear(
            hidden_size, ffn_size,
            gather_output=False  # 不需要 gather，直接传给下一层
        )
        # 行切分：接收切分的输入，输出做 All-Reduce
        self.dense_4h_to_h = RowParallelLinear(
            ffn_size, hidden_size,
            input_is_parallel=True
        )

    def forward(self, x):
        x, _ = self.dense_h_to_4h(x)
        x = F.gelu(x)
        x, _ = self.dense_4h_to_h(x)
        return x
```

---

## 第四章  流水线并行（Pipeline Parallelism）★★

### 4.1  动机：解决 GPU 空闲气泡

朴素模型并行将层分配到不同 GPU，同一时刻只有一个 GPU 在工作。流水线并行通过**将 micro-batch 像工厂流水线一样调度**来提升利用率。

### 4.2  GPipe（Google，2019）

**核心思想**：将一个 mini-batch 切分成 M 个 micro-batch，依次送入流水线。

```
  GPipe 流水线（4 GPU，4 micro-batch）

  时间步  t1      t2      t3      t4      t5      t6      t7
  GPU-0  [F,m1] [F,m2] [F,m3]  [F,m4]                    [B,m4][B,m3][B,m2][B,m1]
  GPU-1         [F,m1] [F,m2]  [F,m3]   [F,m4]           [B,m4]...
  GPU-2                [F,m1]  [F,m2]   [F,m3]  [F,m4]   [B,m4]...
  GPU-3                        [F,m1]   [F,m2]  [F,m3]   [F,m4][B,m4]...

  F = 前向，B = 反向，m_i = 第 i 个 micro-batch

  Bubble（气泡）比例 = (p-1) / (m + p - 1)
  其中 p = GPU数量，m = micro-batch 数量
  当 m >> p 时，气泡比例趋近于 0
```

**GPipe 的缺陷**：为了支持反向传播，需要在前向结束后保存所有 micro-batch 的**激活值**，显存开销 = O(micro-batches × 层数)。

### 4.3  PipeDream（Microsoft，2019）

PipeDream 提出 **1F1B 调度策略**（One Forward One Backward），让前向和反向交替执行：

```
  1F1B 调度（4 GPU，8 micro-batch）

  稳定阶段（Steady State）：
  GPU-0  [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4][F8][B5][B6][B7][B8]
  GPU-1      [F1][F2][F3][F4][B1][F5][B2][F6][B3][F7][B4][F8][B5][B6][B7][B8]
  GPU-2          [F1][F2][F3][F4][B1]...
  GPU-3              [F1][F2][F3][F4][B1]...

  优点：峰值激活值从 O(m) 降至 O(p)（只需保存 in-flight 的 micro-batch）
  缺点：权重更新略有延迟（Weight Stashing 问题）
```

### 4.4  Interleaved Pipeline（Megatron-LM，2021）

Megatron-LM 进一步提出**交错式流水线**，让每个 GPU 持有多个不连续的"chunk"：

```
  普通流水线（每个 GPU 1 chunk）：
  GPU-0: Layer 0-3
  GPU-1: Layer 4-7
  GPU-2: Layer 8-11
  GPU-3: Layer 12-15

  交错式流水线（每个 GPU 2 chunks）：
  GPU-0: Layer 0-1  +  Layer 8-9
  GPU-1: Layer 2-3  +  Layer 10-11
  GPU-2: Layer 4-5  +  Layer 12-13
  GPU-3: Layer 6-7  +  Layer 14-15

  气泡比例降低约 v 倍（v = 每 GPU chunk 数）
  代价：通信次数增加 v 倍
```

### 4.5  流水线并行的关键超参

```python
# Megatron-LM 配置示例
--pipeline-model-parallel-size 4   # 流水线并行度（GPU 按层切分组数）
--num-micro-batches 8              # micro-batch 数量（越大气泡越小）
--num-layers-per-virtual-pipeline-stage 2  # 交错式：每 GPU 每 chunk 的层数
```

| 策略 | 气泡比例 | 激活显存 | 通信次数 |
|------|---------|---------|---------|
| 朴素模型并行 | ~(N-1)/N | 低 | 少 |
| GPipe | (p-1)/(m+p-1) | 高 O(m) | 少 |
| 1F1B | (p-1)/(m+p-1) | 低 O(p) | 少 |
| 交错式 1F1B | (p-1)/(vm+p-1) | 低 O(p) | 多 v 倍 |

---

## 第五章  序列并行（Sequence Parallelism）

### 5.1  动机：长序列下的激活值瓶颈

在 Attention 计算中，激活值随序列长度 L 和批量大小 B 线性增长。当 L = 32K（长上下文场景），激活值可达数十 GB，单卡无法承受。

```
  Transformer 层激活值大小估算：
  ≈ B × L × H × (34 + 5 × A × L / H)

  其中 B=批大小, L=序列长, H=隐层维度, A=注意力头数
  L=32768, H=4096, A=32, B=1 时，≈ 32768 × 4096 × (34 + 5×32×32768/4096) ≈ ~200 GB/层
```

### 5.2  Megatron-LM 序列并行

在张量并行之外，对**序列维度**进行额外切分，处理非 Attention 算子（LayerNorm、Dropout）的激活值。

```
  序列并行 + 张量并行的配合示意

  输入 X (B, L, H)
       |
  SequenceParallel 切分：每个 GPU 处理 (B, L/tp, H)
       |
  ┌────┴────┐
  GPU-0     GPU-1           ← LayerNorm, Dropout 各处理 L/tp 段
  (L/2)     (L/2)
  └────┬────┘
       | All-Gather → (B, L, H)
       |
  张量并行 Attention（按 head 切分）
       |
       | All-Reduce → (B, L, H)
       |
  SequenceParallel 切分：(B, L/tp, H)
       |
  张量并行 MLP（列/行切分）
```

### 5.3  Ring Attention（2023）

Ring Attention 是序列并行的更进一步，专门针对**超长序列**（百万 token 级别）设计：

```
  Ring Attention 原理：

  将序列切分到多个 GPU，每个 GPU 只持有 Q_i（本地 Query 块）
  K, V 在 GPU 之间以 Ring 方式循环传递

  步骤：
  Step 1: GPU_i 用本地 Q_i 与本地 K_i, V_i 计算部分 Attention
  Step 2: GPU_i 将 K_i, V_i 传给 GPU_{i+1}，接收 K_{i-1}, V_{i-1}
  Step 3: GPU_i 用 Q_i 与收到的 K, V 更新 Attention 结果（增量 softmax）
  ...（循环 N 轮）

  通信与计算完全重叠，理论上可扩展到无限长序列
```

---

## 第六章  ZeRO 与混合并行 ★★

### 6.1  优化器状态的显存问题

Adam 优化器在混合精度训练下，每个参数需要存储：

```
  每个参数的显存占用（Adam + 混合精度）：

  FP16 参数         : 2 bytes
  FP16 梯度         : 2 bytes
  FP32 参数副本     : 4 bytes   ← 用于精度累积
  FP32 一阶矩 m     : 4 bytes   ← Adam 状态
  FP32 二阶矩 v     : 4 bytes   ← Adam 状态
  ─────────────────────────────
  合计              : 16 bytes/参数

  7B 模型 = 7B × 16 bytes = 112 GB
  70B 模型= 70B × 16 bytes = 1120 GB（需要 14 张 A100 80GB）
```

### 6.2  ZeRO（Zero Redundancy Optimizer）

DeepSpeed ZeRO 的核心思想：**消除数据并行中的冗余存储**，将参数、梯度、优化器状态分片到所有数据并行的 GPU 上。

```
  ZeRO 三个阶段的显存对比

  ┌──────────────────────────────────────────────────────┐
  │  数据并行（普通）：每张 GPU 存储全量 16 bytes/param     │
  │  GPU-0: [Param][Grad][OS]  完整副本                   │
  │  GPU-1: [Param][Grad][OS]  完整副本（冗余！）          │
  │  GPU-N: [Param][Grad][OS]  完整副本（冗余！）          │
  ├──────────────────────────────────────────────────────┤
  │  ZeRO-1：分片优化器状态（OS）                          │
  │  GPU-0: [Param][Grad][OS shard 0]                    │
  │  GPU-1: [Param][Grad][OS shard 1]                    │
  │  节省：优化器状态 (4+4+4=12 bytes) 按 N 分片           │
  │  每 GPU：2+2+12/N bytes ≈ 4+12/N                     │
  ├──────────────────────────────────────────────────────┤
  │  ZeRO-2：分片优化器状态 + 梯度（Grad）                 │
  │  GPU-0: [Param][Grad shard 0][OS shard 0]            │
  │  GPU-1: [Param][Grad shard 1][OS shard 1]            │
  │  每 GPU：2+(2+12)/N bytes ≈ 2+14/N                   │
  ├──────────────────────────────────────────────────────┤
  │  ZeRO-3：分片参数 + 梯度 + 优化器状态（全量分片）       │
  │  GPU-0: [Param shard 0][Grad shard 0][OS shard 0]    │
  │  GPU-1: [Param shard 1][Grad shard 1][OS shard 1]    │
  │  每 GPU：16/N bytes → 理论上可无限扩展                 │
  └──────────────────────────────────────────────────────┘
```

### 6.3  ZeRO 通信分析

```
  通信量对比（数据并行 N 卡，参数量 Ψ）：

  标准 DDP    All-Reduce 梯度：2Ψ（一次 Scatter-Reduce + 一次 All-Gather）
  ZeRO-1      梯度 All-Reduce 后本地 OS 更新：2Ψ（同 DDP）
  ZeRO-2      Reduce-Scatter 梯度：Ψ
  ZeRO-3      All-Gather 参数（前向+反向各1次）+ Reduce-Scatter 梯度：3Ψ

  ZeRO-3 通信量是 ZeRO-2 的 1.5 倍，但解决了参数显存问题
```

### 6.4  DeepSpeed ZeRO 使用示例

```python
import deepspeed

ds_config = {
    "zero_optimization": {
        "stage": 3,                          # ZeRO-3
        "overlap_comm": True,               # 通信与计算重叠
        "contiguous_gradients": True,
        "reduce_scatter": True,
        "allgather_partitions": True,
        "allgather_bucket_size": 5e8,
    },
    "fp16": { "enabled": True },
    "train_micro_batch_size_per_gpu": 2,
    "gradient_accumulation_steps": 4,
}

model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    config=ds_config,
    model_parameters=model.parameters()
)
```

### 6.5  ZeRO++ 与 ZeRO-Infinity

```
  ZeRO++ (2023)：跨节点通信量进一步压缩
  ├── qgZ：梯度量化（int8/int4），减少 Reduce-Scatter 通信量
  ├── hpZ：参数层次化分组（节点内 All-Gather，跨节点按需获取）
  └── qwZ：权重量化缓存，减少 All-Gather 通信量
  
  总体通信量降低约 4 倍

  ZeRO-Infinity：将参数卸载到 CPU/NVMe SSD
  └── 可训练参数量 = GPU显存 + CPU内存 + SSD容量
  └── 但计算速度受 PCIe 带宽限制
```

### 6.6  3D 并行（混合并行）★

大规模训练通常同时使用多种并行策略，称为 **3D 并行**：

```
  3D 并行：TP × PP × DP

  示例：64 GPU，TP=8，PP=4，DP=2

  拓扑结构：
  ┌─────────────── 数据并行（DP=2，两个完整模型副本）──────────────┐
  │                                                             │
  │  ┌──────── 流水线并行（PP=4，4 级流水线）────────┐            │
  │  │  ┌──张量并行（TP=8，8 GPU 共享同一层)──┐      │            │
  │  │  │  GPU0  GPU1  GPU2 ... GPU7         │     │            │
  │  │  └──────────────────────────────────── ┘    │            │
  │  │  ┌────────────────────────────────────┐     │            │
  │  │  │  GPU8  GPU9 ... GPU15              │     │            │
  │  │  └────────────────────────────────────┘     │            │
  │  │  ...（共 4 级，每级 8 个 GPU）                │           │
  │  └─────────────────────────────────────────────┘            │
  │                                                             │
  │  + 另一个相同的 DP 副本（GPU 32-63）                          │
  └─────────────────────────────────────────────────────────────┘

  通信层次：
    TP All-Reduce  → NVLink（同节点内，带宽 ~600 GB/s）
    PP 点对点通信  → NVLink 或 InfiniBand
    DP All-Reduce  → InfiniBand（跨节点，带宽 ~200 GB/s）
```

---

## 第七章  混合专家（Mixture of Experts）★★

### 7.1  背景与动机

MoE 是一种**稀疏激活**的模型架构，打破了"参数量 = 计算量"的束缚：**增加参数量，但保持每次前向计算的 FLOPs 不变**。

```
  Dense 模型 vs MoE 模型对比：

  Dense Transformer MLP：
    每个 token 经过同一个 FFN（全部参数激活）
    计算量 = 参数量 × 激活率（100%）

  MoE Transformer：
    FFN 被替换为 E 个专家（Expert FFN），每次只激活 top-k 个
    计算量 = 参数量 × (k/E)    ← 激活率大幅降低
```

### 7.2  MoE 架构详解

```
  MoE 层结构

  输入 token x
       │
       ▼
  ┌─────────────┐
  │   Router    │    ← 轻量线性层 + Softmax，输出每个专家的概率
  │ W_g ∈ R^{H×E}│
  └──────┬──────┘
         │ top-k 选择（例如 k=2）
         │
  ┌──────┴──────────────────────────────────┐
  │  Expert 1  Expert 2  Expert 3  Expert E │
  │  FFN_1     FFN_2     FFN_3    FFN_E     │
  └──────┬──────────────────────────────────┘
         │ 加权求和（按 router 分数）
         ▼
      输出 y = Σ g_i · Expert_i(x)
```

### 7.3  Expert 路由与负载均衡

路由（Routing）是 MoE 的核心难题：若 Router 不加约束，所有 token 会倾向于选择同几个专家，导致**专家坍塌（Expert Collapse）**。

```python
# 典型的 Top-2 路由实现
class Router(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        # x: (B*L, H)
        logits = self.gate(x)                           # (B*L, E)
        scores = F.softmax(logits, dim=-1)
        topk_scores, topk_indices = scores.topk(self.top_k, dim=-1)
        # 归一化 top-k 权重
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)
        return topk_scores, topk_indices

# 辅助损失（负载均衡）
def load_balance_loss(router_probs, expert_indices, num_experts):
    # 专家使用频率 f_i
    expert_mask = F.one_hot(expert_indices, num_experts).float()
    f_i = expert_mask.mean(dim=[0, 1])           # 每个专家被选中的比例
    # 路由器对每个专家的平均概率 P_i
    P_i = router_probs.mean(dim=[0, 1])
    # 辅助损失：鼓励均匀分配
    loss = num_experts * (f_i * P_i).sum()
    return loss
```

### 7.4  Expert 并行（EP）

MoE 的 Expert 天然适合并行：将不同 Expert 分配到不同 GPU，每个 GPU 只存储和计算自己负责的 Expert。

```
  Expert 并行通信模式（All-to-All）

  Token 需要被路由到对应 Expert 所在的 GPU：

  GPU-0 持有 token t0,t1,t2,t3  →  各 token 被路由到不同 GPU
       ↓  All-to-All（dispatch）
  GPU-0 处理路由给 Expert 1,2 的所有 token
  GPU-1 处理路由给 Expert 3,4 的所有 token
  ...
       ↓  All-to-All（combine）
  GPU-0 收回 t0,t1,t2,t3 的计算结果

  通信量 = 2 次 All-to-All × batch_size × hidden_size
```

### 7.5  代表性 MoE 模型

| 模型 | 总参数 | 激活参数 | 专家数 | Top-K |
|------|--------|---------|--------|-------|
| Switch Transformer (Google, 2021) | ~1.6T | ~7B | 2048 | 1 |
| GLaM (Google, 2021) | 1.2T | 143B | 64 | 2 |
| Mixtral 8×7B (Mistral, 2023) | 47B | 13B | 8 | 2 |
| DeepSeek-V2 (2024) | 236B | 21B | 160 | 6 |
| Qwen1.5-MoE (2024) | 14.3B | 2.7B | 60 | 4 |

---

## 第八章  推理加速：量化技术 ★★

量化是推理阶段最重要的压缩手段，通过降低数值精度减少显存和计算开销。

### 8.1  量化基础概念

```
  浮点数精度对比：

  FP32   [符号1位][指数8位][尾数23位]   范围：±3.4×10^38   精度高
  FP16   [符号1位][指数5位][尾数10位]   范围：±65504       易溢出
  BF16   [符号1位][指数8位][尾数7位]    范围：±3.4×10^38   训练友好
  INT8   [符号1位][数值7位]             范围：-128~127      推理常用
  INT4   [符号1位][数值3位]             范围：-8~7          极端压缩

  精度 vs 显存（7B 模型参数）：
  FP32 → 28 GB
  FP16 → 14 GB
  INT8 →  7 GB
  INT4 →  3.5 GB
```

### 8.2  量化数学原理

将浮点数 x 映射到整数的基本公式：

```
  线性量化：
    x_q = round(x / scale + zero_point)
    x_dequant = (x_q - zero_point) × scale

  其中：
    scale = (x_max - x_min) / (q_max - q_min)
    zero_point = round(q_min - x_min / scale)

  示例（将 [-1.2, 0.8] 映射到 INT8 [-128, 127]）：
    scale = (0.8-(-1.2)) / (127-(-128)) = 2.0/255 ≈ 0.00784
    zero_point = round(-128 - (-1.2)/0.00784) = round(-128+153) = 25
```

### 8.3  PTQ 与 QAT

```
  训练后量化（PTQ, Post-Training Quantization）：
  ┌───────────────────────────────────────────────────────┐
  │  已训练的 FP32 模型                                    │
  │       ↓ 校准数据（少量样本，约 128-512 条）            │
  │  统计激活值分布 → 确定量化参数（scale/zero_point）     │
  │       ↓                                               │
  │  量化后的 INT8/INT4 模型                               │
  │  优点：无需重新训练，速度快                            │
  │  缺点：精度损失，尤其 INT4 下较明显                    │
  └───────────────────────────────────────────────────────┘

  量化感知训练（QAT, Quantization-Aware Training）：
  ┌───────────────────────────────────────────────────────┐
  │  在训练时模拟量化误差（Fake Quantization）             │
  │  前向：FP32 → 模拟量化 → FP32（模拟低精度效果）        │
  │  反向：直接传梯度（直通估计器 STE）                    │
  │  优点：精度损失更小                                   │
  │  缺点：需要重新训练，成本高                            │
  └───────────────────────────────────────────────────────┘
```

### 8.4  GPTQ（2022）★

GPTQ 是大模型 PTQ 的里程碑工作，基于二阶信息实现高精度 INT4 量化。

```
  GPTQ 核心：逐层最优量化

  目标：找到量化权重 W_q，使输出误差最小
    min ||WX - W_q X||²_F

  算法：
  1. 计算 Hessian 矩阵 H = 2XX^T（反映各权重的重要性）
  2. 逐列（或逐块）量化：
     a. 对第 j 列权重 w_j 进行量化得到 q_j
     b. 计算误差：e_j = (w_j - q_j) / H_jj
     c. 将误差补偿到后续未量化的列：w_{k>j} -= e_j × H_{jk}/H_{jj}
  3. 重复直到所有列被量化

  实际效果（LLaMA-2 7B）：
  FP16 基准 PPL: 5.47
  GPTQ INT4:     5.85 (误差仅 +0.38)
```

### 8.5  AWQ（2023）★

AWQ（Activation-aware Weight Quantization）发现并非所有权重同等重要，少数"显著权重"对输出影响更大。

```
  AWQ 核心思想：

  观察：激活值（Activation）中，约 1% 的通道幅值极大
        这些"显著通道"对应的权重量化误差会被放大

  解决方案：对显著权重进行缩放保护（而非跳过量化）

  数学等价变换：
    原始：Y = Wx
    变换：Y = (W · diag(s)^{-1}) · (diag(s) · x) = W' · x'
    
  s（缩放因子）由激活值统计决定：
    s_j = max|x_j|^α  （α 通常为 0.5）
  
  W'（缩放后的权重）量化误差更小，因为幅值更均匀
  实际推理中：将 s 吸收进临近的 LayerNorm 参数，无额外计算开销
```

### 8.6  FP8 训练（H100 时代）

NVIDIA H100 引入硬件 FP8（E4M3/E5M2），支持以 FP8 进行矩阵乘法：

```
  FP8 混合精度训练流程：

  FP8 E4M3：[符号1][指数4][尾数3]  用于前向权重/激活
  FP8 E5M2：[符号1][指数5][尾数2]  用于反向梯度（更大范围）

  前向计算：
    W (FP8 E4M3) × A (FP8 E4M3) → 输出 (FP16/BF16)
  
  反向计算：
    梯度 (FP8 E5M2) × W (FP8 E4M3) → 权重梯度 (FP16/BF16)
  
  关键技术：
  ├── 动态量化缩放（每次矩阵乘前统计 amax）
  ├── 延迟缩放（Delayed Scaling）：使用历史 amax，减少同步
  └── 高精度优化器状态（仍用 FP32）

  收益：相比 BF16 训练，MFU 提升约 1.5-2x（H100 上）
```

---

## 第九章  推理加速：KV Cache 与注意力优化 ★★

### 9.1  自回归解码的性能瓶颈

LLM 推理采用自回归（Autoregressive）方式，每次只生成一个 token，效率极低：

```
  自回归解码过程：

  Prompt: "The weather today is"    ← Prefill 阶段（并行处理所有 token）
  生成第1个 token: "sunny"          ← Decode 阶段
  生成第2个 token: "and"            ← 每步需要处理历史上下文
  生成第3个 token: "warm"
  ...

  Decode 阶段每步只生成 1 个 token，计算效率极低（矩阵-向量乘法 vs 矩阵-矩阵乘法）
  这使 GPU 的并行算力严重浪费（compute-bound → memory-bound）
```

### 9.2  KV Cache ★

为避免重复计算历史 token 的 Key/Value，将其缓存在显存中：

```
  无 KV Cache（朴素实现）：
  
  生成第 t 个 token 时：
    计算 Q_t, K_{1..t}, V_{1..t}   ← 重复计算了 K_{1..t-1}, V_{1..t-1}
    Attention(Q_t, K_{1..t}, V_{1..t})

  ─────────────────────────────────────────────────────
  有 KV Cache：
  
  每步只计算当前 token 的 K_t, V_t，然后追加到 Cache：
    KV_cache = [K_1,...,K_{t-1}, K_t] 
               [V_1,...,V_{t-1}, V_t]
    Attention(Q_t, KV_cache)
  
  计算量从 O(t²) 降至 O(t)（每步）
  
  KV Cache 显存占用：
    = num_layers × 2 × B × L × num_heads × head_dim × dtype_bytes
    LLaMA-2 7B, L=2048, B=1, FP16:
    = 32 × 2 × 1 × 2048 × 32 × 128 × 2 ≈ 1 GB
    L=32768（32K 上下文）→ ~16 GB
```

### 9.3  Multi-Query Attention 与 GQA ★

为压缩 KV Cache 大小，出现了共享 KV 的注意力变体：

```
  MHA / GQA / MQA 对比：

  MHA（Multi-Head Attention，标准）：
    Q 头数 = K 头数 = V 头数 = H
    KV Cache 大小 = H × head_dim

  MQA（Multi-Query Attention，Google 2019）：
    Q 头数 = H，K/V 头数 = 1（所有 Q 共享一组 KV）
    KV Cache 大小 = 1 × head_dim（节省 H 倍）
    代价：精度略有损失

  GQA（Grouped Query Attention，Google 2023）★
    Q 头数 = H，K/V 头数 = G（G < H，每 H/G 个 Q 共享一组 KV）
    KV Cache 大小 = G × head_dim（节省 H/G 倍）
    LLaMA-2 70B 使用 GQA（H=64, G=8，节省 8 倍）

  图示（H=8, G=2 的 GQA）：

  Q: [h0 h1 h2 h3 | h4 h5 h6 h7]
                  |              |
  K: [      KV_group_0      ] [      KV_group_1      ]
  V: [      KV_group_0      ] [      KV_group_1      ]
```

### 9.4  FlashAttention ★★

> **参考论文**：FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness（Dao et al., 2022）

FlashAttention 解决了标准 Attention 的 IO 瓶颈：标准实现需要将 N×N 的注意力矩阵写回 HBM（高带宽存储），而 FlashAttention **在 SRAM 中完成所有计算**。

```
  标准 Attention 的 IO 瓶颈：

  HBM（高带宽存储）         SRAM（片上高速缓存）
  ┌──────────────────┐       ┌──────────────┐
  │  Q, K, V (N×d)  │──────>│    计算单元   │
  │  S = QK^T (N×N) │<──────│    S,P        │  ← N×N 矩阵需写到 HBM！
  │  P = softmax(S) │──────>│              │
  │  O = PV (N×d)   │<──────│    O          │
  └──────────────────┘       └──────────────┘
  HBM 读写：O(N²)

  ─────────────────────────────────────────────────────────
  FlashAttention Tiling：

  将 Q,K,V 切成小块（Tile），在 SRAM 中逐块计算，
  利用增量 softmax 技巧（log-sum-exp）合并结果：

  for each block Q_i:
    for each block K_j, V_j:
      S_ij = Q_i K_j^T                  ← 在 SRAM 中
      P_ij = softmax(S_ij)  (增量更新)   ← 不需要存整个 N×N
      O_i += P_ij V_j
  
  HBM 读写：O(N × d)（线性！）
  
  性能提升：
    FlashAttention-1 (2022)：比标准 2-4x 加速，显存 O(N) vs O(N²)
    FlashAttention-2 (2023)：进一步优化并行，A100 达到约 72% MFU
    FlashAttention-3 (2024)：利用 H100 异步特性，约 75%+ MFU
```

### 9.5  PagedAttention（vLLM，2023）★

KV Cache 的传统实现需要预分配**连续显存**，导致严重的内存碎片和浪费。PagedAttention 借鉴操作系统虚拟内存分页思想：

```
  传统 KV Cache 问题：

  请求1（长度100）：预分配 KV[0:2048]   → 浪费 1948 个 slot
  请求2（长度500）：预分配 KV[2048:4096] → 浪费 1548 个 slot
  
  内存利用率极低（通常 20-40%）

  ─────────────────────────────────────────────────────────
  PagedAttention：

  KV Cache 被划分为固定大小的 Block（例如 16 个 token/block）
  逻辑块通过 Block Table 映射到物理块

  请求1 逻辑块: [Block 0][Block 1]...
                    ↓         ↓
  物理内存:    [Phys 3] [Phys 7]  ← 非连续物理地址！

  优点：
  ├── 内存利用率提升至 ~96%（几乎无碎片）
  ├── 支持 Copy-on-Write（beam search 共享 KV 块）
  └── 使批量推理（Continuous Batching）成为可能
```

---

## 第十章  推理加速：推测解码与连续批处理 ★

### 10.1  推测解码（Speculative Decoding）

自回归解码每步只生成一个 token，GPU 利用率极低。推测解码用一个**小型草稿模型**先快速生成多个候选 token，再用大模型**并行验证**。

```
  推测解码流程：

  草稿阶段（Draft Phase）：
    草稿模型（例如 7B）自回归生成 K 个候选 token
    t1, t2, t3, t4, t5  ← 速度快（小模型）

  验证阶段（Verify Phase）：
    将 prompt + t1~t5 一次性输入目标模型（例如 70B）
    目标模型并行计算所有位置的概率

    并行验证（一次前向 = K+1 步解码的工作量）：
    位置 i: 目标模型概率 q_i，草稿模型概率 p_i
    接受条件: min(1, q_i/p_i) > random()   ← 拒绝采样

  效果：
  ├── 若草稿模型预测准确，一次前向接受多个 token
  ├── 输出分布与纯目标模型完全等价（无精度损失）
  └── 实测提速 2-3x（草稿接受率 ~70-80% 时）

  变种：
  ├── Self-speculative（用目标模型的浅层做草稿）
  ├── Medusa（多头并行预测，无需额外草稿模型）
  └── EAGLE（用草稿头预测特征，接受率更高）
```

### 10.2  连续批处理（Continuous Batching）

传统静态批处理（Static Batching）将一批请求等到**最长序列完成**才释放资源，效率低。

```
  静态批处理：

  请求 A：生成 100 token ████████████████████
  请求 B：生成 50 token  ██████████░░░░░░░░░░  ← 50步空等 A
  请求 C：生成 200 token ████████████████████████████████████████

  时间轴 ──────────────────────────────────────────────>
  (整个批次等 C 完成才处理下一批，B 完成后 GPU 也不能复用)

  ─────────────────────────────────────────────────────────
  连续批处理（vLLM / TensorRT-LLM）：

  一旦某个请求生成完毕，立即插入新请求：

  时间轴:   [A,B,C 生成中...]
                [B 完成] → 立即插入请求 D
            [A,C,D 生成中...]
                    [A 完成] → 立即插入请求 E
            [C,D,E 生成中...]

  关键实现（Iteration-level Scheduling）：
  ├── 每个 decode step 后重新调度
  ├── 区分 Prefill 请求和 Decode 请求（Chunked Prefill）
  └── PagedAttention 使 KV Cache 可按块动态分配

  效果：吞吐量提升 5-10x（与静态批处理相比）
```

### 10.3  推理引擎核心技术对比

| 技术 | 解决的问题 | 适用场景 | 典型实现 |
|------|-----------|---------|---------|
| KV Cache | 避免重复计算历史 KV | 所有自回归场景 | 标配 |
| GQA/MQA | 减少 KV Cache 显存 | 长序列、大批量 | LLaMA-2/3 |
| FlashAttention | 减少注意力 IO 开销 | 训练+推理 | vLLM, HF |
| PagedAttention | 消除 KV Cache 碎片 | 高并发服务 | vLLM |
| 连续批处理 | 提升 GPU 利用率 | 在线推理服务 | vLLM, TRT-LLM |
| 推测解码 | 提升单请求速度 | 低并发/实时场景 | Medusa, EAGLE |
| INT8/INT4 量化 | 减少显存和加速计算 | 部署/资源受限 | GPTQ, AWQ |

---

## 第十一章  主流框架对比与选型

### 11.1  训练框架

| 框架 | 开发者 | 主要特性 | 适用规模 |
|------|--------|---------|---------|
| **Megatron-LM** | NVIDIA | TP+PP+DP 3D 并行，最优 MFU，FP8 训练 | 超大规模（>10B） |
| **DeepSpeed** | Microsoft | ZeRO 系列，CPU/NVMe Offload，易用性好 | 中大规模 |
| **FSDP** | Meta/PyTorch | ZeRO-3 类似，PyTorch 原生，易于集成 | 中等规模 |
| **Megatron-DeepSpeed** | 联合 | 融合两者优点 | 超大规模 |
| **Nanotron** | HuggingFace | 轻量，代码简洁，适合研究 | 中等规模 |

```
  框架选择决策树：

  参数量 < 10B?
  ├── 是 → 单机多卡 DDP (PyTorch FSDP) 即可
  └── 否 → 需要模型并行？
            ├── 预算有限/灵活性优先 → DeepSpeed ZeRO-3
            └── 追求极致 MFU → Megatron-LM (需要 NVIDIA 集群)
```

### 11.2  推理框架

| 框架 | 开发者 | 主要特性 | 适用场景 |
|------|--------|---------|---------|
| **vLLM** | UC Berkeley | PagedAttention，连续批处理，最高吞吐 | 在线服务高并发 |
| **TensorRT-LLM** | NVIDIA | INT8/INT4 量化，FP8，H100 优化 | NVIDIA GPU 部署 |
| **llama.cpp** | Georgi Gerganov | CPU/Metal/CUDA，极致量化 | 边缘/本地部署 |
| **SGLang** | LMSYS | 结构化生成，RadixAttention | 复杂推理任务 |
| **Ollama** | Ollama | 基于 llama.cpp，易用 | 个人/开发环境 |

---

## 第十二章  总结与技术路线图

### 12.1  各技术适用场景速查

```
  问题类型                   推荐技术
  ──────────────────────────────────────────────────────
  显存不足（单卡放不下模型）
  ├── 参数 <70B             → ZeRO-3 + 梯度检查点
  ├── 参数 <200B            → TP(8) + PP(4) + DP
  └── 参数 >200B            → 3D 并行 + ZeRO + MoE

  训练速度不够
  ├── 通信瓶颈              → 减小 DP size，增大 TP/PP
  ├── 计算瓶颈              → FlashAttention + FP8
  └── 数据瓶颈              → 异步数据加载 + 预处理

  推理延迟高（单请求）
  ├── 显存带宽瓶颈          → 量化(INT4/AWQ) + 推测解码
  └── 算力瓶颈              → 增大批大小（更换 memory-bound）

  推理吞吐低（高并发服务）
  ├── GPU 利用率低           → 连续批处理 + PagedAttention
  ├── KV Cache 显存不足     → GQA + 量化 KV Cache
  └── 长序列推理慢          → FlashAttention + 分块 Prefill
```

### 12.2  完整技术栈鸟瞰图

```
  大模型训练与推理技术全景

  ┌─────────────────────────────────────────────────────┐
  │                     应用层                          │
  │  在线服务(vLLM) / 批量推理(TRT-LLM) / 本地(Ollama) │
  ├─────────────────────────────────────────────────────┤
  │                   推理优化层                        │
  │  推测解码 │ 连续批处理 │ PagedAttention │ 量化       │
  │  KV Cache │ FlashAttn │ GQA/MQA       │ FP8        │
  ├─────────────────────────────────────────────────────┤
  │                   训练并行层                        │
  │  数据并行(DDP) │ 张量并行(TP) │ 流水线并行(PP)      │
  │  序列并行(SP)  │ Expert并行  │ ZeRO-1/2/3          │
  ├─────────────────────────────────────────────────────┤
  │                   模型架构层                        │
  │  Transformer │ MoE │ GQA │ RoPE │ FlashAttention   │
  ├─────────────────────────────────────────────────────┤
  │                   硬件层                           │
  │  GPU(H100/A100) │ NVLink │ InfiniBand │ NVMe       │
  └─────────────────────────────────────────────────────┘
```

### 12.3  学习路径建议

按照由浅入深的顺序，建议依次掌握以下内容：

**第一阶段（基础）**
1. PyTorch DDP + DistributedSampler 实战
2. 梯度累积与混合精度训练（AMP）
3. 激活值检查点（Gradient Checkpointing）

**第二阶段（进阶）**
4. DeepSpeed ZeRO-2/3 配置与调优
5. Megatron-LM 张量并行原理
6. FlashAttention 原理与使用

**第三阶段（专家）**
7. 3D 并行（TP+PP+DP）调优
8. GPTQ/AWQ 量化实战
9. vLLM 服务部署与 PagedAttention
10. MoE 架构设计与 Expert 并行

**第四阶段（前沿）**
11. FP8 训练（Transformer Engine）
12. 推测解码（Medusa/EAGLE）
13. Ring Attention 超长序列
14. ZeRO++ 跨节点优化

---

### 参考资料

- [Megatron-LM 论文](https://arxiv.org/abs/2104.04473) - 张量并行经典
- [ZeRO 论文](https://arxiv.org/abs/1910.02054) - DeepSpeed 优化器分片
- [FlashAttention 论文](https://arxiv.org/abs/2205.14135) - IO 感知注意力
- [FlashAttention-2](https://arxiv.org/abs/2307.08691) - 进一步优化
- [PagedAttention/vLLM](https://arxiv.org/abs/2309.06180) - 高效推理服务
- [AWQ 论文](https://arxiv.org/abs/2306.00978) - 激活感知量化
- [GPTQ 论文](https://arxiv.org/abs/2210.17323) - 二阶量化
- [GPipe](https://arxiv.org/abs/1811.06965) - 流水线并行
- [Mixtral MoE](https://arxiv.org/abs/2401.04088) - 开源 MoE 代表
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434) - 超大 MoE 训练实践
- [Hugging Face Transformers 文档](https://huggingface.co/docs/transformers)
- [DeepSpeed 文档](https://www.deepspeed.ai/docs/)
- [vLLM 文档](https://docs.vllm.ai/)
