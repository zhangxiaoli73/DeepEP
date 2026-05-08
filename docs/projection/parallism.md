# Communication Projection (implementation-agnostic)

目标：给定 Model / Platform / Runtime 配置，估算不同并行策略下（仅 inference）单层通信量与单层通信时间。

## 1. Inputs

### 1.1 Model configuration

- hidden_size: `H`
- intermediate_size: `I`
- num_heads: `Nh`
- num_kv_heads: `Nkv`
- head_dim: `Dh`
- num_layers: `L`
- num_experts: `E`
- topk: `topK`
- shared_experts: `Es` (可选，默认 0)

说明：`vocab_size` 主要影响 embedding 和 lm_head，不是主干每层通信瓶颈，可先忽略。

### 1.2 Platform configuration

- compute peak (fp16): `F_peak` (TFLOPS)
- device memory bandwidth: `BW_mem` (GB/s)
- cross-device bandwidth (intra-node): `BW_link` (GB/s)

说明：这里 `BW_link` 视为已经包含现实折损后的有效带宽。

### 1.3 Runtime configuration

- global batch size: `B`
- sequence length: `S`
- TP degree: `P_t`
- DP degree: `P_d`
- EP degree: `P_e`
- data type bytes: `b` (bf16/fp16 通常 `b=2`)

## 2. Baseline formulas

先用最常见的 ring 近似。为保证 VS Code 与 Chrome 显示一致，本文统一使用纯文本公式。

### 2.1 AllReduce

对 payload `X` bytes：

```text
V_AR(X, P) ~= 2 * (P - 1) / P * X
```

```text
t_AR(X, P) ~= V_AR(X, P) / BW_link
```

### 2.2 AllGather

```text
V_AG(X, P) ~= (P - 1) / P * X
```

### 2.3 ReduceScatter

```text
V_RS(X, P) ~= (P - 1) / P * X
```

### 2.4 AllToAll

粗略估计：

```text
V_A2A(X, P) ~= (P - 1) / P * X
```

其中 `X` 是本 rank 发送总量；实际还需乘以分块和负载不均开销系数 `alpha_a2a >= 1`。

## 3. Per-layer communication model

以下只关注 Transformer 主干与 MoE 关键通信，不绑定具体实现。

### 3.1 TP 相关通信（Attention + MLP）

在 tensor parallel 下，每层通常出现 2 次主通信（可视作 AR，或 RS+AG 组合）：

```text
V_tp_layer ~= 4 * (P_t - 1) / P_t * (B * S / P_d) * H * b
```

### 3.2 EP 相关通信（MoE dispatch/combine）

MoE 中每个 token 路由到 `topK` 个 expert。忽略索引和权重元数据时，dispatch 与 combine 分开计算：

```text
V_ep_dispatch_layer ~= (P_e - 1) / P_e * (B * S / P_d) * topK * H * b
```

```text
V_ep_combine_layer ~= (P_e - 1) / P_e * (B * S / P_d) * topK * H * b
```

```text
V_ep_layer ~= V_ep_dispatch_layer + V_ep_combine_layer
```

### 3.3 Attention Layer 计算量（全局 B*S 相同）

Attention Layer 包括：Q/K/V 投影、Attention kernel (QK^T 后再乘 V)、输出投影。

在保持 global batch_size × sequence_length 相同的前提下，不同并行模式的每 rank 计算量对比：

| 并行模式 | Attention Layer FLOPS | 简化形式 |
|---------|---------------------|---------|
| **TP only** | `(B*S)*(H/P_t)*(4*H + 2*S)` | `(B*S)*H*(4 + 2*S/H) / P_t` |
| **DP only** | `((B*S)/P_d)*H*(4*H + 2*(S/P_d))` | `(B*S)*H*4 / P_d + 2*(B*S)*S / P_d^2` |
| **TP + DP** | `((B*S)/P_d)*(H/P_t)*(4*H + 2*(S/P_d))` | 上式再除以 P_t |
| **TP + EP** | `(B*S)*(H/P_t)*(4*H + 2*S)` | 同 TP only |

**关键洞察：**
- **TP 影响：** 线性减少 `1/P_t`，因为 hidden 被分片
- **DP 影响：** 投影部分减少 `1/P_d`；Attention kernel 中 seq² 项减少 `1/P_d^2`
- 当 `S >> H` 时，DP 的 `1/P_d^2` 收益更大
- **EP 不影响 Attention 计算**（Attention 是 non-expert 层的核心）

## 4. Projection by parallelism mode

## 4.1 TP only

条件：`P_t = N, P_d = 1, P_e = 1`

每层通信量（近似）：

```text
V_layer ~= 4 * (P_t - 1) / P_t * B * S * H * b
```

每层通信时间：

```text
t_layer ~= (4 * (P_t - 1) / P_t * B * S * H * b) / BW_link
```

MLP 层每 rank 计算量（单位：FLOPS）：

```text
FLOPS_mlp ~= 4 * B * S * topK * (H / P_t) * I
```

TP 会将 hidden 分片为 `H/P_t`。每个 token 经过 topK 个 experts，总计 `B*S*topK` 个 token-expert 对。

## 4.2 EP only

条件：`P_t = 1, P_d = 1, P_e = N`

每层通信量（MoE 层，拆分计算）：

```text
V_layer_dispatch ~= (P_e - 1) / P_e * B * S * topK * H * b
```

```text
V_layer_combine ~= (P_e - 1) / P_e * B * S * topK * H * b
```

```text
V_layer ~= 2 * (P_e - 1) / P_e * B * S * topK * H * b
```

MLP 层每 rank 计算量（单位：FLOPS）：

```text
FLOPS_mlp ~= 4 * B * S * (topK / P_e) * H * I
```

EP 中，总计 `B*S*topK` 个 token-expert 对。按均匀分布假设，每个 device 平均处理 `topK/P_e` 个 experts。

## 4.3 DP + EP

条件：`P_t = 1, P_d = P_e = N`

在 inference 下，DP 只做样本切分，不引入梯度同步通信。
每层通信仍由 EP 的 dispatch/combine 主导：

```text
V_layer_dispatch ~= (P_e - 1) / P_e * (B * S / P_d) * topK * H * b
```

```text
V_layer_combine ~= (P_e - 1) / P_e * (B * S / P_d) * topK * H * b
```

```text
V_layer ~= 2 * (P_e - 1) / P_e * (B * S / P_d) * topK * H * b
```

MLP 层每 rank 计算量（单位：FLOPS）：

```text
FLOPS_mlp ~= 4 * B * S * (topK / (P_d * P_e)) * H * I
```

DP 切分 tokens 为 `B*S/P_d`，EP 按均匀分布，每个 device 平均处理 `topK/(P_d*P_e)` 个 experts。

若一个请求批次在 `P_d` 个 DP 组上平均分配，则每个 DP 组内按
`\frac{B \cdot S}{P_d}` 代入 `V_{ep,layer}` 公式即可。

## 4.4 TP + EP

条件：`P_d = 1, P_t = P_e = N`

总层通信量近似为 TP 与 EP 叠加：

```text
V_layer ~= 4 * (P_t - 1) / P_t * B * S * H * b
	+ 2 * (P_e - 1) / P_e * B * S * topK * H * b
```

但注意：TP 会改变每 rank 的 hidden 分片和 MoE 输入排布，实测常需引入修正系数 `beta_layout`：

```text
V_layer_corr ~= beta_layout * (
	4 * (P_t - 1) / P_t * B * S * H * b
	+ 2 * (P_e - 1) / P_e * B * S * topK * H * b
)
beta_layout in [0.9, 1.3]
```

MLP 层每 rank 计算量（单位：FLOPS）：

```text
FLOPS_mlp ~= 4 * B * S * topK * (H / P_t) * I / P_e
```

TP 将 hidden 分片为 `H/P_t`，EP 按均匀分布，每个 device 平均处理 `topK/P_e` 个 experts。

## 5. Practical projection procedure

1. 填入 `H, E, topK, B, S, P_t, P_d, P_e, b, BW_link`.
2. 按并行模式计算单层通信量：`V_layer`，必要时拆分为 `V_layer_dispatch` 与 `V_layer_combine`。
3. 用 `t_{layer} = V_{layer} / BW_link` 得到单层通信时间。
4. 做灵敏度分析：对 `topK`、`P_e`、`BW_link` 做 +/-20% 扰动。

## 6. Key sensitivities (结论导向)

- TP-only: 对 `H`、`B`、`S`、`P_t` 最敏感。
- EP-only: 对 `topK`、`H`、`B`、`S`、`P_e` 最敏感。
- DP+EP: `P_d` 增大可降低每个 DP 组内每 rank 的 token 通信；inference 不存在梯度同步成本。
- TP+EP: 常是通信最重模式，需重点看 `BW_link` 的限制。

## 7. Projection worksheet (快速参考表)

根据并行模式快速查询单层通信量和时间：

| 并行模式 | V_layer_dispatch (bytes) | V_layer_combine (bytes) | V_layer_total (bytes) | t_layer (ms) | FLOPS_mlp |
|---------|------------------------|----------------------|---------------------|-------------|----------|
| **TP only** | 0 | 0 | `4*(P_t-1)/P_t*B*S*H*b` | `4*(P_t-1)/P_t*B*S*H*b/BW_link` | `4*B*S*topK*(H/P_t)*I` |
| **EP only** | `(P_e-1)/P_e*B*S*topK*H*b` | `(P_e-1)/P_e*B*S*topK*H*b` | `2*(P_e-1)/P_e*B*S*topK*H*b` | `2*(P_e-1)/P_e*B*S*topK*H*b/BW_link` | `4*B*S*(topK/P_e)*H*I` |
| **DP + EP** | `(P_e-1)/P_e*(B*S/P_d)*topK*H*b` | `(P_e-1)/P_e*(B*S/P_d)*topK*H*b` | `2*(P_e-1)/P_e*(B*S/P_d)*topK*H*b` | `2*(P_e-1)/P_e*(B*S/P_d)*topK*H*b/BW_link` | `4*B*S*(topK/(P_d*P_e))*H*I` |
| **TP + EP** | `(P_e-1)/P_e*B*S*topK*H*b` | `(P_e-1)/P_e*B*S*topK*H*b` | `4*(P_t-1)/P_t*B*S*H*b + 2*(P_e-1)/P_e*B*S*topK*H*b` (乘 beta_layout) | 见上行总量除以 BW_link | `4*B*S*topK*(H/P_t)*I/P_e` |

**使用说明：**
1. 代入参数（`H, B, S, topK, P_t, P_d, P_e, I, b, BW_link, beta_layout`）后得到具体数值。
2. `t_layer` 的单位取决于 `BW_link` 的单位（若 BW_link 单位为 GB/s，则结果为 ms；若为 bytes/s，则为 s）。
3. 对于 TP+EP，实际通信量需乘以 `beta_layout` 系数（通常 0.9~1.3）。