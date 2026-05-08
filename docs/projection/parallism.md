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
- topk: `K`
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

先用最常见的 ring 近似：

### 2.1 AllReduce

对 payload `X` bytes：

$$
V_{AR}(X, P) \approx 2\frac{P-1}{P}X
$$

$$
t_{AR}(X, P) \approx \frac{V_{AR}(X, P)}{BW_{link}}
$$

### 2.2 AllGather

$$
V_{AG}(X, P) \approx \frac{P-1}{P}X
$$

### 2.3 ReduceScatter

$$
V_{RS}(X, P) \approx \frac{P-1}{P}X
$$

### 2.4 AllToAll

粗略估计：

$$
V_{A2A}(X, P) \approx \frac{P-1}{P}X
$$

其中 `X` 是本 rank 发送总量；实际还需乘以分块和负载不均开销系数 `alpha_a2a >= 1`。

## 3. Per-layer communication model

以下只关注 Transformer 主干与 MoE 关键通信，不绑定具体实现。

### 3.1 TP 相关通信（Attention + MLP）

在 tensor parallel 下，每层通常出现 2 次主通信（可视作 AR，或 RS+AG 组合）：

$$
V_{tp,layer} \approx 4 \cdot \frac{P_t-1}{P_t} \cdot \frac{B \cdot S}{P_d} \cdot H \cdot b
$$

### 3.2 EP 相关通信（MoE dispatch/combine）

MoE 中每个 token 路由到 `K` 个 expert。忽略索引和权重元数据时，dispatch 与 combine 分开计算：

$$
V_{ep,dispatch,layer} \approx \alpha_{a2a} \cdot \frac{P_e-1}{P_e} \cdot \frac{B \cdot S}{P_d} \cdot K \cdot H \cdot b
$$

$$
V_{ep,combine,layer} \approx \alpha_{a2a} \cdot \frac{P_e-1}{P_e} \cdot \frac{B \cdot S}{P_d} \cdot K \cdot H \cdot b
$$

$$
V_{ep,layer} \approx V_{ep,dispatch,layer} + V_{ep,combine,layer}
$$

可选元数据补偿（通常较小，可先忽略）：

- dispatch 侧可加
$$
\alpha_{a2a} \cdot \frac{P_e-1}{P_e} \cdot \frac{B \cdot S}{P_d} \cdot K \cdot (b_{idx}+b_w)
$$

- combine 侧可再加同样一项

其中 `b_idx` 典型 4 或 8 bytes，`b_w` 常为 2 或 4 bytes。

## 4. Projection by parallelism mode

## 4.1 TP only

条件：`P_t = N, P_d = 1, P_e = 1`

每层通信量（近似）：

$$
V_{layer} \approx V_{tp,layer}
$$

每层通信时间：

$$
t_{layer} \approx \frac{V_{layer}}{BW_{link}}
$$

## 4.2 EP only

条件：`P_t = 1, P_d = 1, P_e = N`

每层通信量（MoE 层，拆分计算）：

$$
V_{layer,dispatch} \approx V_{ep,dispatch,layer}
$$

$$
V_{layer,combine} \approx V_{ep,combine,layer}
$$

$$
V_{layer} \approx V_{layer,dispatch} + V_{layer,combine}
$$

## 4.3 DP + EP

条件：`P_t = 1, P_d = P_e = N`

在 inference 下，DP 只做样本切分，不引入梯度同步通信。
每层通信仍由 EP 的 dispatch/combine 主导：

$$
V_{layer,dispatch} \approx V_{ep,dispatch,layer}
$$

$$
V_{layer,combine} \approx V_{ep,combine,layer}
$$

$$
V_{layer} \approx V_{layer,dispatch} + V_{layer,combine}
$$

若一个请求批次在 `P_d` 个 DP 组上平均分配，则每个 DP 组内按
`\frac{B \cdot S}{P_d}` 代入 `V_{ep,layer}` 公式即可。

## 4.4 TP + EP

条件：`P_d = 1, P_t = P_e = N`

总层通信量近似为 TP 与 EP 叠加：

$$
V_{layer} \approx V_{tp,layer} + V_{ep,layer}
$$

但注意：TP 会改变每 rank 的 hidden 分片和 MoE 输入排布，实测常需引入修正系数 `beta_layout`：

$$
V_{layer}^{corr} \approx beta_{layout} \cdot V_{layer}, \quad beta_{layout} \in [0.9, 1.3]
$$

## 5. Practical projection procedure

1. 填入 `H, E, K, B, S, P_t, P_d, P_e, b, BW_link`。
2. 按并行模式计算 `V_tp,layer` 和/或 `V_{ep,dispatch,layer}`、`V_{ep,combine,layer}`。
3. 用 `t_{layer} = V_{layer} / BW_link` 得到单层通信时间。
4. 做灵敏度分析：对 `K`、`P_e`、`BW_link` 做 +/-20% 扰动。

## 6. Key sensitivities (结论导向)

- TP-only: 对 `H`、`B`、`S`、`P_t` 最敏感。
- EP-only: 对 `K`、`H`、`B`、`S`、`P_e` 最敏感，且受负载均衡影响很大（`alpha_a2a`）。
- DP+EP: `P_d` 增大可降低每个 DP 组内每 rank 的 token 通信；inference 不存在梯度同步成本。
- TP+EP: 常是通信最重模式，需重点看 `BW_link` 与 AllToAll 负载均衡（`alpha_a2a`）。

## 7. Optional worksheet (建议补充)

可在后续补一页表格：

- 行：`TP-only`, `EP-only`, `DP+EP`, `TP+EP`
- 列：`V_layer_dispatch (GB)`, `V_layer_combine (GB)`, `V_layer_total (GB)`, `t_layer (ms)`

这样可以快速做架构选型和容量评估。