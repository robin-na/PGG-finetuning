# Persona Stability in PGG Variants: A Shared-Space and Delexicalization Study

## Abstract
本报告研究一个核心问题：在同一 subject pool 的 PGG 数据中，不同机制标签（`PUNISHMENT`/`REWARD`/`COMMUNICATION`）是否改变了由文本 persona 推断出的聚类结构，还是仅改变了表面语言表达。我们在 3691 条已完成样本上比较三套协议：
1. 既有协议（split 后分别降维聚类）；
2. Step1：共享全局语义空间（global embedding + global UMAP）后再 split 聚类；
3. Step2：在共享空间框架下，去除机制词后再做 OpenAI embedding 与聚类。

结果显示：去机制词（Step2）相较 Step1 可显著降低跨 split 对齐距离（alignment）——`PUNISHMENT: 1.045→0.642`、`REWARD: 0.932→0.798`、`COMMUNICATION: 0.886→0.643`。同时 silhouette 普遍下降，说明机制词既是“跨机制差异”的来源，也确实提高了簇可分性。总体结论：persona 结构在机制变化下具有中等稳定性，且相当部分不稳定来自机制词汇本身而非更深层行为结构。

---

## 1. Problem Statement
我们要回答三个问题：
- `RQ1`：当 WITH/WITHOUT 某机制标签时，persona clusters 在几何上是否仍可对齐（stability）？
- `RQ2`：这种差异是否主要由机制词（punish/reward/chat 等）驱动？
- `RQ3`：结论在重采样下是否稳健（bootstrap CI）？

---

## 2. Data and Setting
- 数据：`summary_gpt51_learn.jsonl`
- 使用样本：`n=3691`（`game_finished=True`）
- split 标签：`PUNISHMENT`, `REWARD`, `COMMUNICATION`
- split 规模（旧实验同本次）：
  - PUNISHMENT: WITH 1812 / WITHOUT 1879
  - REWARD: WITH 1815 / WITHOUT 1876
  - COMMUNICATION: WITH 1870 / WITHOUT 1821

输入文本是 persona summary，包含 `<CONTRIBUTION>`, `<PUNISHMENT>`, `<REWARD>`, `<COMMUNICATION>` 等段落。

---

## 3. Method

### 3.1 Representation and Clustering
- Embedding: `text-embedding-3-large` (`1536` dim, batch `200`)
- UMAP: `n_components=2`, `n_neighbors=int(N/5)`, `min_dist=0.5`, `metric=cosine`, `random_state=42`
- Clustering: `KMeans`, 自适应 `k`（基于 base_k=15 与 split 样本量，实际多为 6/7）

### 3.2 Stability Metric
- Cluster alignment 使用 Hungarian assignment，在 split A/B 的 cluster centroids 间最小化总欧氏距离。
- 主指标：`alignment_score = matched centroid distance mean`（越低越稳定）。
- 辅助指标：silhouette（高好）、Davies-Bouldin（低好）、Calinski-Harabasz（高好）。

### 3.3 Experimental Protocols
- **Existing (legacy)**：每个 split 独立建空间后对齐（来自既有结果，作为参考）。
- **Step1 (shared-space original)**：先在全数据拟合一次 embedding+UMAP，再在同一空间内做 split clustering。
- **Step2 (shared-space delex-openai)**：去机制标签和机制词后，在全数据上重新做 OpenAI embedding + global UMAP，再 split clustering。
- **Bootstrap**：每个 tag 每个协议做 100 次 bootstrap，报告 95% CI。

---

## 4. Results

### 4.1 Point Estimates
| Tag | Existing Alignment | Step1 Alignment | Step2 Alignment (delex-openai) | Δ(Step2-Step1) |
|---|---:|---:|---:|---:|
| PUNISHMENT | 0.693 | 1.045 | 0.642 | -0.402 |
| REWARD | 0.172 | 0.932 | 0.798 | -0.134 |
| COMMUNICATION | 0.423 | 0.886 | 0.643 | -0.244 |

解读：在共享空间设定下（Step1），三类机制的 split 差异都较明显；去机制词后（Step2），三类都变得更可对齐（alignment 下降）。

### 4.2 Bootstrap 95% CI (Alignment)
| Tag | Step1 CI | Step2 CI | CI Overlap |
|---|---|---|---:|
| PUNISHMENT | [1.009, 1.166] | [0.603, 0.852] | 0 |
| REWARD | [0.882, 0.967] | [0.760, 0.941] | 0.059 |
| COMMUNICATION | [0.868, 0.999] | [0.484, 0.726] | 0 |

解读：
- `PUNISHMENT` 与 `COMMUNICATION` 的 alignment 改善在 CI 上无重叠，证据强；
- `REWARD` 方向一致但 CI 有部分重叠，证据中等。

### 4.3 Silhouette Trade-off
Step2 相比 Step1 的 silhouette（WITH/WITHOUT）普遍下降（例如 PUNISHMENT WITH: 0.440→0.362），表示去机制词后簇分离度下降。

这说明：
- 机制词确实携带强判别信号（提升 separability）；
- 同时它们也引入跨机制 split 的“词汇层差异”（降低结构稳定性）。

---

## 5. Interpretation

### 5.1 What is stable?
在去机制词后，三种 split 的 alignment 都下降到约 0.64~0.80，说明存在跨机制共享的行为结构（并非完全由 game variant 决定）。

### 5.2 What is unstable?
机制词（punish/reward/chat/sanction 等）会显著放大 split 间距离：Step1 的 alignment 系统性高于 Step2。

### 5.3 Why Existing vs Step1 differ?
Existing 协议中每个 split 独立建空间，alignment 绝对值不能直接与 shared-space 绝对比较；Step1/Step2 更符合“同一坐标系内比较稳定性”的因果解释要求。

---

## 6. Conclusion
- 在共享空间下，persona stability 是“中等稳定”：不同机制标签会改变 cluster geometry。
- 去机制词 + OpenAI embedding 后，稳定性显著提升（尤其 PUNISHMENT、COMMUNICATION），说明不稳定性的重要来源是机制词汇，而非全部深层行为差异。
- 但 silhouette 明显下降，表示机制词也提供了真实可分的结构信息；去词并非“更好聚类”，而是更接近“机制不敏感”的比较目标。

简言之：
**你观察到的 split 差异既包含真实策略结构变化，也包含机制词导致的表示偏移。Step2 将后者显著剥离出来。**

---

## 7. Threats to Validity
- 仍使用 KMeans + 2D UMAP；若用高维直接聚类（或谱聚类）可能改变绝对值。
- alignment 基于 centroid，不反映簇形状/密度全貌。
- delex 词表是规则式，可能同时删掉部分有效语义。
- bootstrap 仅 100 次；可提升到 500+ 获得更平滑 CI。

---

## 8. Artifacts and Reproducibility
### Key outputs
- Existing enhanced summary: `/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_advanced_delex_openai/step0_existing_enhanced.png`
- Step1 summary: `/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_advanced_delex_openai/step1_shared_space_original/summary_dashboard.png`
- Step2 summary: `/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_advanced_delex_openai/step2_shared_space_delex/summary_dashboard.png`
- Step1 vs Step2: `/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_advanced_delex_openai/step12_compare.png`
- Bootstrap CI (Step1): `/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_advanced_delex_openai/step1_shared_space_original/bootstrap_step1_original_ci.png`
- Bootstrap CI (Step2): `/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_advanced_delex_openai/step2_shared_space_delex/bootstrap_step2_delex_ci.png`
- Run manifest: `/Users/kehangzh/Desktop/PGG-finetuning/Persona/persona_stability_advanced_delex_openai/manifest.json`

### Script
- `/Users/kehangzh/Desktop/PGG-finetuning/Persona/cluster_persona_stability_advanced.py`
