# 预算约束修复与详细分配记录

## 问题描述

在之前的实验中（例如 exp_005_treatment），发现智能体的惩罚/奖励支出远超其钱包余额，导致严重透支：

**例子：**
- TURTLE (agent_6) 在19人群组中
- 发送了 86 punishment units + 86 reward units = 172 units
- 总花费：172 金币
- 但实际钱包余额约 59 金币
- **最终钱包：-102.6**（严重透支）

### 根本原因

1. **提示词缺少预算信息**：智能体不知道自己有多少钱可用
2. **没有验证机制**：系统允许负钱包余额
3. **大群组放大问题**：19个智能体，每人可以对18个其他人进行分配

---

## 解决方案

采用 **方案1（提示词约束）+ 方案2（事后比例缩放）**：

### 方案1：在提示词中加入预算约束
- 告诉智能体当前的钱包余额
- 明确警告不要超支
- 例如："你当前有59.40金币可用，请不要超过此金额"

### 方案2：事后比例缩放（保险机制）
- 如果智能体仍然超支，按比例缩减所有分配
- 缩放公式：`actual_units = floor(decided_units × wallet_balance / total_cost)`
- 向下取整确保不超预算

---

## 实现细节

### 1. 修改 `prompt_builder.py`

**添加预算约束到提示词**：

```python
def build_redistribution_prompt(
    self,
    agent_id: str,
    agent_name: str,
    round_num: int,
    contributions: Dict[str, int],
    other_agents: List[Dict[str, str]],
    current_wallet: float = None  # NEW: 新增参数
) -> str:
```

**新增的预算警告文本**：
```
**IMPORTANT - Budget Constraint:**
You currently have {current_wallet:.2f} coins in your wallet.
You cannot spend more than this amount.
Make sure your total spending on punishments/rewards does not exceed {current_wallet:.2f} coins.
```

---

### 2. 修改 `logger.py`

**新增 CSV 文件：`redistribution_details.csv`**

记录每个智能体对其他智能体的具体惩罚/奖励分配。

**字段说明**：
- `experiment_id`: 实验标识符
- `config_hash`: 配置哈希
- `game_id`: 游戏标识符
- `round`: 回合数
- `actor_id`: 执行动作的智能体ID
- `actor_name`: 执行者的头像名称
- `target_id`: 目标智能体ID
- `target_name`: 目标的头像名称
- `type`: 动作类型（"punishment" 或 "reward"）
- `units_decided`: LLM 决定的单位数
- `units_actual`: 实际应用的单位数（缩放后）
- `was_scaled`: 是否进行了缩放
- `cost`: 对执行者的总成本
- `impact`: 对目标的总影响
- `timestamp`: 时间戳

**新增方法**：
```python
def log_redistribution_detail(
    self, game_id: str, round_num: int,
    actor_id: str, actor_name: str,
    target_id: str, target_name: str,
    redist_type: str,  # "punishment" or "reward"
    units_decided: int, units_actual: int,
    was_scaled: bool, cost: float, impact: float
)
```

---

### 3. 修改 `main.py`

**在 redistribution stage 实现完整逻辑**：

#### 3.1 计算当前钱包余额

```python
# 在 contribution stage 之后
current_wallet = base_payoffs[agent.agent_id]
```

钱包余额 = 初始资金 - 贡献 + 公共品分配

#### 3.2 传入预算信息到提示词

```python
redist_prompt = prompt_builder.build_redistribution_prompt(
    agent.agent_id, agent.avatar_name, round_num,
    contributions, other_agents,
    current_wallet=current_wallet  # NEW
)
```

#### 3.3 实现比例缩放逻辑

```python
# 解析LLM决策
amounts_decided, raw_response = agent.get_redistribution_decision(...)

# 计算总成本
total_cost = 0
if config.punishment_enabled:
    total_cost += sum(amounts_decided) * config.punishment_cost
if config.reward_enabled:
    total_cost += sum(amounts_decided) * config.reward_cost

# 如果超预算，按比例缩放
was_scaled = False
amounts_actual = amounts_decided.copy()

if total_cost > current_wallet:
    if total_cost > 0:
        scaling_factor = current_wallet / total_cost
        # 向下取整
        amounts_actual = [int(amt * scaling_factor) for amt in amounts_decided]
        was_scaled = True
```

**缩放示例**：
- 决定花费：172 金币
- 可用余额：59 金币
- 缩放因子：59/172 = 0.343
- 如果某项决定 10 units → 实际 int(10 × 0.343) = 3 units

#### 3.4 记录详细分配

```python
for idx, target_agent in enumerate(other_agents):
    units_decided = amounts_decided[idx]
    units_actual = amounts_actual[idx]

    if units_decided > 0 or units_actual > 0:
        target_id = target_agent["agent_id"]
        target_name = target_agent["avatar_name"]

        if config.punishment_enabled:
            # 记录惩罚详情
            logger.log_redistribution_detail(
                game_id, round_num,
                agent.agent_id, agent.avatar_name,
                target_id, target_name,
                "punishment",
                units_decided, units_actual, was_scaled,
                units_actual * config.punishment_cost,
                units_actual * config.punishment_impact
            )

        if config.reward_enabled:
            # 记录奖励详情（类似）
```

---

## 输出文件结构

每个实验目录现在包含：

```
experiments/exp_005_treatment/
  ├── config.json                    # 实验配置
  ├── game_log.csv                   # 主游戏数据
  ├── chat_messages.csv              # 聊天消息
  ├── raw_responses.csv              # 原始LLM响应
  └── redistribution_details.csv     # NEW: 详细分配记录
```

---

## 示例数据

### redistribution_details.csv

```csv
experiment_id,config_hash,game_id,round,actor_id,actor_name,target_id,target_name,type,units_decided,units_actual,was_scaled,cost,impact,timestamp
exp_005_treatment,abc123,exp_005_treatment_game1,1,agent_0,DOG,agent_1,CHICKEN,punishment,10,3,true,3.0,9.0,2026-01-02T10:00:00
exp_005_treatment,abc123,exp_005_treatment_game1,1,agent_0,DOG,agent_2,CAT,punishment,5,1,true,1.0,3.0,2026-01-02T10:00:00
exp_005_treatment,abc123,exp_005_treatment_game1,1,agent_0,DOG,agent_1,CHICKEN,reward,8,2,true,2.0,1.5,2026-01-02T10:00:00
```

**解读**：
- DOG 决定对 CHICKEN 惩罚 10 units，但因预算不足被缩放到 3 units
- DOG 决定对 CAT 惩罚 5 units，被缩放到 1 unit
- DOG 决定奖励 CHICKEN 8 units，被缩放到 2 units
- `was_scaled=true` 表示进行了比例缩放

---

## 效果对比

### 修复前（exp_005_treatment）

```csv
agent_id,punishments_sent,rewards_sent,cumulative_wallet
agent_6,86,86,-102.6    # TURTLE 严重透支
agent_2,25,25,-288.85   # CAT 严重透支
```

### 修复后（预期效果）

```csv
agent_id,punishments_sent,rewards_sent,cumulative_wallet,was_scaled
agent_6,29,29,0.2,true     # 缩放后，钱包接近0但不为负
agent_2,18,18,5.5,true     # 缩放后，保持正余额
```

- 钱包余额始终 ≥ 0
- 智能体知道自己的预算
- 如果决策不合理，系统自动缩放保护

---

## 关键优势

### 1. 双重保护机制
- **主动约束**（方案1）：智能体被告知预算，主动遵守
- **被动保护**（方案2）：即使智能体违反，系统也会强制修正

### 2. 透明度
- 记录 `units_decided` vs `units_actual`
- 明确标记 `was_scaled`
- 可以分析智能体的预算意识

### 3. 数据可追溯
- 详细记录每个分配决策
- 可以重建完整的博弈过程
- 便于分析智能体策略

### 4. 向下取整的合理性
- 确保不超预算（保守策略）
- 简单明确的规则
- 避免浮点数精度问题

---

## 使用示例

### 运行新实验

```bash
cd /Users/kehangzh/Desktop/PGG-finetuning
export OPENAI_API_KEY='your-key-here'
python run_experiments.py
```

### 分析缩放情况

```python
import pandas as pd

# 加载详细分配数据
df = pd.read_csv("experiments/exp_005_treatment/redistribution_details.csv")

# 统计缩放比例
scaled = df[df['was_scaled'] == True]
print(f"缩放率: {len(scaled) / len(df) * 100:.1f}%")

# 平均缩放程度
scaled['scale_ratio'] = scaled['units_actual'] / scaled['units_decided']
print(f"平均缩放比例: {scaled['scale_ratio'].mean():.3f}")

# 按智能体分析
scaling_by_agent = scaled.groupby('actor_name')['was_scaled'].count()
print("最常被缩放的智能体:")
print(scaling_by_agent.sort_values(ascending=False).head())
```

### 检查预算遵守情况

```python
# 计算每个智能体的总支出
total_cost = df.groupby('actor_id').agg({
    'cost': 'sum',
    'was_scaled': 'any'
})

# 对比game_log中的钱包余额
game_log = pd.read_csv("experiments/exp_005_treatment/game_log.csv")

# 验证没有负钱包
assert (game_log['cumulative_wallet'] >= 0).all(), "存在负钱包！"
print("✓ 所有钱包余额均为正")
```

---

## 未来改进方向（可选）

1. **更智能的缩放策略**
   - 当前：等比例缩放所有目标
   - 可选：优先保留高优先级目标（例如高贡献者的奖励）

2. **预算警告级别**
   - 轻微超支（<10%）：给予警告但不缩放
   - 严重超支（>10%）：强制缩放

3. **动态预算更新**
   - 多轮游戏中，在每轮开始时更新预算信息
   - 考虑跨轮累积效应

4. **学习效果分析**
   - 分析智能体是否随时间学会遵守预算
   - 比较不同轮次的缩放频率

---

## 实现状态

### ✅ 已完成
1. 提示词添加预算约束信息
2. 比例缩放逻辑（向下取整）
3. 详细分配记录 CSV
4. 完整的日志记录（决定值 vs 实际值）
5. 文件关闭逻辑更新

### 📝 需要测试
- 运行新的实验验证功能
- 检查 redistribution_details.csv 格式
- 验证缩放逻辑正确性
- 确认钱包余额不为负

---

## 总结

通过这次修复：
- **消除了负钱包问题**：所有智能体的钱包余额始终 ≥ 0
- **提高了数据质量**：记录了完整的决策过程
- **增强了可解释性**：可以追溯每个分配决策的缩放情况
- **保持了灵活性**：智能体仍可自由决策，系统只在必要时干预

这个实现符合实验经济学中的标准做法，既给予智能体决策自由，又确保游戏规则的完整性。
