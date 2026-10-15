# Agent Context Compressor

**零 LLM 在线成本的智能体上下文压缩流水线** —— 抽取式保真压缩 + 双支路架构 + 机械生命周期 + 审计闸 + 检索兜底。

> 设计动机：抽取式压缩的「失忆」问题——生成式摘要丢决策链、语气反转（「是不是X」→「X」）、多代压缩的错误固化。本项目的答案：**保真 > 流畅**，宁可读起来是账本，不可编造或漏记。

## 架构

```
原始消息流 ──┬─→ 信息压缩支路（管体积·逐条瘦身）
             │     Stage 1   机械清洗（正则规则, 实测砍 78% 冗余, 5ms）
             │     Stage 1.5 token 级删留头（bge-m3+LoRA 二分类, 140ms 纯 CPU）
             │
             └─→ 提炼摘要支路（管语义·跨条账本）
                   Stage 2   五要素账本（时间/文件/变更/教训/Next, 0.2ms 纯 Python）
                   决策链时间线（指令→动作→结果三元组, 1.1ms）
                   因果判别头（mmBERT-base+LoRA :8767, 歧义配对 rerank, fail-open）
                        ↓ 各自从原文一次成型, 禁互相压缩
             中间层 = 原文尾(50条) + 压缩流 + 账本 + 蔡格尼克免疫区
                        ↓
             验收闸（路径/数字/指令实录覆盖检查, fail 即拦）
                        ↓
             检索兜底（gap 检测钩子 → 向量检索找回超窗历史）
```

## 四条硬规则（来自真实失忆事故）

1. 压缩一次定型，禁二次压缩 —— 摘要被再摘要 = telephone game，第一代的错误会固化成不动点
2. 状态机流转严禁 LLM 参与 —— SQLite/时间戳/计数器不会撒谎；LLM 只允许做要旨精炼
3. 验收闸常设 —— 每次压缩后机械检查路径/数字/指令实录覆盖，fail 即拦
4. 蔡格尼克免疫 —— 未完成事项（open_loop）机械免疫一切衰减，恒定保留

## 实测数据（生产会话真实数据）

| 指标 | 数值 |
| :--- | :--- |
| Stage 1 机械清洗 | 削减 78.23%，5.12ms，实体保全 100% |
| Stage 1.5 token 压缩 | 25% 稀疏化，~140ms 纯 CPU 零显存 |
| v2 压缩头（教师标注 2225 条） | token F1 **0.917** |
| 因果判别头（自监督 6820 对） | mmBERT-base F1 0.873 |
| 决策链三元组 | 1.1ms / 120 turns，指令→动作→结果全保 |
| 锚点覆盖（路径/数字） | 100%（验收闸三连 PASS） |
| gap 检测→检索回填 | 17.8s/gap（异步后台，不阻塞压缩主链 24ms） |

### 两个反直觉发现

- 机械规则是强基线：927 条机械标签训出的模型（F1 0.790）赢过 7 倍数据量的教师标注版（0.780）——教师标签的价值在规模化后才兑现（2225 条→0.917）
- 任务不同，基座另裁：token 标注任务 bge-m3 胜（0.780 vs 0.763）；判别任务 mmBERT-base 反超（0.873 vs 0.847）——「更强的公开基准」不等于「本任务更强」

## 快速开始

```bash
# 1. 准备底座模型（bge-m3 / mmBERT-base, HF 可下）
export MODEL_DIR=~/models

# 2. 从你自己的会话数据构建教师标注（任一 27B/大模型均可当教师）
python3 training/sample_v2_manifest.py       # 分层采样 manifest
python3 training/build_ab_dataset.py         # 字符级对齐 → token 标签

# 3. 训练 token 删留头
python3 training/train_ab_lora.py \
  --base-model $MODEL_DIR/bge-m3 \
  --output-dir models/my-lingua \
  --target-modules query,key,value

# 4. 因果判别头（可选, 自监督零标注成本）
python3 training/build_causal_pairs.py       # 从会话库机械配对
python3 training/train_causal.py \
  --base-model $MODEL_DIR/mmBERT-base \
  --output-dir models/causal-head \
  --report /tmp/causal.json \
  --target-modules Wqkv,Wo
python3 services/causal_head_server.py &     # :8767

# 5. 作为上下文引擎插件接入你的 agent 框架（src/ml_pipeline_plugin）
```

## 状态目录约定

脚本通过 `HERMES_HOME` 环境变量定位状态目录（默认 `~/.agent_ctx`）。接入 Hermes 的机器设 `export HERMES_HOME=~/.hermes` 即可复用现有状态库；独立部署无需任何配置。

## 仓库结构

```
src/            核心压缩管线 + 插件接入层（context_engine / audit_correction）
hooks/          gap 检测钩子 + 判别头服务 + 审计账本
training/       双头训练/评测/数据构建（底座无关, bge-m3 与 mmBERT 同数据 A/B）
services/       embedding service看门狗（探活+自愈）
docs/           设计文档与实测报告
```

## 设计文档

- [双支路架构与 A/B 硬数据](docs/design.md)

## License

Apache-2.0
