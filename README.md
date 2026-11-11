# Step 1：冻结 Receiver 的单一流程

公开 ETO 数据 → Sender 计划/latent → adapter 三分支训练 →
ALFWorld 评估 → 完整性审计和汇总。

本分支仅保留当前 Step 1 流程，不是上游联合训练方法的完整复现。
历史采集、联合训练、压缩、评估和 shell 编排流程及其 CLI 入口已移除。
从本目录使用 `uv` 运行命令；依赖以 `pyproject.toml` 和 `uv.lock` 为准，
`setup.py` 为 setuptools 入口。环境准备与平台限制见 [ENVIRONMENT.md](ENVIRONMENT.md)。

## 文件职责

- `configs/step1.yaml`：当前完整实验配置，按模型、数据准备、训练和评估分组。
- `configs/config.py`：直接读取 YAML，并展开数据和输出路径；各模块直接取对应字段。
- `utils/hashing.py`：文件哈希与 JSON 写入工具。
- `alfworld_utils.py`：canonical 游戏枚举、环境创建、观察清理与原有动作解析规则。
- `model/`
  - `receiver.py`：Receiver ChatML 渲染、监督标签与潜在消息注入；消息注入在首个 user 结束符之前。
  - `adapter.py`：适配器算子，包含原样保留的上游 AdaptiveProjection，不再依赖旧训练目录。
  - `runtime.py`：数据/latent 读取、监督位置对齐与 NLL；Base 加载、冻结边界、embedding 注入与 CE 前向在 `receiver.py`。
- `data_collection/`
  - `gen_training_data.py`：成功 ETO 子集转换、原 seed-42 的 5% selection 划分。
  - `gen_eval_tasks.py`：准备 140 个 valid_seen 初始状态，不读取专家动作作为答案。
  - `gen_sender_messages.py`：唯一的采集入口；沿用采样配置、任务 RNG、bf16 缓存与 EOS 对齐。
  - `replay_eto_trajectories.py`：公开轨迹在当前环境的重放审计；规范化观察比较，不修改动作、不筛除失败轨迹。
- `train/train_step1.py`：唯一的三分支训练入口；内含 latent-prefix/text-suffix 混合、CE、负 JS、KL 和 cosine 对齐；默认 3 epochs、无时间上限，选择集 NLL 选 checkpoint。
- `eval/`
  - `eval_adapter.py`：唯一的评估入口；initial/best/last、原有七组对照与 20-action 上限。
  - `summarize_results.py`：核验输入、缓存、checkpoint、续训日志和评估结果，再生成报告。
- `docs/`：执行规范、来源记录与历史证据。

## 环境与执行顺序

### 实验配置

默认使用 [`configs/step1.yaml`](configs/step1.yaml)，不传参数时保持原来的实验默认值。
调整模型、学习率、batch、epoch、损失权重或数据路径，只需修改 YAML，不再修改实现代码。
当前默认配置的 Base 与缓存使用 `bfloat16`。
Adapter 参数量由配置维度计算，不再手填常量。

建立另一次实验时，复制整个文件，并使用独立的 `data.paths.run_dir` 和（需要重新划分时）
`data.paths.prepared_dir`；每份 YAML 都必须完整，不使用隐藏的配置继承。
保持现有模块布局，不为此次配置整理迁移到 `src/`。

```bash
cp configs/step1.yaml configs/my_experiment.yaml
# 修改 my_experiment.yaml 中的参数和输出目录；各阶段使用同一配置。
uv run --no-sync python -m data_collection.gen_sender_messages --config configs/my_experiment.yaml
uv run --no-sync python -m train.train_step1 --config configs/my_experiment.yaml
uv run --no-sync python -m eval.eval_adapter --config configs/my_experiment.yaml --selection
uv run --no-sync python -m eval.summarize_results --config configs/my_experiment.yaml
```

所有数据准备入口也支持 `--config`。YAML 相对路径以 `code/` 为基准；
`${data.paths.run_dir}` 等字段引用会递归展开，支持嵌套字典和列表；完整字段引用保留原值类型。
以 `_dir` / `_path` 结尾的字段自动按路径处理，相对路径以 `code/` 为基准；
既有的 `alfworld_games`、`train_latents`、`eval_latents` 保留路径语义。
新增符合命名约定的路径字段无需再修改读取器；`name_or_path` 保留 Hugging Face 模型标识语义，
不自动转成绝对路径，其他普通字符串也不转换。
循环引用会报错；不额外做实验字段、类型或范围校验。
公共根目录集中在 `data.paths`，也供训练和评估输出引用，不再设置顶层 `paths`。
`data` 内按准备环节划分为 `source`（原始数据）、`replay`（重放审计）、
`successful`（成功筛选）、`split`（转换与划分）、`eval_tasks`（评估任务准备）和
`collection`（Sender 计划与 latent 采集）；采集参数不再作为顶层分组。
支持 `${data.successful.output_dir}` 等多级路径引用。
配置读取器只读取 YAML，不再读取 `PREPARED` / `RUN` 环境变量。
现有命令行仅覆盖其对应入口选项，不会把一个阶段的 `--out-dir` 自动传给下一阶段；
正式实验建议统一改 YAML。`--smoke`、`--resume-epoch` 和诊断参数仍是执行控制选项。

各阶段的 `config.json` 或 manifest 保存 `experiment.settings`（路径展开后的完整 YAML）
与 `experiment.arguments`（含命令行覆盖的实际入口参数）；训练 checkpoint 也携带该记录。
续训继续使用原有的源码与数据哈希核验；旧产物必须使用原代码快照，不修改旧记录。
本次仅整理配置，未启动训练或改变默认实验目标。

### 2026-11-09 完整训练配置

[`configs/step1_full_20261109.yaml`](configs/step1_full_20261109.yaml) 沿用 18 步检查的训练参数：
batch 16、3 epochs、峰值学习率 1e-5、3% warmup、无时间/更新步数上限。
复用 `data/prepared/` 的 2,244 条 fit / 118 条 selection 和
`data/runs/frozen_receiver_v1/latents/` 的 Sender 缓存，从相同种子重新初始化 adapter。
新运行输出到 `data/runs/frozen_receiver_full_20261109/train/`，计划 423 次更新、13 步 warmup。
保存每 25 步及每个 epoch 的 checkpoint，初始化和各 epoch 的 selection NLL 用于选择最佳模型。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run --no-sync python -u -m train.train_step1 \
  --config configs/step1_full_20261109.yaml
```

GB10 使用 `ENVIRONMENT.md` 中记录的 GPU 环境（torch 2.9.1+cu128），运行命令保留
`--no-sync`；锁文件的 torch 2.4.1 在本机 ARM 上会解析成 CPU 版本，不能用于本次训练。

预处理和运行根目录分别由 YAML 的 `data.paths.prepared_dir`、`data.paths.run_dir` 配置。
默认路径为 `code/data/prepared` 和 `code/data/runs/frozen_receiver_v1`；修改根目录请编辑配置文件。

```yaml
data:
  paths:
    prepared_dir: data/prepared
    run_dir: data/runs/frozen_receiver_v1
```

1. 重放公开轨迹并保留完整审计，再导出成功子集、转换数据与固定划分
```
uv run --no-sync python -m data_collection.replay_eto_trajectories
uv run --no-sync python -m data_collection.filter_successful_eto
uv run --no-sync python -m data_collection.gen_training_data
uv run --no-sync python -m data_collection.gen_eval_tasks
```

2. 分别采集训练/selection 与 valid_seen 缓存
```
uv run --no-sync python -m data_collection.gen_sender_messages
uv run --no-sync python -m data_collection.gen_sender_messages --split eval
```

3. 训练与独立任务评估
```
uv run --no-sync python -m train.train_step1
uv run --no-sync python -m eval.eval_adapter --selection
```

4. 核验并汇总；不依赖已删除的编排脚本或 pipeline_complete.json
```
uv run --no-sync python -m eval.summarize_results
```

### 轨迹重放审计

审计直接读取 YAML 的 `data.source.eto_path` 和包含 `train/` 的 `data.source.alfworld_games` 目录。
输出目录（默认 `data/runs/frozen_receiver_v1/replay/`）必须是新目录。审计不加载模型、不使用 GPU，也不额外限制动作步数；
执行原始动作直到轨迹结束或环境终止。失败和异常均逐条保留，不用于自动过滤样本。

输出为 `replay.jsonl`（每条轨迹的成功状态、初始观察匹配、观察差异和异常）、
`summary.json`（汇总）及 `config.json`（来源、代码哈希、依赖版本）。
`gen_training_data.py` 默认读取成功子集，并按 ID 从配置的审计目录中的 `replay.jsonl` 附加审计信息、校验成功状态。
审计不证明该公开数据就是论文最终训练数据，也不是模型成功率评估。

首次检查可加 `--limit 2`；这只重放前两条，不能直接供全量数据转换使用。
全量运行省略 `--limit`。已有结果不会被覆盖。

### 独立导出成功轨迹

如需成功示范子集，在全量审计后运行：

```bash
uv run --no-sync python -m data_collection.filter_successful_eto
```

脚本核验源数据和审计哈希、完整性及 ID 一一对应，仅保留 `success=true`、
`reason=won` 且无执行/关闭错误的原始记录，不要求中间观察完全匹配。
默认输出 `data/runs/frozen_receiver_v1/successful_eto/alfworld_sft_successful.json` 和来源/筛选 manifest；
保留原始记录内容及顺序，不覆盖已有目录，不修改原数据或审计。
这只是当前环境回放成功子集，不是模型评估结果，也不代表论文最终训练数据。
数据源路径配置在 `data.source`；成功子集只配置 `data.successful.output_dir`，
文件名固定为 `alfworld_sft_successful.json`，不再单独配置 `output_path`。
划分程序直接从 `data.successful.output_dir` 读取 `alfworld_sft_successful.json`，不再单独配置输入目录。
`data.split.output_dir` 指定划分输出目录，默认引用 `data.paths.prepared_dir`；
划分程序固定写入 `fit.jsonl`、`selection.jsonl` 和 `manifest.json`，采集、训练及评估从该目录读取对应文件。
审计只配置 `data.replay.output_dir`，记录文件固定为该目录下的 `replay.jsonl`，
不再单独配置 `audit_path`。评估输入使用 `data.eval_tasks.output_path`。
导出不会自动重新划分；随后运行 `gen_training_data`，它要求源数据全部为审计成功记录。

## 续训

短程稳定性检查可用 `--stop-after-updates 5`：只停止执行，不缩短由全量 fit 和
`--epochs` 决定的学习率日程。不要用 `--smoke` 代替真实 batch/warmup 检查。
`--diagnostic-probes <JSON>` 接受形如
`[{"id":"alfworld_287","latent_fraction":0.1},{"id":"alfworld_1470","latent_fraction":0.5}]`
的固定 fit 样本配置，逐步记录纯 latent NLL、混合 NLL、消息变化及末层投影变化，
并保存每步 checkpoint；探针不用于选择 checkpoint，测量恢复训练模式和 RNG 状态。
不足一个 epoch 的短程 checkpoint 不适用于当前仅支持完整 epoch 边界的续训入口。

2026-11-05 起，默认取消训练时间上限（`--max-hours 0`）；仍可显式设置正数作为可选上限。
默认 epoch 数仍为 3，这是本地实验预算选择，不是由论文推导出的数值；作者公开 `train.sh` 默认为 10。
训练错配分支仅替换 latent，文本计划保持当前任务不变。donor 的原始 latent 在进入 adapter 前
按当前任务通信长度截断，或重复后截断并加标准差 0.01 的高斯噪声；等长不加噪声。
混合方式改为作者公开代码的 latent 前缀/当前任务文本后缀，均匀采样 latent 比例
`r ∈ {0.1,0.2,...,0.9}`，切分点为 `round(r*L)`；匹配与错配使用同一比例。
比例不采样 0 或 1，但极短序列经取整仍可能没有某一通道的位置。
匹配和错配分支仍均保留梯度。评估对照的构造未随本次修改。

2026-11-05 起，辅助损失权重改用作者公开代码的范围与按损失调整方式：
`lambda_sep = 0.01 + 0.49 * clip(max(0, 0.69 - JS) / 0.69, 0, 1)`，
`lambda_align = 0.01 + 0.04 * clip((KL + cosine_distance) / 4, 0, 1)`。
按每条样本完整监督位置的损失计算系数，再按监督 token 数累积梯度；保留系数的梯度。
本次只对齐权重机制，仍使用负标准 JS、KL(matched||text) 和 logits cosine，
不采用作者不同的 divergence、KL 方向或内部 0.7/0.3 系数，不能称为完整复现作者损失。
日志保存逐样本系数、token 加权平均系数和实际总损失。旧线性权重 smoke 结果仅对应旧代码，
不改写既有记录或使用旧 checkpoint 续训新目标。

使用相同数据、配置与源码，从完整 epoch 边界继续；输出目录必须是新目录。

```bash
uv run --no-sync python -m train.train_step1 \
  --out-dir data/runs/frozen_receiver_v1/continued_train \
  --epochs 3 --resume-epoch data/runs/frozen_receiver_v1/train/epoch_01.pt
```

- `--epochs 3` 是总共三轮，不是额外三轮；保留已有 AdamW 重建语义，不声称完整恢复优化器/RNG。
- 必须保留来源目录的 `initial.pt`、`best.pt`、epoch checkpoint 与 `training.jsonl`。
- 新目录继承历史最优 checkpoint 的真实权重、NLL 与 step，初始化也参与比较；同分保留较早版本。
- 如果来源 `best.pt` 来自续训点之后，拒绝该回退续训，避免混入未来选择结果。
- 数据、源码及 checkpoint 的严格校验不会因整理而放宽。旧路径/旧版本运行应使用原代码快照，
  不要修改旧 manifest 或跳过哈希。现有实验产物未移动或重写。
- 评估续训目录时将 `--train-dir` 指向新目录，评估输出另选新目录；汇总时传入对应
  `--train-dir`、`--eval-dir`。汇总会追溯并校验父运行日志，只计入续训起点之前的更新。

## 规范与历史

当前模块直接位于本目录，命令使用 `python -m data_collection.…`、
`python -m train.train_step1` 和 `python -m eval.…`，不提供旧 `scripts` 或 `step1` 兼容入口。
源码哈希使用根目录模块路径（如 `train/train_step1.py`）；目录迁移不是旧 checkpoint 的兼容转换。
旧运行须使用其原始源码快照，不改写 manifest、不放宽续训校验。

2026-11-03 按用户要求清理了本地历史实验产物，包括旧 fit/selection、latent、
adapter checkpoint、重放结果、日志与环境诊断；原始 ETO 和 ALFWorld 已迁入上述
固定目录，预训练模型缓存保留。历史文档里的旧产物路径仅为来源记录，本地已不存在。
清理没有启动新实验；重跑需从重放审计和数据转换开始。
原始数据路径为 `data/eto/alfworld_sft.json` 与 `data/alfworld/json_2.1.1/`。

- [执行规范与解释边界](docs/FROZEN_RECEIVER_V1.md)
- [论文/公开代码来源核验](docs/PAPER_ALIGNMENT.md)
- [历史记录](docs/HISTORY.md)：旧命令、旧测试结果和一次性脚本只作为历史证据，不是当前入口。
