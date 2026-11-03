# Step 1：冻结 Receiver 的单一流程

公开 ETO 数据 → Sender 计划/latent → adapter 三分支训练 →
ALFWorld 评估 → 完整性审计和汇总。

Sender 为冻结的 7B Instruct，Receiver 为冻结的
7B Base；

## 文件职责

- `common.py`：模型 revision、维度、固定种子与预算常量，文件哈希/JSON 工具数据路径与默认产物目录
- `alfworld_utils.py`：canonical 游戏枚举、环境创建、观察清理与原有动作解析规则。
- `model`
  - `serialization.py`：唯一的 Receiver ChatML 渲染；消息注入在首个 user 结束符之前。
  - `adapter.py`：适配器算子、冻结边界、embedding 注入与 CE 前向；包含原样保留的上游 AdaptiveProjection，不再依赖旧训练目录。
  - `runtime.py`：Base 加载、数据/latent 读取、监督位置对齐与 NLL。
- `data_collection`
  - `prepare_data.py`：公开 ETO 轨迹转换、原 seed-42 的 5% selection 划分；不筛除失败轨迹。
  - `prepare_eval.py`：准备 140 个 valid_seen 初始状态，不读取专家动作作为答案。
  - `collect_latents.py`：唯一的采集入口；沿用采样配置、任务 RNG、bf16 缓存与 EOS 对齐。
  - `audit_eto_trajectories.py`：公开轨迹在当前环境的重放审计；规范化观察比较，不修改动作、不筛除失败轨迹。
- `train/train_step1.py`：唯一的三分支训练入口；内含 text-prefix/latent-suffix 混合、CE、负 JS、KL 和 cosine 对齐；默认 3 epochs / 24 小时，选择集 NLL 选 checkpoint。
- `eval`
  - `eval_adapter.py`：唯一的评估入口；initial/best/last、原有七组对照与 20-action 上限。
  - `summarize_results.py`：核验输入、缓存、checkpoint、续训日志和评估结果，再生成报告。
- `docs/`：执行规范、来源记录与历史证据。

## 环境与执行顺序

所有命令从 `latent-communication/code/` 执行。沿用已有 uv 环境；恢复说明见
[`ENVIRONMENT.md`](../ENVIRONMENT.md)。
原始 ETO 和 ALFWorld 路径直接使用 `common.py` 中的 `ETO_DATA`、`ALFWORLD_GAMES`，
不通过命令行传递。`$PREPARED` 与 `$RUN` 为新的输出目录。
模型在已有缓存中，以离线方式加载。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PREPARED="$PWD/data/prepared"
export RUN="$PWD/data/runs/frozen_receiver_v1"

# 1. 重放公开轨迹，保留失败记录；再转换数据与固定划分
uv run --no-sync python -m scripts.data_collection.audit_eto_trajectories \
  --out-dir "$RUN/replay"
uv run --no-sync python -m scripts.data_collection.prepare_data \
  --audit "$RUN/replay/replay.jsonl" --out-dir "$PREPARED"
uv run --no-sync python -m scripts.data_collection.prepare_eval \
  --out "$RUN/valid_seen.jsonl"

# 2. 分别采集训练/selection 与 valid_seen 缓存
uv run --no-sync python -m scripts.data_collection.collect_latents \
  --data "$PREPARED/fit.jsonl" "$PREPARED/selection.jsonl" --out-dir "$RUN/latents"
uv run --no-sync python -m scripts.data_collection.collect_latents \
  --data "$RUN/valid_seen.jsonl" --out-dir "$RUN/eval_latents"

# 3. 训练与独立任务评估
uv run --no-sync python -m scripts.train.train_step1 \
  --fit "$PREPARED/fit.jsonl" --selection "$PREPARED/selection.jsonl" \
  --latents "$RUN/latents" --out-dir "$RUN/train"
uv run --no-sync python -m scripts.eval.eval_adapter \
  --data "$RUN/valid_seen.jsonl" --latents "$RUN/eval_latents" \
  --train-dir "$RUN/train" --out-dir "$RUN/eval" \
  --selection "$PREPARED/selection.jsonl" --train-latents "$RUN/latents"

# 4. 核验并汇总；不依赖已删除的编排脚本或 pipeline_complete.json
uv run --no-sync python -m scripts.eval.summarize_results --root "$RUN"
```

当前目录没有历史文档提到的 `diagnostics/` 缓存审计入口，上述命令不是完整的
实验有效性认证流程。汇总仍要求已有的缓存审计产物；缺少时不得跳过校验或
声称新运行已通过完整审计。历史命令和产物应使用各自对应的代码 revision。

原始数据位置固定配置为 `data/eto/alfworld_sft.json` 和 `data/json_2.1.1/`，
需要更换时修改 `common.py`。上述 PREPARED、RUN 路径也是代码默认值。
例如 `uv run --no-sync python -m scripts.data_collection.prepare_data` 会读取默认原始
ETO 和默认重放审计，将训练/selection 写到 `data/prepared/`；预处理直接使用 ETO 内的
游戏目录路径，补上 `game.tw-pddl` 后转为绝对路径，不修改原始 JSON。
默认训练读取该划分和 `$RUN/latents/`，写入 `$RUN/train/`。
修改 `$RUN` 等 Shell 变量不会自动改变代码默认值；自定义运行需像上面的命令一样
显式传参，尤其是评估 latent 采集必须同时指定 `--data` 与 `--out-dir`，否则默认采集
训练/selection。`--help` 显示实际默认路径。既有输出的防覆盖/续跑检查保持不变。

### 轨迹重放审计

审计直接读取 `common.py` 的原始 ETO JSON 和包含 `train/` 的 ALFWorld 目录。
`--out-dir` 必须是新目录。审计不加载模型、不使用 GPU，也不额外限制动作步数；
执行原始动作直到轨迹结束或环境终止。失败和异常均逐条保留，不用于自动过滤样本。

输出为 `replay.jsonl`（每条轨迹的成功状态、初始观察匹配、观察差异和异常）、
`summary.json`（汇总）及 `config.json`（来源、代码哈希、依赖版本）。
将 `replay.jsonl` 传给 `prepare_data.py --audit`；这一步只按 ID 附加审计信息。
审计不证明该公开数据就是论文最终训练数据，也不是模型成功率评估。

首次检查可加 `--limit 2`；这只重放前两条，不能直接供全量数据转换使用。
全量运行省略 `--limit`。已有结果不会被覆盖。

## 续训

使用相同数据、配置与源码，从完整 epoch 边界继续；输出目录必须是新目录。

```bash
uv run --no-sync python -m scripts.train.train_step1 \
  --fit "$PREPARED/fit.jsonl" --selection "$PREPARED/selection.jsonl" \
  --latents "$RUN/latents" --out-dir "$RUN/continued_train" \
  --epochs 3 --resume-epoch "$RUN/train/epoch_01.pt"
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

当前 Python 包及命令入口统一为 `scripts`，不提供 `step1` 兼容入口。
源码哈希使用 `scripts/...` 路径；目录迁移不是旧 checkpoint 的兼容转换。
旧运行须使用其原始源码快照，不改写 manifest、不放宽续训校验。

2026-11-03 按用户要求清理了本地历史实验产物，包括旧 fit/selection、latent、
adapter checkpoint、重放结果、日志与环境诊断；原始 ETO 和 ALFWorld 已迁入上述
固定目录，预训练模型缓存保留。历史文档里的旧产物路径仅为来源记录，本地已不存在。
清理没有启动新实验；重跑需从重放审计和数据转换开始。

- [执行规范与解释边界](docs/FROZEN_RECEIVER_V1.md)
- [论文/公开代码来源核验](docs/PAPER_ALIGNMENT.md)
- [历史记录](docs/HISTORY.md)：旧命令、旧测试结果和一次性脚本只作为历史证据，不是当前入口。
