# Step 1：冻结 Receiver 的单一流程

仅保留当前新版：公开 ETO 数据 → Sender 计划/latent → adapter 三分支训练 →
ALFWorld 评估 → 完整性审计和汇总。Sender 为冻结的 7B Instruct，Receiver 为冻结的
7B Base；这仍是既有冻结变体，不等同于原论文联合训练的完整复现。
整理代码不授权启动新实验，也不改变已有训练目标、数据过滤规则或默认预算。

## 文件职责

- `common.py`：唯一的模型 revision、维度、固定种子与预算常量；文件哈希/JSON 工具。
- `serialization.py`：唯一的 Receiver ChatML 渲染；消息注入在首个 user 结束符之前。
- `adapter.py`：适配器算子、冻结边界、embedding 注入与 CE 前向。
- `runtime.py`：Base 加载、数据/latent 读取、监督位置对齐与 NLL。
- `env_utils.py`：canonical 游戏枚举、环境创建、观察清理与原有动作解析规则。
- `prepare_data.py`：公开 ETO 轨迹转换、原 seed-42 的 5% selection 划分；不筛除失败轨迹。
- `prepare_eval.py`：准备 140 个 valid_seen 初始状态，不读取专家动作作为答案。
- `collect_latents.py`：唯一的采集入口；沿用采样配置、任务 RNG、bf16 缓存与 EOS 对齐。
- `train_step1.py`：唯一的三分支训练入口；内含 text-prefix/latent-suffix 混合、CE、负 JS、KL 和 cosine 对齐；默认 3 epochs / 24 小时，选择集 NLL 选 checkpoint。
- `eval_adapter.py`：唯一的评估入口；initial/best/last、原有七组对照与 20-action 上限。
- `summarize_results.py`：核验输入、缓存、checkpoint、续训日志和评估结果，再生成报告。
- `data_collection/audit_eto_trajectories.py`：公开轨迹在当前环境的重放审计；规范化观察比较，不修改动作、不筛除失败轨迹。
- `diagnostics/audit_latent_cache.py`：全量缓存哈希、形状、有限值与对齐审计。
- `diagnostics/verify_cached_batch.py`：独立的同批次强制 token 重放，核验缓存位置。
- `docs/`：执行规范、来源记录与历史证据。

旧 CE-only 流程、停用入口、一次性编排脚本以及 `step1/` 全部测试文件已删除。
缓存/数据审计属于实验产物的有效性校验，不是保留的测试套件。没有另建配置目录或第二套实现。

## 环境与执行顺序

所有命令从 `latent-communication/code/` 执行。沿用已有 uv 环境；恢复说明见
[`ENVIRONMENT.md`](../ENVIRONMENT.md)。以下是命令说明，不代表已执行或新授权。
`$ETO` 为原始 ETO JSON，`$GAMES` 为 ALFWorld `json_2.1.1`，`$PREPARED` 与
`$RUN` 为新的输出目录。模型在已有缓存中，以离线方式加载。

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 1. 重放公开轨迹，保留失败记录；再转换数据与固定划分
uv run --no-sync python -m step1.data_collection.audit_eto_trajectories \
  --data "$ETO" --games "$GAMES" --out-dir "$RUN/replay"
uv run --no-sync python -m step1.data_collection.prepare_data \
  --source "$ETO" --audit "$RUN/replay/replay.jsonl" --out-dir "$PREPARED"
uv run --no-sync python -m step1.prepare_eval \
  --source "$ETO" --games "$GAMES" --out "$RUN/valid_seen.jsonl"

# 2. 分别采集训练/selection 与 valid_seen 缓存
uv run --no-sync python -m step1.collect_latents \
  --data "$PREPARED/fit.jsonl" "$PREPARED/selection.jsonl" --out-dir "$RUN/latents"
uv run --no-sync python -m step1.collect_latents \
  --data "$RUN/valid_seen.jsonl" --out-dir "$RUN/eval_latents"

# 3. 独立核验采集批次与全量缓存
uv run --no-sync python -m step1.diagnostics.verify_cached_batch \
  --data "$PREPARED/fit.jsonl" --latents "$RUN/latents" --out "$RUN/train_cache_exact.json"
uv run --no-sync python -m step1.diagnostics.audit_latent_cache \
  --data "$PREPARED/fit.jsonl" "$PREPARED/selection.jsonl" --latents "$RUN/latents" \
  --exact-replay "$RUN/train_cache_exact.json" --out "$RUN/cache_audit.json"
uv run --no-sync python -m step1.diagnostics.verify_cached_batch \
  --data "$RUN/valid_seen.jsonl" --latents "$RUN/eval_latents" --out "$RUN/eval_cache_exact.json"
uv run --no-sync python -m step1.diagnostics.audit_latent_cache \
  --data "$RUN/valid_seen.jsonl" --latents "$RUN/eval_latents" \
  --exact-replay "$RUN/eval_cache_exact.json" --out "$RUN/eval_cache_audit.json"

# 4. 训练与独立任务评估
uv run --no-sync python -m step1.train_step1 \
  --fit "$PREPARED/fit.jsonl" --selection "$PREPARED/selection.jsonl" \
  --latents "$RUN/latents" --out-dir "$RUN/train"
uv run --no-sync python -m step1.eval_adapter \
  --data "$RUN/valid_seen.jsonl" --latents "$RUN/eval_latents" \
  --train-dir "$RUN/train" --out-dir "$RUN/eval" \
  --selection "$PREPARED/selection.jsonl" --train-latents "$RUN/latents"

# 5. 核验并汇总；不依赖已删除的编排脚本或 pipeline_complete.json
uv run --no-sync python -m step1.summarize_results --root "$RUN"
```

### 轨迹重放审计

`--data` 是原始 ETO JSON，`--games` 是包含 `train/` 的 `json_2.1.1/` 目录，
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
uv run --no-sync python -m step1.train_step1 \
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

- [执行规范与解释边界](docs/FROZEN_RECEIVER_V1.md)
- [论文/公开代码来源核验](docs/PAPER_ALIGNMENT.md)
- [历史记录](docs/HISTORY.md)：旧命令、旧测试结果和一次性脚本只作为历史证据，不是当前入口。
