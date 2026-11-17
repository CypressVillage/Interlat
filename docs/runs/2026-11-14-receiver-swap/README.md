# 11 月 14 日汇报的 Receiver 替换诊断：运行记录

本目录归档 Issue #1 中 2026-11-14 汇报的五组 `valid_seen` 评测。实际运行目录标记为 11 月 11–12 日；11 月 14 日是汇报日期，不是从运行日志核实的执行日期。归档于 2026-11-17 完成，不启动或重跑实验。

本实验保留针对 `Qwen2.5-7B` Base 训练的 Adapter，仅在评测时更换为 `Qwen2.5-7B-Instruct`。它是历史诊断，不是训练和评测底模一致的正式效果实验，也不是 Step 0 或 Gate 0 通过记录。完整模型 revision、训练设置、评测参数、文件指纹及本地保存位置见 [`run_manifest.json`](run_manifest.json)。

## 结果与来源

五组均使用同一份 140 题任务清单；从各自 `summary.json` 重新统计，并与原 `comparison.json` 核对：

| 组别 | 成功任务 | 成功率 | 原始运行目录名 |
|---|---|---|---|
| Matched | 52/140 | 37.14% | `instruct_receiver_full_20261111` |
| Mismatched | 61/140 | 43.57% | `instruct_controls_full_20261112` |
| Random | 17/140 | 12.14% | `instruct_controls_full_20261112` |
| Text | 9/140 | 6.43% | `instruct_text_full_20261112` |
| No-Comm | 6/140 | 4.29% | `instruct_receiver_full_20261111` |

以上目录位于子仓库 `data/runs/frozen_receiver_full_20261109/`。原始轨迹、latent 和 checkpoint 保留在本地，不复制到 Git；未核实远端备份或其他机器的可访问性。轻量逐题结果另存为 [`episode_results.json`](episode_results.json)，可在没有模型权重时核对计数与配对任务。

## 归档内容

- `run_manifest.json`：各组来源、参数、模型 revision、checkpoint 和结果文件 SHA-256、从记录参数重建的命令。重建命令不是原始 shell 日志，不证明当时的启动器、环境变量或硬件。
- `configs/`：三次评测、Adapter 训练、评测 Sender 缓存的原 `config.json`，按字节复制，不改写历史参数。
- `mismatched_mapping.json`：实际使用的 140 对错配映射，按字节复制。文件 SHA-256 为 `5acf93a8efbfe7d5a99ded01a8348f8a55f66a0ed26a2a17463ca6abe7ec9446`。文件内部自带的 `mapping_sha256` 是另一种摘要，不能代替整文件指纹。
- `build_manifest.py`：从现有材料生成、核验上述归档，不加载模型，不运行训练或环境交互。

评测 Sender 缓存原配置为采样、`max_new_tokens=5000`、`temperature=0.7`、`top_p=0.9`、`top_k=50`，随机源来自 `42:id`。这些是历史记录，不应改写成后续 Step 0 批准的贪心、512 tokens 设置。历史提示词与源码的详细核查见下节；本次不修改后续实验实现。

## 历史 Sender 与源码核查（第三项）

**发送端实际记录的输入缺少环境规则。** 对 140 题缓存 metadata 逐一检查，保存的 Sender Prompt 均只包含“Please provide a general plan to solve this task.”、任务目标与初始观察，没有额外的 ALFWorld 环境规则或合法动作模板。精确输入示例见 `sender_prompt_examples.json`；每题 metadata 与 Prompt 指纹见 `historical_audit.json`。该结论来自历史保存的输入，而非推测当前脚本就是历史脚本。训练缓存配置同样记录采样与 5000 tokens；本次逐题 Prompt 检查限于评测缓存。

**文本组使用同一次缓存的规划，而非另调 Sender。** 已逐题对照 140 份 Text 组 `message.json` 与 Sender metadata：计划文本一致，通信 token IDs 等于缓存的生成 token IDs 去除末尾 EOS（若存在）。消息记录的文件哈希也与原运行的 `complete.json` 一致。Matched 使用目标题隐状态，Mismatched 使用错配 donor 隐状态；Random 是构造的高斯向量，No-Comm 不注入消息，不能笼统称为五组都注入同一份 Sender 缓存。

**两版历史入口不在已检查 Git 历史中，但快照完整保留。** 搜索范围覆盖核查时本地 `main`、`refactor`、`step1` 及 remote-tracking refs 所指提交的全部祖先，固定搜索端点写入 `historical_audit.json`：

- Matched/No-Comm 与 Mismatched/Random 的入口 SHA-256 没有匹配到 Git blob；已将原运行目录中哈希一致的两版快照原样归档到 `sources/`。
- Text 组入口及各组配置记录的其他评测源码均找到匹配的 Git blob；报告记录匹配路径、blob ID 和一个确实包含该文件版本的代表性 commit。Text 入口也原样归档。
- 找到文件版本不等于证明当时运行在那个 commit 上；原始执行 commit 仍记为未知。Sender 配置未记录采集源码哈希，因此无法仅靠当前同名脚本确定历史采集入口的精确版本或原命令。

`sources/` 仅用于保存历史入口，不是当前运行入口，也不是可直接独立运行的完整代码包；它们仍依赖对应历史版本的其他模块。这里不修正历史 No-Comm 提示词、损失或解码行为，不改写旧配置，不重新运行实验。后续 Step 0 的环境规则、贪心及 512 tokens 修改应单独实施。

## 已核验与未核验

归档脚本核验五组各有 140 个不重复且一致的任务 ID、成功计数、任务清单与 Adapter checkpoint 指纹、评测 Sender 配置指纹、原评测间引用关系，以及错配映射覆盖和 donor 无自配的双射。保存到各运行目录的三份 `eval_receiver_swap.py` 快照均与运行配置记录的哈希一致。因此，未在此前 11 个提交中找到入口文件的完全匹配版本，不等于历史源码快照已经丢失；其本地路径和指纹已记入清单。

本次没有重放任务、全面核验所有 latent 张量或审查实验语义正确性。运行的精确 Git commit、原始完整 shell 命令、GPU 和运行环境锁定记录尚未据此确定；本地基准 `f566bc2` 不能冒充历史运行 commit。源码历史搜索不包含远端尚未 fetch 的对象或 dangling objects。当前结果也不能证明样本专属通信增益。

## 使用方式

在子仓库 `latent-communication/code/` 中执行只读核验（需要本地原始材料仍存在）：

```bash
uv run --no-sync python docs/runs/2026-11-14-receiver-swap/build_manifest.py --check
```

首次生成或有意重新生成时去掉 `--check`；该模式会覆盖本目录的生成文件，但不会改写 `data/` 中的历史材料。生成器不使用当前 Git HEAD 作为历史运行版本，也不修改旧记录来匹配现有代码。归档说明和脚本应与 JSON 文件一同纳入 Git；`.gitignore` 只对本目录及其 `configs/` 中的 JSON 放行。
