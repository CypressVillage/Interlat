# Interlat 第一阶段冻结 Receiver 变体：对齐核验

用户于 2026-10-25 选择“先完整对齐再训练”。这是准备与采用记录，不是已完成
训练或导师确认的新协议。已有 CE-only 诊断保留在原目录。

## 已定位来源

- 论文：<https://arxiv.org/html/2511.09149v5>，重点 §3.2、§5、附录 H。
- Interlat 公开代码参考 commit：`66a89cb4d4097b2f86cbe48ed9851d6e8578f821`。
- ETO 官方仓库：`Yifan-Song793/ETO`，核验 revision
  `a2fc5da38f8d00cfaf3f9b6370d586eebaf72904`；README 指向
  `agent-eto/eto-sft-trajectory`。
- ETO 数据 revision：`a8f745ef799a5201d2288b9c2a47e40a0ff70a4c`，
  `data/alfworld_sft.json`，下载经 hf-mirror.com，SHA-256
  `9bc9720466cdbae9da26f60ab61ccdd34ae2d1e8c9f1b2ca5a1acb6606f66972`。
- Base Receiver：`Qwen/Qwen2.5-7B`，模型服务镜像返回 revision
  `d149729398750b98c0af14eb82c78cfe92750796`。下载完成，四个 safetensors
  分片均可打开、tensor keys 与索引完全匹配，已保存各分片 SHA-256。
  Config/tokenizer 可离线加载；尚未做 Base GPU 前向核验。记录见
  `data/interlat_alignment_20261025/base_verification.json`。

## 数据的实际发现

数据卡登记 3,118 条，但下载的原始 JSON 实际有 3,119 条，ID 与 game 路径均唯一。
其中 3,118 条可匹配先前 canonical 清单（原 train-fit 2,823 / selection 295），
另有 `alfworld_1176` 不在 canonical 清单；对应游戏文件实际存在。
该轨迹是清洁布料任务，原文没有执行 clean 动作，不能直接认定它是成功专家轨迹。
正在全部重放以核验当前环境中的成功终止与输入对应关系，不静默删除记录。

这些 2,823 / 295 只是沿旧划分做交集核查，不是新实验采用的数据划分。
论文的 5% 验证比例和已确认 Step 1 的 332 条 selection 不相同，必须明确登记。
原数据包括 human 指令、assistant `OK`、human 初始观察与目标，再接多轮
Thought/Action/Observation；不能无记录地替换成旧诊断的模板化 Thoughts。
引用 ETO 数据不等于采用 ETO 的 DPO：当前要对齐的是 Interlat 的监督训练。

## 上一轮说明的纠正与待明确项

1. 论文混合公式是 **text prefix + latent suffix**，采样
   `r ∈ {0,0.1,...,1}`，并非仅 0.1..0.9。
   上游 `adaptive_mix_with_balanced_distribution` 实际为 **latent prefix + text suffix**，
   主调用采样 0.1..0.9，使用 round 且分别按两边长度截取。二者不等价。
   以论文对齐时应明确采用论文公式，不能称代码实现与之相同。
2. 论文分离项是负 JS；公开代码计算带 0.69 margin 的项，且
   `F.kl_div(p.log(), m)` 表示 KL(m||p)，不是通常 JS 中的 KL(p||m)。
   不能直接复制函数并声称实现论文公式。
3. 论文对齐项写 KL(latent||text)，附加 normalized-logit cosine；
   公开实现使用 `kl_div(log_softmax(latent), softmax(text))`，方向为
   KL(text||latent)，cosine 也在 softmax 概率上计算。
4. 上游对分离/对齐 logits 按标签位置切片，没有像任务 causal CE 一样显式左移
   一位；重实现必须按同一被预测 response token 对齐，不按注入后绝对位置配对。
5. 论文说明权重动态退火范围，但目前没有核验到足以逐步复现的完整 schedule；
   上游动态权重范围也不同。alpha/beta 的归并方式和负分支 stop-gradient 选择
   必须明确。论文主公式与上游随机分支 no_grad 不能未经说明混为同一梯度目标。
6. 约 2.2k steps 是一条训练现象，不是论文规定的训练预算。
   不能据此承诺训练到该点就会有效，也不能直接当作停止标准。
7. 冻结参数不代表梯度信号必然“更弱”；它限制可更新函数族，但梯度幅度与
   有效性需测量。此前“梯度信号比原论文弱”的表述应撤回。
8. 原 Sender prompt 见 `data_collection/alfworld_collection.py:290-307`：
   `Please provide a general plan to solve this task...`。
   应继续核查 task 字段抽取和实际解码设置，再重建 latent；不沿用包含一步
   动作指令的旧消息缓存，也不另加自创计划格式后声称完全对齐。

## 当前执行状态

### 2026-10-26 预处理与目标核验

Sender原提示词六类fit样本探针已完成（greedy、512-token cap，属于明确登记的
诊断解码设置，不代表已核实论文完整采样配置）。输出长度依次为482、286、
512、512、414、423；四条EOS结束，clean/cool两条达到上限。clean样本
`alfworld_2281` 出现重复标题、异常片段和数字串；cool样本主要建议自然冷却，
不能据此认为掌握了ALFWorld的冷却操作。heat样本提到了微波炉，但还引入
设置时间、功率等环境不支持的操作。因此原prompt产生更长多步文字，但
不能据此认定消息质量提高或已适合完整训练。此探针尚无adapter参与，异常
与之前Receiver注入后的乱码是不同路径，不能混为同一原因。
原文、token ID、EOS状态和逐词元latent保存在
`data/interlat_alignment_20261026/sender_probe/probe.json`及对应`.pt`。
不选择性删除失败探针，也不直接用这些六条结果代替完整数据集。

本地 HEAD 的 `core_training/data_processor.py` 与
`core_training/hidden_state_loader.py` 相对锁定上游 commit 无差异，已用
`git show`/`git diff` 核验。前者 543–565 行直接读取 JSON、合并第0/2条
human 消息、添加计划引导、删除 OK/任务中间交换；没有按 won/reward 过滤，
也没有动作名称替换。loader 只丢弃无法转换为 tensor 的 latent 行，缺失任务
ID 查询会报错。不能从这些代码推断作者在传入 JSON 前是否另有清洗。

已实现 `prepare_paper_data.py`，保留全部3,119条原文和游戏映射，附带重放质量
标记。采用论文5%比例、上游seed42 shuffle和floor取整：fit 2,964条、selection
155条；不是旧Step1划分，也不能声称复原了论文未公开的精确split。
文件在 `data/interlat_alignment_20261026/prepared/`，manifest包含SHA-256。
单元检查确认Sender只收到初始任务，不含未来专家答案，动作原文未改写。

Base Receiver GPU前向和对输入梯度检查通过，base可训练参数为0；这不是任务
能力检查，见 `base_gpu_check.json`。GB10平台既有环境差异仍然存在。

已加入独立 `paper_objectives.py`：论文方向的负JS、KL(latent||text)、
逐监督位置logit cosine及text-prefix/latent-suffix混合。测试使用可手算的
二元分布检查KL方向、JS数值和双分支梯度，并检查混合位置和端点梯度。
这些只是目标原语，还未接入端到端训练；没有假设未核实的系数schedule。
新训练仍需：重新生成位置对齐的Sender计划/latent、实现Base对话注入与三路
causal logits对齐、明确辅助系数/训练预算等复现选择，再做最小训练核验。

### 首轮轨迹重放观察

### 归一化复核结果（已完成）

`replay_normalized/summary.json`：3,119 条初始观察全部在空白归一化后匹配，
零执行异常；成功仍为 2,362、失败 757。中间观察无差异的有 1,809 条：
其中成功 1,644、失败 165；存在中间文字差异的有 1,310 条：成功 718、
失败 592。差异不全是环境状态不同，例如 alarm clock/alarmclock、CD/cd
属于词形改写；但同类改写若出现在动作中可实际导致拒识，如 alfworld_9 的
`go to dining table 1` 在当前环境返回 `Nothing happens.`。
无观察差异的 165 条失败说明不能把全部失败归因于版本/命名问题。
以上统计未额外限制20步，按原始轨迹执行至结束或环境终止，不是模型评估。

这次核验确认公开 ETO 文件来源和当前环境对应关系，尚不能证明该文件未经
进一步预处理就是 Interlat 论文实际使用的最终训练文件。按原分布训练与
按当前环境成功过滤为两个不同研究选择；不得静默过滤或改写动作。

3,119 条全部执行完成，无异常；2,362 条当前环境判成功，757 条未成功。
首轮 `initial_matches=0` 和 observation mismatches 不能直接解释为数据不一致：
审计比较未归一化空白，也未应用 evaluator 的 `process_ob` 去除位置前缀。
已修正审计比较并启动 `replay_normalized/`，保留旧目录，不修改任何动作。
初始观察和中间反馈是否一致应以修订审计结果为准。

有直接可核验的失败例：`alfworld_6` 要求加热杯子放入冰箱，九个动作只有
寻找/拿取杯子、开关微波炉、进入并打开冰箱、放下杯子，没有 heat 动作。
逐步实际反馈经 process_ob 与原文一致，结束时 won=False；因此不能把
全部 757 条失败先归因于环境版本差异，也不能把官方发布数据全部标为成功轨迹。
是否保留原数据以复现其训练分布、还是过滤以满足旧 Step 1 成功轨迹条件，
会改变实验含义；在核验完成前不静默选择，也不以手写专家补齐冒充原数据。

- 原始 ETO 数据已下载，映射审计见 `data/interlat_alignment_20261025/mapping_audit.json`。
- Base 权重下载与 3,119 条原轨迹重放已启动；输出均存忽略目录。
- Base 权重下载和文件结构核验已完成，轨迹重放仍由独立任务处理。
- 已核验 Sender 的 task 来自原始 `conversations[2]['value']`
  (`data_collection/alfworld_collection.py:322`)，即初始观察和任务目标，
  不包含后续专家答案。旧诊断额外拼接交互指令不属于该原始 prompt。
- 尚未启动新训练。原公式/代码不一致和未公开精确配置需逐项解决；无法核实的
  项目必须明确为复现选择，不能声称除冻结外全部相同。

### 2026-10-26 用户 Goal 授权后的独立 v1 实验

前文是此前核验的时间点记录。当前新增执行规范见
[`FROZEN_RECEIVER_V1.md`](FROZEN_RECEIVER_V1.md)，不改写旧 Step 1A 协议。
按论文 Appendix H 将消息注入到首个 user 内部、结束符之前；原 Step 1A 的
结束符后注入路径保留。Sender 采用公开采集器采样默认值而非六样本探针设置，
完整3,119条采集已启动，位置对齐的文本/token ID/bf16 latent均独立存储。
公开采集器虽然计算了修订的 `hidden_seq`，`infer_chain` 实际仍保存旧的
`hidden_state`；本版采用论文定义的预测前 hidden，并显式记录此差异。

新增三路目标、冻结 Base runtime、预算训练器和成对评估器；CPU数值/因果位置
检查已通过，端到端 GPU 烟测已完成一次更新，完整Base参数SHA-256前后不变
（`data/interlat_frozen_receiver_v1_20261026/early_smoke/summary.json`）。正式训练预算为seed0、至多3轮/558次
更新或24小时优化时间；后续以155条selection NLL选点，再评估140条canonical
valid_seen，valid_unseen不使用。当前不得将已写代码或采集中状态称为训练完成。

缓存核验保留了一次完整前向与增量解码的BF16数值差异记录
（`cache_probe_audit.json`）；原批次强制逐token重放的3,829个向量全部逐位一致
（`cache_exact_replay.json`）。后者支持该批次的预测位置提取正确，不把前者
失败的近似阈值检查静默删除。完整采集结束后的审计、训练、评估执行链已启动，
通过进程退出通知衔接，日志为`data/interlat_frozen_receiver_v1_20261026/pipeline.log`。

正式训练前又核验了text teacher的结尾控制符：公开采集器保存的是
`decode(..., skip_special_tokens=True)`的计划文字，训练端用
`tokenizer(plan, add_special_tokens=False)`。初版新runtime直接用了含终止EOS的
token ID，与该文本内容路径不一致。现仅在通信view中同步去掉终止EOS及其预测
hidden，采样缓存原样保留；达到token上限而无EOS的计划不裁切。新增回归检查
确认对应位置和缓存不可变。原EOS-inclusive烟测保留，内容边界修订后的烟测
另存`content_smoke/`及`content_evaluator_smoke.json`；不把两版结果混为一版训练。

本机无 `gh`，匿名读取私有 Issue API 返回404，未重新读取远端完整线程；本地
2026-10-08 Exchange已含导师全量训练节奏回复，本次变更的授权来自用户Goal，
不是从远端线程推断。仅添加本地会话Issue关联，没有发布评论或改变状态。
