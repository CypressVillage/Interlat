# 环境说明

本项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境，`uv.lock` 锁定全部依赖版本，可精确复现。

## 环境概览

| 项 | 值 |
|---|---|
| 管理工具 | uv（建议 >= 0.12） |
| Python | 3.10（uv 托管解释器，见 `.python-version`） |
| 虚拟环境 | `.venv/`（项目根目录） |
| 锁文件 | `uv.lock`（174 个包） |
| torch | 2.4.1+cu121（PyPI 官方轮子，CUDA 运行时由 pip 包自带，**系统无需安装 CUDA Toolkit**） |
| 关键依赖 | torchvision 0.19.1 / transformers 4.46.3 / numpy 1.24.4 / alfworld 0.3.5 / textworld 1.6.1 |
| 已验证驱动 | NVIDIA 595（cu121 要求驱动 >= 530.30） |

textworld 1.6.1 来自本地源码 `vendor/textworld-1.6.1/`（PyPI sdist 的 `setup.sh` 构建时需要从外网下载 Inform7，`vendor` 中已预置 `I7_6M62_Linux_all.tar.gz`，离线也能构建）。

`vendor/` 体积大（~127MB，含 Inform7 离线包与构建产物），**不入 git**（已在 `.gitignore` 排除），但 `pyproject.toml` 以 path 依赖引用它，缺失时 `uv sync` 会失败。fresh clone 后需先按下一节重建。

## 重建 vendor/textworld-1.6.1（fresh clone 后必须做一次）

```bash
mkdir -p vendor && cd vendor
# 1. 从 PyPI 下载 sdist 并解压（任意 pip 均可；也可从 PyPI 项目页手动下载 textworld-1.6.1.tar.gz）
pip download textworld==1.6.1 --no-deps --no-binary :all: -d .
tar xf textworld-1.6.1.tar.gz && rm textworld-1.6.1.tar.gz

# 2. 删除 sdist 自带的残缺 inform7-6M62/ 存根目录（会骗过 setup.sh 的已安装检查，导致构建失败）
rm -rf textworld-1.6.1/textworld/thirdparty/inform7-6M62/

# 3. 预置 Inform7 离线包（可选；不预置则 setup.sh 构建时联网下载）
curl -LO http://emshort.com/inform-app-archive/6M62/I7_6M62_Linux_all.tar.gz
mv I7_6M62_Linux_all.tar.gz textworld-1.6.1/textworld/thirdparty/

cd ..
# 4. 回到项目根目录后正常执行 uv sync --frozen
```

- Inform7 包校验（sha256）：`684e33d37e6fd21a1822233ddf35937f3a365c4a366486a113c5f32015d93cbd`
- 手头有旧机器时，直接 `rsync`/`scp` 整个 `vendor/` 过来也可以（构建产物一并带走无需重新编译）。

## 完整复现步骤

```bash
# 0. 系统前置：NVIDIA 驱动 + gcc/make（textworld 源码编译 C 扩展用）+ curl
#    Ubuntu: sudo apt install build-essential

# 1. 安装 uv（已有可跳过）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 一键创建环境（自动：安装托管 CPython 3.10 → 创建 .venv → 按 uv.lock 精确安装 → 可编辑安装本项目并从 vendor 源码编译 textworld）
#    fresh clone 需先完成上文「重建 vendor/textworld-1.6.1」
uv sync --frozen

# 3. 验证
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
uv run python -c "import textworld, alfworld.agents; print('textworld/alfworld OK')"
uv run torch-gpu-smi 2>/dev/null || uv run python -c "import torch; x=torch.randn(64,64).cuda(); print('GPU 计算 OK', (x@x).shape)"
```

装开发工具：`uv sync --extra dev`；装文档工具：`uv sync --extra docs`。

## 依赖管理规则

- **改依赖**：`uv add <pkg>` / `uv remove <pkg>`（自动更新 `pyproject.toml` 和 `uv.lock`），然后用 `uv sync --frozen` 应用
- **还原环境**：`uv sync --frozen`（会删除锁外的多余包，保证与锁完全一致）
- **版本护栏**：`pyproject.toml` 的 `[tool.uv] constraint-dependencies` 保存了 112 条实测可用的版本约束（numpy 1.24.x、transformers 4.46.3 等），普通 `uv lock --upgrade` 也不会越过它们；如需主动升级某个包，先评估并同步修改对应约束

## pyproject/uv.lock 管不了的部分

| 内容 | 原因 | 处理方式 |
|---|---|---|
| flash-attn | 构建需要系统级 nvcc（CUDA toolkit） | 默认**不安装**（在 `full` extra 中，锁文件仅记录版本）。需要时：① 优先用官方预编译 wheel，按 torch/python/cuda 版本从 [flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases) 下载后 `uv pip install <wheel>`；② 或装好 nvcc 后 `uv pip install flash-attn --no-build-isolation` |
| NVIDIA 驱动 | 系统层，包管理无法触达 | 宿主机安装驱动即可（>= 530.30），CUDA 运行时走 pip 包 |
| alfworld 游戏数据 | 包安装后的数据下载，uv 无 post-install 钩子 | 手动执行 alfworld 的数据下载脚本（历史日志见 `alfworld-data-download.log`），或将其纳入部署脚本 |
| 系统工具链 gcc/make | 系统包 | `apt install build-essential` |

需要强一致的系统级环境（驱动 + 工具链 + Python）时，建议补一个基于 `nvidia/cuda:12.1-devel` 或 `pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel` 的 Dockerfile，容器内只需 `uv sync --frozen`。

## 故障排查

### GB10 ARM 诊断环境（2026-10-23，与 A100 锁定环境不同）

以下为历史环境诊断记录。2026-11-03 按用户要求删除了本地旧实验产物，
包括 `data/gb10_env_probe/`；文中旧日志路径不表示文件仍存在。环境本身与模型缓存保留。
当时 GPU 检查记录位于 `data/gb10_env_probe/cuda_smoke_result.json`。
GB10 使用现有 `.venv` 中的 `torch==2.9.1+cu128`，不是上文 x86/A100
使用的 torch 2.4.1。不能把本机诊断宣称为锁定环境的正式复现，也不要直接
`uv sync` 覆盖此环境；原 `pyproject.toml` / `uv.lock` 未为此改写。
本机执行命令使用 `uv run --no-sync python ...`，仍由 uv 管理的 Python 3.10 运行。

首次准备时 `vendor/` 不存在，普通 `uv run` 解析本地依赖失败。恢复
TextWorld 1.6.1 源码后发现其 Inform7 6M62 仅提供 x86 编译器，不提供
`aarch64` 包。本机 vendor 的 `setup.sh` 仅对 ARM 跳过编译器/解释器二进制
解包，保留公共资源并原生构建 Glulx；只用于官方预生成 `.tw-pddl` 的 PDDL
环境，不支持 Inform7 游戏编译。此 patch 位于被忽略的 `vendor/` 中。
不要通过装载 x86 二进制假称 ARM 编译支持。

补装依赖时保留现有 torch/CUDA 包。torchvision 使用 PyTorch cu128 索引的
0.24.1 ARM wheel（`uv pip install --no-deps torchvision==0.24.1 --index-url
https://download.pytorch.org/whl/cu128`）；模型侧保持 transformers 4.46.3、
tokenizers 0.20.3、numpy 1.24.4，环境侧保持 alfworld 0.3.5、TextWorld 1.6.1。
OpenCV 使用 4.11.0.86，避免新版本强制升级 NumPy。
完整已安装版本另存各次运行的 `requirements.installed.txt`；该清单是环境
记录，不是 ARM 平台已经验证的完整锁文件。

本次 `uv pip check` 还发现现有 `nvidia-cusparselt-cu12==0.7.1` 包的
平台标记与 ARM 不符；为避免训练过程中更换 CUDA 库，未覆盖该既有包。
已补齐缺少的 prompt-toolkit 3.0.52。普通 CUDA 矩阵乘法及先前的
bf16/SDPA/AdamW 检查通过，但这些不能证明稀疏 CUDA 功能或整个环境合格。
torch 还会提示设备 capability 12.1 超出其声明的 12.0 上限；这些警告
需保留，正式运行前应单独修复和核验平台环境。

ALFWorld 数据来自官方 release 0.2.2 的 `json_2.1.1_json.zip` 与
`json_2.1.1_tw-pddl.zip`，存放于 `data/step1_05b_20261023/alfworld/`。
游戏数据已于 2026-11-03 迁至 `data/alfworld/json_2.1.1/`，旧下载压缩包已删除。
仅枚举 reserved split 不等于运行其任务；本次不运行 `valid_unseen`。

### 常见问题

- **`libcudnn.so.9` / `libnccl.so.2` 导入错误**：`site-packages/nvidia/` 是所有 `nvidia-*` 包共享的命名空间，**切勿混装不同 CUDA 大版本的 nvidia 包**（如 cu13），卸载时会连带删除同名文件。修复：`uv sync --frozen --reinstall-package nvidia-cudnn-cu12 --reinstall-package nvidia-nccl-cu12`
- **textworld 构建失败**：确认 gcc/make 已装；确认 `vendor/textworld-1.6.1/textworld/thirdparty/I7_6M62_Linux_all.tar.gz` 存在，且 `thirdparty/` 下**没有**残缺的 `inform7-6M62/` 存根目录（PyPI sdist 自带残缺存根，会骗过 setup.sh 的已安装检查——按上文重建步骤操作即可避免）
- **环境彻底损坏**：直接 `rm -rf .venv && uv sync --frozen`，几分钟内按锁重建（uv 缓存加速）
