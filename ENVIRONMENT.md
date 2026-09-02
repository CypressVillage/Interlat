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

- **`libcudnn.so.9` / `libnccl.so.2` 导入错误**：`site-packages/nvidia/` 是所有 `nvidia-*` 包共享的命名空间，**切勿混装不同 CUDA 大版本的 nvidia 包**（如 cu13），卸载时会连带删除同名文件。修复：`uv sync --frozen --reinstall-package nvidia-cudnn-cu12 --reinstall-package nvidia-nccl-cu12`
- **textworld 构建失败**：确认 gcc/make 已装；确认 `vendor/textworld-1.6.1/textworld/thirdparty/I7_6M62_Linux_all.tar.gz` 存在，且 `thirdparty/` 下**没有**残缺的 `inform7-6M62/` 存根目录（PyPI sdist 自带残缺存根，会骗过 setup.sh 的已安装检查——按上文重建步骤操作即可避免）
- **环境彻底损坏**：直接 `rm -rf .venv && uv sync --frozen`，几分钟内按锁重建（uv 缓存加速）
