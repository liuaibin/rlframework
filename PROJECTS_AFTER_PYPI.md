# rlframework PyPI 发布后的项目结构

本文档说明 rlframework 发布到 PyPI 后，业务项目如何独立使用它。

---

## 整体布局

```
# rlframework（PyPI仓库）
pypi.org/
└── rlframework 0.2.0

# 项目A（独立仓库）
github.com/your-org/project-a-robotics/

# 项目B（独立仓库）
github.com/your-org/project-b-trading/

# 本地开发目录
~/workspace/
├── project_a_robotics/     # 克隆自仓库A
└── project_b_trading/      # 克隆自仓库B
```

---

## 发布流程

### 1. 准备 rlframework 发产

```bash
# 在 rlframework 仓库
cd rlframework

# 更新版本
vim pyproject.toml
# version = "0.2.0"

# 增加 build 依赖
pip install build twine

# 构建
python -m build

# 上传到 PyPI
twine upload dist/rlframework-0.2.0-py3-none-any.whl
twine upload dist/rlframework-0.2.0.tar.gz

# 验证
pip search rlframework  # 或在 pypi.org 上查看
```

现在 rlframework 可以通过以下方式安装：
```bash
pip install rlframework>=0.2.0
```

---

## 项目 A 的独立结构

项目 A 是完全独立的 GitHub 仓库。

### 目录结构

```
project-a-robotics/                    # GitHub 仓库名
├── src/
│   └── project_a/                    # Python 包名
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── robot_arm_env.py      # 机器人环境（项目特定）
│       │   └── sim_config.py         # 模拟配置
│       ├── algorithms/
│       │   ├── __init__.py
│       │   └── robot_ppo.py          # 扩展的PPO算法
│       ├── configs/
│       │   ├── __init__.py
│       │   ├── default.yaml          # 默认配置
│       │   ├── small_model.yaml      # 小模型配置
│       │   └── large_model.yaml      # 大模型配置
│       └── main.py                   # 训练入口
│
├── tests/
│   ├── __init__.py
│   ├── test_envs.py
│   ├── test_algorithms.py
│   └── conftest.py
│
├── notebooks/
│   ├── explore_env.ipynb             # 环境探索
│   └── analyze_results.ipynb         # 结果分析
│
├── data/                             # 本地数据（Git忽略）
│   ├── checkpoints/                  # 训练检查点
│   ├── datasets/                     # 训练数据
│   └── metrics/                      # 指标日志
│
├── docs/
│   ├── architecture.md               # 架构说明
│   ├── setup.md                      # 安装指南
│   └── training.md                   # 训练指南
│
├── scripts/
│   ├── setup_env.sh                  # 环境设置脚本
│   ├── train.sh                      # 训练脚本
│   └── evaluate.sh                   # 评估脚本
│
├── pyproject.toml                    # 项目元数据
├── Makefile                          # 开发任务
├── .gitignore                        # Git忽略规则
├── README.md                         # 项目说明
└── LICENSE
```

### pyproject.toml

```toml
[project]
name = "project-a-robotics"
version = "1.0.0"
description = "Robotic arm control using rlframework"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [{name = "Your Team"}]

dependencies = [
    "rlframework>=0.2.0",        # ✅ 从 PyPI 安装
    "pybullet>=3.1.0",           # 项目特定的依赖
    "numpy>=1.24.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
    "ruff>=0.4.0",
    "mypy>=1.10.0",
]
jupyter = [
    "jupyter>=1.0.0",
    "ipywidgets>=8.0.0",
]
all = [
    "project-a-robotics[dev,jupyter]",
]

[project.scripts]
project-a-train = "project_a.main:main"

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
include = ["project_a*"]
```

### Makefile

```makefile
.PHONY: venv install install-all lint format test clean help

VENV_DIR ?= .venv
VENV_PYTHON := $(VENV_DIR)/bin/python
VENV_PIP := $(VENV_DIR)/bin/pip

help:
	@echo "项目A - 机器人控制"
	@echo ""
	@echo "可用命令:"
	@echo "  install      创建虚拟环境并安装依赖"
	@echo "  train        运行训练脚本"
	@echo "  test         运行测试"
	@echo "  lint         代码检查"

venv:
	python3 -m venv $(VENV_DIR)

install: | venv
	$(VENV_PIP) install -e ".[dev]"
	@echo "✓ 项目A开发环境已安装"

train: install
	source $(VENV_DIR)/bin/activate && python src/project_a/main.py

test: install
	$(VENV_PYTHON) -m pytest tests/ -v

lint:
	$(VENV_PYTHON) -m ruff check src tests

format:
	$(VENV_PYTHON) -m ruff format src tests
```

### 使用方式

```bash
# 克隆并进入项目
git clone https://github.com/your-org/project-a-robotics.git
cd project-a-robotics

# 安装（自动安装 rlframework）
make install

# 激活虚拟环境（可选，通常被 Makefile 自动处理）
source .venv/bin/activate

# 运行训练
make train
# 或
python src/project_a/main.py

# 运行测试
make test
```

### main.py 示例

```python
"""文件：src/project_a/main.py"""

import os
import ray
import yaml
from pathlib import Path

# ✅ 导入 rlframework（从 PyPI 安装）
from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.callbacks import FrameworkCallback
from rlframework.observability.reporters import FileReporter
from rlframework.storage.checkpoint_manager import CheckpointManager

# 导入项目特定的模块
from project_a.envs.robot_arm_env import RobotArmEnv
from project_a.algorithms.robot_ppo import RobotPPO


def load_config(config_path: str = None):
    """加载配置文件"""
    if config_path is None:
        config_path = Path(__file__).parent / "configs" / "default.yaml"
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    # 初始化 Ray
    ray.init(ignore_reinit_error=True)
    
    # 加载配置
    cfg = load_config()
    
    # 创建数据目录
    data_dir = Path(__file__).parent.parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 创建报告器
    reporter = FileReporter(str(data_dir / "metrics.jsonl"))
    
    # 创建检查点管理器
    ckpt_mgr = CheckpointManager(
        save_dir=str(data_dir / "checkpoints"),
        keep_best=cfg["training"]["keep_best_checkpoints"]
    )
    
    # 配置算法（使用项目特定的 RobotPPO）
    config = (
        RobotPPO
        .get_default_config()
        .environment(
            RobotArmEnv,
            env_config=cfg["env"]
        )
        .framework("torch")
        .rollouts(num_rollout_workers=cfg["training"]["num_workers"])
        .training(
            lr=cfg["training"]["learning_rate"],
            gamma=cfg["training"]["gamma"],
        )
        .resources(num_gpus=cfg["resources"]["num_gpus"])
    )
    
    algo = config.build()
    
    # 训练循环
    for i in range(cfg["training"]["num_iterations"]):
        result = algo.train()
        reporter.write(result)
        ckpt_mgr.maybe_save(algo.get_policy(), result)
        
        if (i + 1) % 50 == 0:
            reward = result["env_runners"]["episode_reward_mean"]
            print(f"[项目A] 迭代 {i+1}: 奖励 = {reward:.2f}")
    
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
```

---

## 项目 B 的独立结构

项目 B 也是完全独立的 GitHub 仓库，结构类似但针对交易场景。

### 目录结构

```
project-b-trading/                    # GitHub 仓库名
├── src/
│   └── project_b/                   # Python 包名
│       ├── __init__.py
│       ├── envs/
│       │   ├── __init__.py
│       │   ├── market_env.py        # 市场环境
│       │   └── data_loader.py       # 数据加载器
│       ├── algorithms/
│       │   ├── __init__.py
│       │   └── trading_sac.py       # 交易SAC
│       ├── configs/
│       │   ├── __init__.py
│       │   ├── default.yaml
│       │   └── production.yaml
│       └── main.py
│
├── tests/
│   ├── test_envs.py
│   └── test_algorithms.py
│
├── data/                            # 本地数据
│   ├── checkpoints/
│   ├── market_data/                 # 市场历史数据
│   └── metrics/
│
├── scripts/
│   ├── download_data.py             # 下载市场数据
│   ├── backtest.py                  # 回测脚本
│   └── live_trading.py              # 实盘交易
│
├── pyproject.toml
├── Makefile
├── README.md
└── LICENSE
```

### pyproject.toml

```toml
[project]
name = "project-b-trading"
version = "0.5.0"
description = "Quantitative trading using rlframework"
requires-python = ">=3.10"

dependencies = [
    "rlframework>=0.2.0",        # ✅ 从 PyPI 安装
    "pandas>=2.0.0",
    "yfinance>=0.2.0",
    "backtrader>=1.9.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
]
```

### main.py 示例

```python
"""文件：src/project_b/main.py"""

import ray

# ✅ 导入 rlframework（从 PyPI 安装）
from rlframework.algorithms.sac import CustomSACConfig
from rlframework.observability.reporters import FileReporter
from rlframework.storage.checkpoint_manager import CheckpointManager

# 导入项目特定的模块
from project_b.envs.market_env import MarketEnv
from project_b.algorithms.trading_sac import TradingSAC


def main():
    ray.init(ignore_reinit_error=True)
    
    reporter = FileReporter("./data/trading_metrics.jsonl")
    ckpt_mgr = CheckpointManager(
        save_dir="./data/checkpoints",
        keep_best=3
    )
    
    # 使用 rlframework 的 SAC 算法
    config = (
        TradingSAC
        .get_default_config()
        .environment(
            MarketEnv,
            env_config={
                "lookback_window": 60,
                "initial_balance": 100000
            }
        )
        .framework("torch")
        .training(lr=3e-4)
    )
    
    algo = config.build()
    
    for i in range(300):
        result = algo.train()
        reporter.write(result)
        ckpt_mgr.maybe_save(algo.get_policy(), result)
        
        if (i + 1) % 30 == 0:
            print(f"[项目B] 迭代 {i+1}")
    
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
```

---

## 安装和使用流程

### 将 rlframework 发布到 PyPI

```bash
# 1. 在 rlframework 目录
cd rlframework
pip install build twine

# 2. 构建
python -m build

# 3. 上传
twine upload dist/*

# 4. 验证（任何地方）
pip install rlframework
python -c "import rlframework; print(rlframework.__version__)"
```

### 项目 A 使用 rlframework

```bash
# 1. 克隆项目 A
git clone https://github.com/your-org/project-a-robotics.git
cd project-a-robotics

# 2. 安装（会自动从 PyPI 安装 rlframework）
make install
# 等同于
# python -m venv .venv
# .venv/bin/pip install -e ".[dev]"
# 这会安装 rlframework>=0.2.0

# 3. 验证
source .venv/bin/activate
python -c "from rlframework.algorithms.ppo import CustomPPO; print('✓')"

# 4. 运行
make train
```

### 项目 B 使用 rlframework

```bash
# 完全相同的流程
git clone https://github.com/your-org/project-b-trading.git
cd project-b-trading

make install
source .venv/bin/activate

# 验证项目 B 也获得了最新的 rlframework
python -c "from rlframework.algorithms.sac import CustomSAC; print('✓')"

make train
```

---

## 关键要点

| 方面 | 项目 A | 项目 B | rlframework |
|------|--------|--------|-----------|
| **仓库** | 独立仓库A | 独立仓库B | PyPI 包 |
| **安装** | `make install` | `make install` | `pip install rlframework` |
| **位置** | `~/project_a_robotics/` | `~/project_b_trading/` | `.venv/lib/pythonX.X/site-packages/rlframework/` |
| **更新** | 从仓库A拉取更新 | 从仓库B拉取更新 | `pip install --upgrade rlframework` |
| **版本控制** | 项目A的 `pyproject.toml` | 项目B的 `pyproject.toml` | 由 PyPI 管理 |

---

## 最新的 rlframework 版本更新流程

当 rlframework 发布新版本（0.3.0）时：

```bash
# 1. rlframework 更新
cd rlframework
# 编辑 pyproject.toml: version = "0.3.0"
python -m build
twine upload dist/*

# 2. 项目 A 更新（可选，取决于配置）
cd ../project_a_robotics
# 更新 pyproject.toml 的依赖
# dependencies = ["rlframework>=0.3.0"]
make clean-venv
make install  # 安装新版本

# 3. 项目 B 也可以更新
cd ../project_b_trading
# 同上
```

---

## 总结：PyPI 模式的优势

✅ **完全独立** — 项目A、B 互不影响
✅ **版本明确** — 每个项目指定 rlframework 版本
✅ **易于更新** — pip install --upgrade
✅ **团队合作** — 不同团队可独立维护项目
✅ **开源友好** — 可以公开发布项目

这是**生产级的最佳实践**！
