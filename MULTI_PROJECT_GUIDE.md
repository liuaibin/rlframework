# 使用 rlframework 编排多个业务项目

本文档说明如何使用 rlframework 作为共享框架来支持多个强化学习业务项目。

## 整体架构

```
monorepo/
├── rlframework/              # 共享框架（核心库）
│   ├── algorithms/
│   ├── config/
│   ├── logging/
│   ├── storage/
│   ├── utils/
│   ├── pyproject.toml
│   └── README.md
│
├── projects/                 # 业务项目目录
│   ├── project_a_robotics/   # 项目A：机器人控制
│   ├── project_b_trading/    # 项目B：量化交易
│   ├── project_c_gaming/     # 项目C：游戏 AI
│   └── ...
│
├── shared/                   # 共享代码（可选）
│   ├── custom_envs/         # 自定义环境库
│   ├── custom_models/       # 自定义模型库
│   └── dataset/             # 共享数据集
│
└── README.md
```

---

## 方案 1：通过 pip 发布（推荐用于独立团队）

### 发布流程

#### 1. 准备 rlframework 发布

```bash
# 在 rlframework 目录
cd rlframework

# 增加版本号
# 编辑 pyproject.toml: version = "0.2.0"

# 构建包
python -m build

# 上传到 PyPI（或内部 PyPI）
twine upload dist/*
```

#### 2. 项目 A 使用 rlframework

```bash
# 创建项目目录
mkdir project_a_robotics
cd project_a_robotics

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate

# 创建自己的 pyproject.toml
cat > pyproject.toml << 'EOF'
[project]
name = "project-a-robotics"
version = "1.0.0"
description = "Robotics control using rlframework"
requires-python = ">=3.10"

dependencies = [
    "rlframework>=0.2.0",  # 依赖共享框架
    "pybullet>=3.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
]
EOF

# 安装
pip install -e ".[dev]"
```

#### 3. 项目 B 使用 rlframework

```bash
mkdir project_b_trading
cd project_b_trading

cat > pyproject.toml << 'EOF'
[project]
name = "project-b-trading"
version = "1.0.0"
description = "Quantitative trading using rlframework"
requires-python = ">=3.10"

dependencies = [
    "rlframework>=0.2.0",  # 同一个框架版本
    "pandas>=2.0.0",
    "yfinance>=0.2.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
]
EOF

pip install -e ".[dev]"
```

---

## 方案 2：Monorepo + Git 子模块（推荐用于同一团队）

### 项目结构

```bash
# 初始化主仓库
git init monorepo
cd monorepo

# 添加 rlframework 作为子模块
git submodule add https://github.com/your-org/rlframework.git rlframework

# 创建项目目录
mkdir projects
mkdir shared
```

### 项目 A 的结构

```
projects/project_a_robotics/
├── src/
│   └── project_a/
│       ├── __init__.py
│       ├── envs/           # 项目特定的环境
│       │   ├── robot_env.py
│       │   └── __init__.py
│       ├── algorithms/      # 项目特定的算法扩展
│       │   ├── robot_ppo.py
│       │   └── __init__.py
│       ├── configs/        # 项目特定的配置
│       │   ├── default.yaml
│       │   └── __init__.py
│       └── main.py         # 项目入口
│
├── tests/
│   ├── test_envs.py
│   └── test_algorithms.py
│
├── data/                   # 项目特定的数据
│   ├── checkpoints/
│   └── datasets/
│
├── pyproject.toml
├── Makefile
└── README.md
```

### 在 Monorepo 中安装依赖

```bash
# 从项目 A 安装 rlframework（作为本地路径依赖）
cd projects/project_a_robotics

cat > pyproject.toml << 'EOF'
[project]
name = "project-a-robotics"
version = "1.0.0"

dependencies = [
    # 使用本地路径指向共享框架
    "rlframework @ file://../../rlframework",
    "pybullet>=3.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-cov>=5.0.0",
]
EOF

pip install -e ".[dev]"
```

---

## 实战示例：项目 A（机器人）和项目 B（交易）

### 项目 A：机器人控制

#### **文件：projects/project_a_robotics/src/project_a/envs/robot_env.py**

```python
"""项目A特定的机器人环境"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


class RobotArmEnv(gym.Env):
    """7自由度机械臂控制环境"""
    
    def __init__(self, config=None):
        config = config or {}
        self.num_joints = config.get("num_joints", 7)
        self.simulation_step = config.get("simulation_step", 0.001)
        
        # 观察：关节角度 + 末端位置 + 目标位置
        obs_dim = self.num_joints * 2 + 6  # 角度、速度、位置
        self.observation_space = Box(-np.pi, np.pi, shape=(obs_dim,), dtype=np.float32)
        
        # 动作：每个关节的扭矩
        self.action_space = Box(-1.0, 1.0, shape=(self.num_joints,), dtype=np.float32)
    
    def reset(self, *, seed=None, options=None):
        self.joint_angles = np.zeros(self.num_joints, dtype=np.float32)
        self.joint_velocities = np.zeros(self.num_joints, dtype=np.float32)
        self.target_pos = np.random.uniform(-1, 1, size=3).astype(np.float32)
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        # 模拟物理步骤
        self.joint_velocities += action * self.simulation_step
        self.joint_angles += self.joint_velocities * self.simulation_step
        
        obs = self._get_observation()
        
        # 计算奖励：到目标的距离
        current_pos = self._fk(self.joint_angles)  # 正向运动学
        distance = np.linalg.norm(current_pos - self.target_pos)
        reward = -distance  # 越接近目标越好
        
        terminated = distance < 0.1  # 接近目标
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        current_pos = self._fk(self.joint_angles)
        obs = np.concatenate([
            self.joint_angles,
            self.joint_velocities,
            current_pos,
            self.target_pos
        ])
        return obs.astype(np.float32)
    
    def _fk(self, angles):
        """简化的正向运动学"""
        return np.sum(np.sin(angles[:3])), np.sum(np.cos(angles[:3])), angles[6]
```

#### **文件：projects/project_a_robotics/src/project_a/algorithms/robot_ppo.py**

```python
"""项目A特定的PPO算法扩展"""

from rlframework.algorithms.ppo import CustomPPO


class RobotPPO(CustomPPO):
    """针对机器人控制优化的PPO"""

    def on_after_training_step(self, result):
        """添加自定义指标"""
```

#### **文件：projects/project_a_robotics/src/project_a/main.py**

```python
"""项目A的训练主程序"""

import os
import ray
from project_a.envs.robot_env import RobotArmEnv
from project_a.algorithms.robot_ppo import RobotPPO
from rlframework.logging.reporters import FileReporter
from rlframework.storage.checkpoint_manager import CheckpointManager


def main():
    # 初始化
    ray.init(ignore_reinit_error=True)
    
    # 项目A特定的配置
    config = (
        RobotPPO
        .get_default_config()
        .environment(
            RobotArmEnv,
            env_config={
                "num_joints": 7,
                "simulation_step": 0.001
            }
        )
        .framework("torch")
        .rollouts(num_rollout_workers=4)
        .training(
            lr=1e-4,
            gamma=0.99,
        )
        .resources(num_gpus=1)  # 项目A需要GPU优化
    )
    
    # 项目A特定的存储路径
    reporter = FileReporter("./data/metrics.jsonl")
    ckpt_mgr = CheckpointManager(
        save_dir="./data/checkpoints",
        keep_best=5
    )
    
    algo = config.build()
    
    for i in range(500):
        result = algo.train()
        reporter.write(result)
        ckpt_mgr.maybe_save(algo.get_policy(), result)
        
        if (i + 1) % 50 == 0:
            reward = result["env_runners"]["episode_reward_mean"]
            print(f"[项目A] 迭代 {i+1}: 平均奖励 = {reward:.2f}")
    
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
```

---

### 项目 B：量化交易

#### **文件：projects/project_b_trading/src/project_b/envs/trading_env.py**

```python
"""项目B特定的交易环境"""

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete


class TradingEnv(gym.Env):
    """股票交易环境"""
    
    def __init__(self, config=None):
        config = config or {}
        self.lookback_window = config.get("lookback_window", 60)
        self.initial_balance = config.get("initial_balance", 100000)
        
        # 观察：价格历史 + 技术指标（MACD, RSI等）
        obs_dim = self.lookback_window + 10  # 价格 + 指标
        self.observation_space = Box(0, 1, shape=(obs_dim,), dtype=np.float32)
        
        # 动作：0=卖, 1=持, 2=买
        self.action_space = Discrete(3)
    
    def reset(self, *, seed=None, options=None):
        self.balance = self.initial_balance
        self.position = 0  # 持仓量
        self.price_history = np.random.uniform(100, 200, self.lookback_window).astype(np.float32)
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        current_price = self.price_history[-1]
        
        # 执行交易
        if action == 0:  # 卖
            self.balance += self.position * current_price
            self.position = 0
        elif action == 2:  # 买
            shares = self.balance / current_price
            self.position = shares
            self.balance = 0
        # action == 1：持，什么都不做
        
        # 更新价格
        new_price = current_price * np.exp(np.random.normal(0, 0.01))
        self.price_history = np.append(self.price_history, new_price)[-self.lookback_window:]
        
        # 重新计算余额
        total_value = self.balance + self.position * self.price_history[-1]
        reward = (total_value - self.initial_balance) / self.initial_balance
        
        obs = self._get_observation()
        terminated = False  # 交易环境通常不终止
        truncated = False
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        """包含技术指标的观察"""
        normalized_prices = (self.price_history - self.price_history.min()) / (self.price_history.max() - self.price_history.min() + 1e-8)
        # 简化版：只用价格，实际应该加入MACD, RSI等
        return normalized_prices.astype(np.float32)
```

#### **文件：projects/project_b_trading/src/project_b/algorithms/trading_sac.py**

```python
"""项目B选择SAC算法更适合连续交易"""

from rlframework.algorithms.sac import CustomSAC


class TradingSAC(CustomSAC):
    """针对交易优化的SAC算法"""

    def on_after_training_step(self, result):
        """交易后处理"""

#### **文件：projects/project_b_trading/src/project_b/main.py**

```python
"""项目B的训练主程序"""

import os
import ray
from project_b.envs.trading_env import TradingEnv
from project_b.algorithms.trading_sac import TradingSAC
from rlframework.logging.reporters import FileReporter
from rlframework.storage.checkpoint_manager import CheckpointManager


def main():
    ray.init(ignore_reinit_error=True)
    
    # 项目B特定的配置
    config = (
        TradingSAC
        .get_default_config()
        .environment(
            TradingEnv,
            env_config={
                "lookback_window": 60,
                "initial_balance": 100000
            }
        )
        .framework("torch")
        .rollouts(num_rollout_workers=2)  # 交易需要更少的workers（时间序列敏感）
        .training(
            lr=3e-4,
            gamma=0.99,
        )
        .resources(num_gpus=0)  # 项目B可以用CPU
    )
    
    reporter = FileReporter("./data/trading_metrics.jsonl")
    ckpt_mgr = CheckpointManager(
        save_dir="./data/checkpoints",
        keep_best=3
    )
    
    algo = config.build()
    
    for i in range(300):
        result = algo.train()
        reporter.write(result)
        ckpt_mgr.maybe_save(algo.get_policy(), result)
        
        if (i + 1) % 30 == 0:
            reward = result["env_runners"]["episode_reward_mean"]
            print(f"[项目B] 迭代 {i+1}: 平均回报率 = {reward:.4f}")
    
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
```

---

## 共享组件

### 共享自定义环境库

```
shared/custom_envs/
├── __init__.py
├── common.py          # 通用环境基类
├── wrappers.py        # 通用环境包装器
└── components.py      # 通用环境组件
```

#### **文件：shared/custom_envs/common.py**

```python
"""多个项目可共享的环境基类"""

import gymnasium as gym
import numpy as np


class BaseControlEnv(gym.Env):
    """通用控制环境基类"""
    
    def __init__(self, config=None):
        config = config or {}
        self.state_dim = config.get("state_dim", 10)
        self.action_dim = config.get("action_dim", 4)
        self.max_steps = config.get("max_steps", 1000)
        self.current_step = 0
    
    def _get_reward(self, state, action):
        """子类覆盖实现具体的奖励函数"""
        raise NotImplementedError
    
    def reset(self, *, seed=None, options=None):
        self.current_step = 0
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        self.current_step += 1
        reward = self._get_reward(self.state, action)
        terminated = self.current_step >= self.max_steps
        return self.state, reward, terminated, False, {}
```

在项目中使用：

```python
from shared.custom_envs.common import BaseControlEnv

class MyCustomEnv(BaseControlEnv):
    def _get_reward(self, state, action):
        # 项目特定的奖励逻辑
        return np.sum(state * action)
```

---

## 项目间的最佳实践

### 1. 版本管理

```bash
# 主仓库
projects/
├── project_a_robotics/
│   └── requirements-lock.txt  # 锁定依赖版本
├── project_b_trading/
│   └── requirements-lock.txt
└── rlframework/
    └── pyproject.toml         # 版本 0.2.0
```

### 2. CI/CD 流程

```yaml
# .github/workflows/multi_project_test.yml
name: Multi-Project Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        project: [project_a_robotics, project_b_trading]
    steps:
      - uses: actions/checkout@v3
        with:
          submodules: true
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install project
        run: |
          cd projects/${{ matrix.project }}
          pip install -e ".[dev]"
      
      - name: Run tests
        run: |
          cd projects/${{ matrix.project }}
          make test
```

### 3. 共享文档

```
docs/
├── architecture.md       # 整体架构说明
├── rlframework/          # 框架文档
├── project_a/            # 项目A文档
├── project_b/            # 项目B文档
└── shared_components.md  # 共享组件文档
```

### 4. 数据和模型共享

```
shared_models/
├── pretrained/
│   └── base_policy.pt   # 基础策略模型
└── datasets/
    ├── robotics_data.pkl
    └── trading_data.pkl
```

项目中加载：

```python
import torch
from pathlib import Path

model_path = Path(__file__).parent.parent.parent / "shared_models" / "pretrained" / "base_policy.pt"
state_dict = torch.load(model_path)
policy.load_state_dict(state_dict)
```

---

## 总结

| 方案 | 优点 | 缺点 | 适用场景 |
|------|------|------|---------|
| **PyPI 发布** | 独立、版本明确、易于更新 | 需要维护PyPI | 不同团队、外部项目 |
| **Monorepo** | 共享代码、统一版本控制、开发快 | 仓库较大，依赖复杂 | 同一团队、紧密协作 |
| **Git 子模块** | 平衡两者的优点 | 需要学习曲线 | 推荐方案 |

**推荐流程：**
1. **开发阶段** → 使用 Monorepo + Git 子模块
2. **生产部署** → 发布到 PyPI，各项目独立安装
3. **版本更新** → 在 Monorepo 中更新，发布新版本
