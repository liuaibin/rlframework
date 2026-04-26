# rlframework

**一个企业级的强化学习框架，构建于 RLlib 之上**

rlframework 提供了一套高级工具和最佳实践，用于构建、训练和部署深度强化学习应用。它简化了 RLlib 的使用，提供了更好的可扩展性、日志记录、检查点管理和模型存储等功能。

## 特性

- ✅ **自定义评估指标** — 通过回调在训练/评估阶段注入任意指标（成功率、效率等），自动区分 train/eval 阶段上报
- ✅ **动态学习率调度** — 支持 `[[timestep, lr], ...]` 格式的 LR Schedule，按 step 线性插值衰减
- ✅ **自定义模型组件** — `@ComponentRegistry` 装饰器注册自定义 Encoder/Actor/Critic/Q-Head，支持组合式拼接
- ✅ **自定义 Replay Buffer** — 提供 `PrioritizedSumTreeBuffer`（优先级采样）和 `ReservoirReplayBuffer`（均匀采样保证）
- ✅ **自定义训练步骤** — 覆写 `on_before_training_step` / `on_after_training_step` 注入任意逻辑
- ✅ **自定义模型保存** — 本地、MinIO、S3 多后端支持，检查点自动上传，异步非阻塞
- ✅ **指标打点与可视化** — FileReporter / InfluxDBReporter / PrometheusReporter，支持 Grafana 看板开箱即用
- ✅ **简化的算法接口** — Mixin 模式扩展 PPO、SAC，无需修改 RLlib 源码

## 快速开始

### 安装

#### 前置要求：安装 uv（推荐）

`uv` 是一个现代化的 Python 包管理工具，安装方式：

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 或者使用 pip 安装
pip install uv
```

#### 方式 1：使用 uv（推荐，如果已安装）
```bash
make help              # 查看所有命令
make uv-venv          # 创建虚拟环境
source .venv/bin/activate
make sync             # 同步依赖
```

#### 方式 2：使用 pip（标准方式）
```bash
make install          # 自动创建虚拟环境并安装
source .venv/bin/activate
```

#### 方式 3：手动安装
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 验证安装
```bash
python -c "import rlframework; print('✓ rlframework 安装成功')"
```

## 基础示例

### 最小化示例：使用 CartPole 训练 PPO

```python
from rlframework.algorithms.ppo import CustomPPOConfig
import ray

# 初始化 Ray
ray.init(ignore_reinit_error=True)

# 配置 PPO 算法
config = (
    CustomPPOConfig()
    .environment("CartPole-v1")      # 使用标准 Gym 环境
    .framework("torch")
    .env_runners(num_env_runners=1)
    .training(
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
    )
    .metrics(
        reporters=["file"],
        reporter_configs={"file": {"filepath": "./results/metrics.jsonl"}},
    )
    .checkpointing(freq=10, local_dir="./results/checkpoints", upload_async=True)
)

# 构建和训练
algo = config.build()

for i in range(100):
    result = algo.train()

    if (i + 1) % 10 == 0:
        reward = result["env_runners"]["episode_return_mean"]
        print(f"迭代 {i+1}: 平均奖励 = {reward:.2f}")

algo.stop()
ray.shutdown()
```

## 项目结构

```
rlframework/
├── algorithms/              # 算法实现和扩展
│   ├── base.py             # FrameworkAlgorithmMixin（所有算法的基类）
│   ├── ppo.py              # CustomPPO 和 CustomPPOConfig
│   ├── sac.py              # CustomSAC 和 CustomSACConfig
│   └── __init__.py
│
├── config/                 # 配置管理
│   ├── framework_config.py # 全局框架配置
│   └── __init__.py
│
├── envs/                   # 环境封装和包装器
│   ├── remote_env.py       # 远程环境支持
│   ├── wrappers/           # 自定义环境包装器
│   └── __init__.py
│
├── callbacks/              # 训练回调
│   ├── framework_callback.py
│   └── __init__.py
│
├── observability/          # 指标上报与观测
│   ├── reporters/          # 各种报告器后端
│   └── __init__.py
│
├── models/                 # 神经网络模型
│   ├── catalog.py          # 模型目录
│   ├── components/         # 神经网络组件
│   └── __init__.py
│
├── storage/                # 模型和检查点存储
│   ├── checkpoint_manager.py  # 检查点管理
│   ├── backends/              # 存储后端（本地、S3、MinIO）
│   └── __init__.py
│
├── utils/                  # 工具函数
│   ├── data_utils.py
│   ├── exceptions.py
│   ├── replay_buffers.py
│   ├── torch_utils.py
│   └── __init__.py
│
├── examples/               # 使用示例
│   ├── 01_ppo_cartpole.py           # 基础示例
│   ├── 02_sac_pendulum.py           # SAC 示例
│   ├── 03_custom_algorithm.py       # 自定义算法
│   ├── 04_full_production.py        # 生产级示例
│   ├── 05_custom_model_composition.py
│   ├── 06_custom_eval_metrics.py
│   ├── 07_lr_schedule.py
│   ├── 09_custom_replay_buffer.py
│   ├── 11_minio_model_saving.py
│   ├── 12_grafana_metrics.py
│   ├── 13_custom_env.py
│   └── README.md                    # 示例文档
│
├── tests/                  # 单元测试
│   ├── test_algorithms.py
│   ├── test_logging.py
│   ├── test_storage.py
│   ├── test_utils.py
│   └── conftest.py
│
├── pyproject.toml          # 项目元数据和依赖
├── Makefile                # 开发任务
└── README.md               # 本文件
```

## 高级功能指南

本节详细说明框架的 7 大可扩展能力，每个功能均配有 `examples/` 中的可运行示例。

---

### 1. 自定义评估指标

通过继承 `FrameworkCallback` 并覆写 `on_episode_end`，在任意 episode 结束时注入自定义指标。训练和评估阶段的指标自动添加 `train/` / `eval/` 前缀并上报。

**完整示例：** [`examples/06_custom_eval_metrics.py`](examples/06_custom_eval_metrics.py)

```python
from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.callbacks import FrameworkCallback
from rlframework.observability.reporters import FileReporter

class SuccessRateCallback(FrameworkCallback):
    def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):
        ep_return = episode.get_return()
        if metrics_logger is not None:
            metrics_logger.log_value("success", float(ep_return > 195.0), reduce="mean")

reporters = [FileReporter(filepath="./logs/eval_metrics.jsonl")]

config = (
    CustomPPOConfig()
    .environment("CartPole-v1")
    .evaluation(evaluation_interval=5, evaluation_duration=10)
    .callbacks(SuccessRateCallback.with_reporters(reporters))
)
```

---

### 2. 动态学习率调度

通过 `[[timestep, lr], ...]` 格式的学习率表，RLlib 在两点之间线性插值，自动随 step 调整学习率。

**完整示例：** [`examples/07_lr_schedule.py`](examples/07_lr_schedule.py)

```python
lr_schedule = [
    [0,     1e-3],    # 起始 1e-3
    [20000, 5e-4],    # 20k 步降至 5e-4
    [50000, 1e-4],    # 50k 步降至 1e-4
    [80000, 1e-5],    # 80k 步降至 1e-5
]

config = CustomPPOConfig().training(lr=lr_schedule, ...)
```

---

### 3. 自定义模型

使用 `@ComponentRegistry` 装饰器注册自定义模型组件（Encoder、Actor Head、Critic Head、Q Head 等），通过 `config.framework_models()` 组合使用。

**完整示例：** [`examples/05_custom_model_composition.py`](examples/05_custom_model_composition.py)（自定义 Encoder + 自定义 Critic）

```python
from rlframework.models.catalog import ComponentRegistry

@ComponentRegistry.register_encoder("my_encoder")
def build_my_encoder(obs_space, action_space, model_config, framework):
    return MyEncoder(...)

@ComponentRegistry.register_critic_head("dueling_critic")
def build_dueling_critic(input_dim, model_config, framework):
    return DuelingCriticHead(...)

config = (
    CustomPPOConfig()
    .framework_models(
        encoder="my_encoder",
        actor_head="default",
        critic_head="dueling_critic",
    )
)
```

> **组合式模型示例：** [`examples/05_custom_model_composition.py`](examples/05_custom_model_composition.py)

---

### 4. 自定义 Replay Buffer

框架内置两种 Replay Buffer，可直接通过 `replay_buffer_config` 接入 RLlib：

| Buffer | 特点 | 适用算法 |
|--------|------|----------|
| `PrioritizedSumTreeBuffer` | SumTree 优先级采样，O(log n) 复杂度 | SAC（以及其他支持回放的 RLlib 算法） |
| `ReservoirReplayBuffer` | 水库采样，每个样本均匀保留概率 | 旧 API 栈下的实验场景 |

**完整示例：** [`examples/09_custom_replay_buffer.py`](examples/09_custom_replay_buffer.py)

```python
from rlframework.algorithms.sac import CustomSACConfig
from rlframework.utils.replay_buffers import PrioritizedSumTreeBuffer

config = (
    CustomSACConfig()
    .environment("Pendulum-v1")
    .training(
        replay_buffer_config={
            "type": PrioritizedSumTreeBuffer,
            "capacity": 50_000,
            "alpha": 0.6,
            "beta": 0.4,
        }
    )
)
```

---

### 5. 自定义训练步骤

通过覆写 `on_before_training_step` / `on_after_training_step` 钩子，在每次训练迭代前后注入自定义逻辑，无需修改 RLlib 内部代码。

**完整示例：** [`examples/03_custom_algorithm.py`](examples/03_custom_algorithm.py)

```python
import time

from rlframework.algorithms.ppo import CustomPPO

class TimedPPO(CustomPPO):
    def setup(self, config):
        super().setup(config)
        self._best_reward = float("-inf")

    def on_before_training_step(self) -> None:
        self._step_start = time.perf_counter()

    def on_after_training_step(self, result):
        elapsed = time.perf_counter() - self._step_start
        result["custom/step_wall_time_s"] = round(elapsed, 4)
        return result
```

---

### 6. 自定义模型保存（MinIO / S3）

检查点管理器支持本地、MinIO、S3 三种后端，可自动异步上传到对象存储，不阻塞训练。

**完整示例（MinIO）：** [`examples/11_minio_model_saving.py`](examples/11_minio_model_saving.py)

```python
from rlframework.storage.checkpoint_manager import CheckpointManager

ckpt_mgr = CheckpointManager(
    backend="minio",
    backend_config={
        "endpoint": "minio:9000",
        "access_key": "admin",
        "secret_key": "password",
        "bucket": "rl-checkpoints",
    },
    upload_async=True,
)

result = algo.train()
local_ckpt = algo.save_to_path("./checkpoints/iter_1")
ckpt_mgr.upload(local_ckpt, remote_name="ppo_cartpole/iter_1")
```

```python
# S3 后端
ckpt_mgr = CheckpointManager(
    backend="s3",
    backend_config={"bucket": "my-bucket", "region_name": "us-east-1"},
    ...
)
```

---

### 7. 指标打点（Grafana 集成）

框架提供三个报告器后端，配合 Grafana 使用：

| 报告器 | 后端 | Grafana 集成方式 |
|--------|------|-----------------|
| `FileReporter` | 本地 JSONL 文件 | 导入 JSON 查询 |
| `InfluxDBReporter` | InfluxDB v2 | InfluxDB 数据源 |
| `PrometheusReporter` | Prometheus Push Gateway | Prometheus 数据源 |

**完整示例：** [`examples/12_grafana_metrics.py`](examples/12_grafana_metrics.py)

```python
from rlframework.observability.reporters import (
    FileReporter,
    InfluxDBReporter,
    PrometheusReporter,
)

reporters = [
    FileReporter(filepath="./logs/metrics.jsonl"),
    InfluxDBReporter(url="http://localhost:8086", org="rl",
                     bucket="metrics", token="my-token"),
    PrometheusReporter(gateway="http://localhost:9091", job="rl_training"),
]

config = CustomPPOConfig().callbacks(
    FrameworkCallback.with_reporters(reporters)
)
```

训练和评估指标自动添加 `train/` / `eval/` 前缀，InfluxDB 和 Prometheus 均支持按 `phase` 标签筛选。

---

## 主要模块使用说明

### 算法 (algorithms)

#### 创建自定义 PPO
```python
from rlframework.algorithms.ppo import CustomPPO

class MyPPO(CustomPPO):
    """自定义 PPO 算法"""

    def on_after_training_step(self, result):
        """在训练步骤后处理结果"""
        result["my_custom_metric"] = 42.0
        return result
```

#### 自定义算法
```python
from rlframework.algorithms.ppo import CustomPPO, CustomPPOConfig

class MyPPO(CustomPPO):
    ...

# 使用自定义算法
config = CustomPPOConfig(algo_class=MyPPO).environment("CartPole-v1")
```

### 2. 观测与指标上报 (observability)

#### 文件报告器
```python
from rlframework.observability.reporters import FileReporter

reporter = FileReporter("metrics.jsonl")
reporter.report({"episode_return_mean": 123.4}, iteration=1)
```

#### InfluxDB 报告器（需要安装 influxdb）
```python
from rlframework.observability.reporters import InfluxDBReporter

reporter = InfluxDBReporter(
    url="http://localhost:8086",
    org="rl",
    bucket="metrics",
    token="your-token",
    measurement="training",
)
reporter.report({"episode_return_mean": 123.4}, iteration=1)
```

#### Prometheus 报告器（需要安装 prometheus-client）
```python
from rlframework.observability.reporters import PrometheusReporter

reporter = PrometheusReporter(gateway="http://localhost:9091", job="rl_training")
reporter.report({"episode_return_mean": 123.4}, iteration=1)
```

### 3. 检查点管理 (storage)

#### 基本用法
```python
from rlframework.storage.checkpoint_manager import CheckpointManager

ckpt_mgr = CheckpointManager(
    backend="local",
    backend_config={"root": "./checkpoints"},
    upload_async=True,
)

# 在训练循环中
result = algo.train()
local_ckpt = algo.save_to_path("./checkpoints/iter_1")
ckpt_mgr.upload(local_ckpt, remote_name="exp1/iter_1")

# 恢复到本地目录
ckpt_mgr.download("exp1/iter_1", "./restored/iter_1")
```

#### 使用不同的存储后端

**本地存储**
```python
from rlframework.storage.backends import LocalBackend

backend = LocalBackend("./checkpoints")
```

**MinIO 存储**
```python
from rlframework.storage.backends import MinIOBackend

backend = MinIOBackend(
    bucket="rl-checkpoints",
    endpoint="minio.example.com:9000",
    access_key="minioadmin",
    secret_key="minioadmin"
)
```

**AWS S3 存储**
```python
from rlframework.storage.backends import S3Backend

backend = S3Backend(
    bucket="rl-checkpoints",
    region_name="us-west-2"
)
```

## 使用自定义环境

### 定义自定义环境

```python
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Discrete

class MyEnv(gym.Env):
    """自定义强化学习环境"""
    
    def __init__(self, config=None):
        config = config or {}
        self.observation_space = Box(-1, 1, shape=(4,), dtype=np.float32)
        self.action_space = Discrete(2)
    
    def reset(self, *, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}
    
    def step(self, action):
        obs = np.zeros(4, dtype=np.float32)
        reward = 1.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info
```

### 在训练中使用

```python
config = (
    CustomPPOConfig()
    .environment(
        MyEnv,
        env_config={
            "param1": 1,
            "param2": "demo"
        }
    )
)
```

## 开发

### 运行测试
```bash
make test              # 运行所有测试
make test-fast        # 运行快速测试（跳过慢速测试）
make test-cov         # 运行测试并生成覆盖率报告
```

### 代码检查和格式化
```bash
make lint              # 检查代码风格
make format           # 自动格式化代码
make typecheck        # 类型检查
make check            # 运行所有检查（lint + format-check + typecheck）
```

### 查看所有可用命令
```bash
make help
```

## 配置说明

### pyproject.toml 中的可选依赖

```bash
# 仅基础依赖
pip install -e "."

# 基础 + 开发工具
pip install -e ".[dev]"

# 基础 + MinIO 存储支持
pip install -e ".[minio]"

# 基础 + AWS S3 支持
pip install -e ".[s3]"

# 基础 + InfluxDB 支持
pip install -e ".[influxdb]"

# 基础 + Prometheus 支持
pip install -e ".[prometheus]"

# 所有依赖
pip install -e ".[all]"
```

## 要求

- Python >= 3.10, < 3.13
- Ray >= 2.10.0
- PyTorch >= 2.2.0
- Gymnasium >= 0.29.0
- NumPy >= 1.24.0

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！

### 开发流程

1. 创建特性分支：`git checkout -b feature/my-feature`
2. 提交更改：`git commit -am 'Add my feature'`
3. 推送分支：`git push origin feature/my-feature`
4. 创建 Pull Request

### 代码风格

- 使用 `ruff` 进行代码检查和格式化
- 所有公开 API 需要类型注解
- 编写单元测试覆盖新功能

## 参考资源

- [RLlib 官方文档](https://docs.ray.io/en/latest/rllib/index.html)
- [Gymnasium 文档](https://gymnasium.farama.org/)
- [项目示例](./examples/)

## 常见问题

### Q: 如何使用 GPU 训练？
```python
config.resources(num_gpus=1)  # 使用 1 个 GPU
```

### Q: 如何并行化训练？
```python
config.env_runners(num_env_runners=8)  # 使用 8 个并行 worker
```

### Q: 如何保存和恢复训练状态？
```python
# 保存
checkpoint_path = algo.save_to_path("./checkpoints/iter_100")

# 恢复
algo.restore_from_path(checkpoint_path)
```

## 支持

如有问题，请提交 Issue 或联系开发团队。
