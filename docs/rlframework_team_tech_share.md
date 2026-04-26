# rlframework 团队技术分享：基于 RLlib 的强化学习工程化框架

> 分享目标：让团队快速理解 rlframework 为什么存在、它基于 RLlib 做了哪些增强、业务项目应该如何接入，以及后续如何一起沉淀更多强化学习工程能力。

## 1. 分享定位

### 适合听众

- 正在使用或准备使用 RLlib 做强化学习训练的同学
- 需要接入自定义环境、模型、指标、checkpoint、对象存储的算法/工程同学
- 需要把实验代码推进到可复用、可观测、可恢复、可上线训练流程的项目负责人

### 建议时长

| 环节 | 时长 | 内容 |
| --- | ---: | --- |
| 背景与痛点 | 5 min | 为什么直接使用 RLlib 还不够 |
| 总体架构 | 10 min | rlframework 分层架构和设计原则 |
| 核心能力 | 20 min | 配置、算法 hook、模型、环境、指标、存储、Replay Buffer |
| 示例演示 | 15 min | 从最小训练脚本到生产化训练配置 |
| 接入建议 | 5 min | 业务项目如何迁移和扩展 |
| Q&A | 5 min | 讨论团队后续共建方向 |

## 2. 一句话介绍

rlframework 是一个 **RLlib-native 的强化学习工程化框架**。

它不重写 RLlib 的训练核心，而是在 RLlib 原生能力之上，补齐团队项目中反复需要的工程能力：

- 统一配置入口
- 自定义算法 hook
- 模型组件扩展
- 完整自定义 RLModule
- 环境封装
- 指标上报
- checkpoint 管理
- 本地 / MinIO / S3 存储
- 自定义 Replay Buffer
- 通用工具和配置校验

可以这样理解：

```text
RLlib 负责“怎么训练”
rlframework 负责“怎么把训练工程化、标准化、可扩展化”
```

## 3. 为什么需要这个框架

### 3.1 直接使用 RLlib 的常见问题

在业务项目里直接写 RLlib 训练脚本，早期很快，但项目变复杂后通常会遇到这些问题：

| 问题 | 典型表现 | 影响 |
| --- | --- | --- |
| 配置分散 | 每个项目自己写环境、训练、checkpoint、日志配置 | 难复用，难排查 |
| 指标不统一 | 有的写文件，有的写 TensorBoard，有的直接 print | 后期难接 Grafana / Prometheus |
| checkpoint 逻辑重复 | 每个训练脚本都写保存、压缩、上传、重试 | 容易漏，容易阻塞训练 |
| 模型扩展成本高 | 改 encoder/head/RLModule 时需要理解大量 RLlib 内部细节 | 上手门槛高 |
| 环境封装重复 | reset/step 签名、远端环境接入、wrapper 每个项目重复写 | 模板代码多 |
| 算法定制不规范 | curriculum、动态参数、自定义指标直接塞进训练循环 | 后续维护困难 |

### 3.2 rlframework 的目标

rlframework 的目标不是替代 RLlib，而是把团队项目里反复出现的工程模板沉淀成框架能力：

1. **降低接入成本**：训练脚本保持 RLlib 风格，但常用工程能力一行配置接入。
2. **统一工程规范**：指标、checkpoint、存储、模型扩展都有标准位置。
3. **保持 RLlib 兼容**：训练主流程仍由 RLlib 接管，方便复用 RLlib 生态和升级路径。
4. **方便团队共建**：新项目新增的环境、模型、reporter、backend 可以沉淀回框架。

## 4. 总体架构

建议分享时先展示总览图：

- 简化分层图：`docs/rlframework_layered_overview.puml`
- 模块依赖图：`docs/rlframework_simple_architecture.puml`
- 完整细节图：`docs/rlframework_architecture.puml`

### 4.1 分层视角

```text
应用层
  用户训练脚本 / 业务项目

配置与装配层
  CustomPPOConfig / CustomSACConfig
  FrameworkConfigMixin

算法扩展层
  CustomPPO / CustomSAC
  FrameworkAlgorithmMixin

模型与环境扩展层
  ComponentRegistry / CompositeCatalog / Custom*RLModule
  BaseEnv / RemoteEnv / Wrappers

工程横切层
  FrameworkCallback
  Reporters
  CheckpointManager

上游运行时层
  RLlib Runtime
  Gymnasium Env API

外部系统层
  metrics.jsonl / InfluxDB / Prometheus
  Local FS / MinIO / S3
```

### 4.2 核心设计原则

#### 原则一：不重写 RLlib 训练核心

采样、Learner 更新、权重同步、Replay Buffer 主流程、EnvRunner、训练循环仍然由 RLlib 负责。

rlframework 继承和组合 RLlib 原生接口：

- `PPOConfig` / `SACConfig`
- `PPO` / `SAC`
- `RLModule`
- `Catalog`
- `RLlibCallback`
- `ReplayBuffer`
- `gymnasium.Env`

#### 原则二：在工程横切点增强

rlframework 主要在这些位置增强：

- 配置 build 前：注入 callback、reporter、checkpoint manager、RLModuleSpec
- training_step 前后：提供算法 hook
- episode / train result / evaluation result：提取和上报指标
- checkpoint 保存后：统一上传到本地、MinIO、S3
- 模型构建阶段：通过 Catalog 或 RLModule 扩展网络结构
- 环境接入阶段：统一 Gymnasium 环境封装

#### 原则三：扩展点清晰

| 想做什么 | 推荐扩展点 |
| --- | --- |
| 加训练前后逻辑 | 继承 `CustomPPO` / `CustomSAC`，覆盖 hook |
| 加自定义指标 | 继承 `FrameworkCallback` 或使用 reporter |
| 换模型局部组件 | `ComponentRegistry` + `framework_models()` |
| 完整自定义模型 | 继承 `Custom*RLModule`，通过 `RLModuleSpec` 接入 |
| 接入自定义环境 | 继承 `BaseEnv` / `RemoteEnv` |
| 接入对象存储 | 使用 `CheckpointManager` 和 storage backend |
| 接入新指标系统 | 实现 `BaseReporter` |
| 使用自定义 Replay Buffer | 在 `replay_buffer_config` 里指定 buffer type |

## 5. rlframework 基于 RLlib 做了哪些事情

### 5.1 配置装配：把复杂工程能力收敛到统一入口

配置层是业务项目最常接触的入口。

- `CustomPPOConfig` 继承 RLlib `PPOConfig`
- `CustomSACConfig` 继承 RLlib `SACConfig`
- 二者混入 `FrameworkConfigMixin`

框架新增了这些链式配置能力：

```python
config = (
    CustomPPOConfig()
    .environment("CartPole-v1")
    .training(lr=3e-4)
    .storage(backend="minio", ...)
    .metrics(reporters=["file", "prometheus"])
    .framework_checkpointing(freq=10, local_dir="./checkpoints")
)
```

背后自动完成：

- 构建 reporters
- 构建 CheckpointManager
- 注入 FrameworkCallback
- 注入自定义模型 catalog 或 RLModuleSpec
- 做基础配置校验

推广重点：**业务同学仍然写 RLlib 风格配置，但不再重复写工程模板代码。**

### 5.2 算法扩展：只加 hook，不改训练主流程

算法层主要包括：

- `FrameworkAlgorithmMixin`
- `CustomPPO`
- `CustomSAC`

`CustomPPO` / `CustomSAC` 继承 RLlib 的 `PPO` / `SAC`，并混入 `FrameworkAlgorithmMixin`。

提供两个核心 hook：

```python
class MyPPO(CustomPPO):
    def on_before_training_step(self):
        # 每次 training_step 前执行
        pass

    def on_after_training_step(self, result):
        # 每次 training_step 后执行，可写入自定义指标
        result["custom/my_metric"] = 1.0
        return result
```

适用场景：

- curriculum learning
- 动态调整训练参数
- 记录训练耗时
- 注入自定义统计指标
- 训练过程中的业务状态同步

推广重点：**算法扩展有固定入口，不需要复制或修改 RLlib 的训练循环。**

### 5.3 模型扩展：支持“局部替换”和“完整自定义”两种粒度

模型扩展有两条路径。

#### 路径一：ComponentRegistry + CompositeCatalog

适合大多数场景：只替换 encoder、actor head、critic head、Q head、action distribution 等局部组件。

```python
from rlframework.models.catalog import ComponentRegistry

@ComponentRegistry.register_encoder("my_encoder")
def build_my_encoder(obs_space, action_space, model_config, framework):
    return MyEncoder(...)

config = (
    CustomPPOConfig()
    .framework_models(
        encoder="my_encoder",
        actor_head="default",
        critic_head="default",
    )
)
```

优点：

- 不需要完整重写 RLModule
- 可以复用 RLlib 默认结构
- 适合快速试验不同网络组件

#### 路径二：Custom*RLModule + RLModuleSpec

适合需要完全控制 forward、训练输出、state 保存恢复的场景。

```python
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from rlframework.models.rl_module import CustomPPORLModule

class MyPPOModule(CustomPPORLModule):
    def setup(self):
        ...

    def _forward(self, batch, **kwargs):
        ...

    def _forward_train(self, batch, **kwargs):
        ...

config = CustomPPOConfig().rl_module(
    rl_module_spec=RLModuleSpec(
        rl_module_class=MyPPOModule,
        model_config_dict={...},
    )
)
```

注意：完整自定义 RLModule 可以只指定 `rl_module_class`，不一定需要 `catalog_class`。

推广重点：**模型层既能支持轻量组件化试验，也能支持复杂模型完全接管。**

### 5.4 环境封装：统一本地环境和远端环境接入

环境层提供：

- `BaseEnv`
- `RemoteEnv`
- `NormalizeObsWrapper`
- `RecordEpisodeStatsWrapper`

`BaseEnv` 适合本地 Gymnasium 环境。用户只需要实现：

```python
def _reset(self, seed=None, options=None):
    ...

def _step(self, action):
    ...
```

`RemoteEnv` 适合远端仿真服务或环境服务，例如 HTTP、gRPC、Unix socket。用户只需要实现：

```python
def _send_reset(self, seed=None, options=None):
    ...

def _send_step(self, action):
    ...

def _send_close(self):
    ...
```

推广重点：**环境接入只关注业务状态转换或远端通信，不重复处理 Gymnasium 模板逻辑。**

### 5.5 指标观测：统一训练和评估指标出口

指标相关模块：

- `FrameworkCallback`
- `BaseReporter`
- `FileReporter`
- `InfluxDBReporter`
- `PrometheusReporter`

训练时，RLlib 产生的 result 会进入 `FrameworkCallback`，框架会：

1. 提取训练指标
2. 提取评估指标
3. flatten 成统一结构
4. 自动区分 train / eval 阶段
5. fan-out 到多个 reporter

示例：

```python
from rlframework.callbacks import FrameworkCallback
from rlframework.observability.reporters import FileReporter, PrometheusReporter

reporters = [
    FileReporter(filepath="./logs/metrics.jsonl"),
    PrometheusReporter(gateway="http://localhost:9091", job="rl_training"),
]

config = CustomPPOConfig().callbacks(
    FrameworkCallback.with_reporters(reporters)
)
```

推广重点：**训练指标不再散落在 print、临时文件和脚本逻辑里，而是统一出口，方便接 Grafana。**

### 5.6 Checkpoint 和模型存储：统一保存、上传、重试、异步

存储层包括：

- `CheckpointManager`
- `LocalBackend`
- `MinIOBackend`
- `S3Backend`

支持能力：

- 周期性 checkpoint
- best model 保存
- 本地 / MinIO / S3 上传
- 目录 checkpoint 自动打包
- 上传失败重试
- 异步上传，避免阻塞训练

示例：

```python
config = (
    CustomPPOConfig()
    .storage(
        backend="minio",
        endpoint="minio:9000",
        access_key="admin",
        secret_key="password",
        bucket="rl-checkpoints",
        upload_async=True,
    )
    .framework_checkpointing(freq=10, local_dir="./checkpoints")
)
```

推广重点：**训练脚本不再手写保存、压缩、上传、重试逻辑。**

### 5.7 Replay Buffer：兼容 RLlib 新 API 的自定义 Buffer

当前框架提供：

- `PrioritizedSumTreeBuffer`

它继承自 RLlib 的 `PrioritizedEpisodeReplayBuffer`，属于新 EnvRunner API 需要的 `EpisodeReplayBuffer` 系列。

示例：

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

框架还在 `CustomSACConfig.validate()` 中处理了 RLlib 对 replay buffer type 的硬编码白名单误判：如果自定义 buffer 是 `EpisodeReplayBuffer` 子类，就允许通过。

推广重点：**自定义 Replay Buffer 可以保持 RLlib 新 API 兼容，而不是绕开 RLlib 机制。**

## 6. 一轮训练的运行流程

```text
1. 用户创建 CustomPPOConfig / CustomSACConfig
2. 配置 environment / training / model / metrics / storage / checkpointing
3. 调用 config.build()
4. FrameworkConfigMixin 注入 FrameworkCallback、Reporters、CheckpointManager、模型配置
5. RLlib build 算法实例并接管训练主流程
6. CustomPPO / CustomSAC 在 training_step 前后执行 hook
7. RLlib 触发 Callback，FrameworkCallback 提取指标并上报
8. 到达 checkpoint 条件时，CheckpointManager 保存并上传模型
```

## 7. 最小接入示例

```python
import ray
from rlframework.algorithms.ppo import CustomPPOConfig

ray.init(ignore_reinit_error=True)

config = (
    CustomPPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .env_runners(num_env_runners=1)
    .training(lr=3e-4, gamma=0.99)
    .metrics(
        reporters=["file"],
        reporter_configs={"file": {"filepath": "./logs/metrics.jsonl"}},
    )
    .framework_checkpointing(freq=10, local_dir="./checkpoints")
)

algo = config.build()

for i in range(100):
    result = algo.train()
    reward = result.get("env_runners", {}).get("episode_return_mean")
    print(f"iter={i}, reward={reward}")

algo.stop()
ray.shutdown()
```

这个示例已经包含：

- RLlib PPO 训练
- 统一配置入口
- 文件指标输出
- 周期性 checkpoint
- 可继续扩展 storage、reporter、model、env、algorithm hook

## 8. 推荐演示路线

### Demo 1：最小训练脚本

文件：`examples/01_ppo_cartpole.py`

讲解点：

- 业务代码如何创建 config
- 如何接 FileReporter
- 如何保存本地 checkpoint

### Demo 2：自定义算法 hook

文件：`examples/03_custom_algorithm.py`

讲解点：

- `CustomPPO` 如何继承
- `on_before_training_step` / `on_after_training_step` 如何注入逻辑
- 为什么不需要改 RLlib 源码

### Demo 3：自定义模型组件

文件：`examples/05_custom_model_composition.py`

讲解点：

- `ComponentRegistry` 如何注册 encoder/head
- `framework_models()` 如何组合组件
- 什么时候用 CompositeCatalog，什么时候用 Custom RLModule

### Demo 4：生产化训练配置

文件：`examples/04_full_production.py`

讲解点：

- MinIO checkpoint 上传
- InfluxDB / Prometheus 指标上报
- 多 reporter fan-out
- 训练脚本如何保持简洁

## 9. 业务项目接入建议

### 9.1 新项目推荐接入方式

1. 从 `CustomPPOConfig` 或 `CustomSACConfig` 开始。
2. 先接入本地 `FileReporter`，保证指标可追溯。
3. 接入 `framework_checkpointing()`，保证训练可恢复。
4. 有对象存储需求时再接 MinIO / S3。
5. 模型先走 `ComponentRegistry`，只有确实需要完整 forward 控制时再写 `Custom*RLModule`。
6. 自定义训练逻辑优先写到算法 hook，不要散落在训练脚本里。

### 9.2 老项目迁移建议

| 原有写法 | 迁移到 rlframework |
| --- | --- |
| 原生 `PPOConfig` / `SACConfig` | `CustomPPOConfig` / `CustomSACConfig` |
| 手写 metrics 文件 | `FileReporter` |
| 手写 Prometheus / InfluxDB 上报 | `PrometheusReporter` / `InfluxDBReporter` |
| 手写 checkpoint 上传 | `CheckpointManager` + storage backend |
| 训练循环里塞自定义逻辑 | `FrameworkAlgorithmMixin` hook |
| 复制 RLlib 模型代码改 head | `ComponentRegistry` + `CompositeCatalog` |
| 完全自定义模型 forward | `Custom*RLModule` + `RLModuleSpec` |

## 10. 推广价值总结

### 对算法同学

- 更少模板代码，更多精力放在 reward、模型、训练策略上
- 模型扩展路径清晰：局部组件替换或完整 RLModule
- 自定义训练逻辑有稳定 hook，不需要魔改 RLlib

### 对工程同学

- 指标、checkpoint、存储、日志都有统一抽象
- 更容易接入 Grafana、Prometheus、InfluxDB、MinIO、S3
- 训练任务更可观测、可恢复、可复现

### 对团队

- 项目经验可以沉淀成框架能力
- 新项目启动成本降低
- 训练工程规范统一
- 后续升级 RLlib 时路径更清晰

## 11. 当前能力清单

| 能力 | 状态 | 示例 |
| --- | --- | --- |
| PPO / SAC 配置封装 | 已支持 | `examples/01_ppo_cartpole.py`, `examples/02_sac_pendulum.py` |
| 算法 hook | 已支持 | `examples/03_custom_algorithm.py` |
| 生产化训练配置 | 已支持 | `examples/04_full_production.py` |
| 模型组件组合 | 已支持 | `examples/05_custom_model_composition.py` |
| 完整自定义 RLModule | 已支持 | `examples/06_custom_rl_module.py` |
| 自定义评估指标 | 已支持 | `examples/06_custom_eval_metrics.py` |
| 动态学习率调度 | 已支持 | `examples/07_lr_schedule.py` |
| 自定义 Replay Buffer | 已支持 | `examples/09_custom_replay_buffer.py` |
| MinIO / S3 模型保存 | 已支持 | `examples/11_minio_model_saving.py` |
| Grafana 指标链路 | 已支持 | `examples/12_grafana_metrics.py` |
| 自定义环境 | 已支持 | `examples/13_custom_env.py` |

## 12. 后续共建方向

可以和团队讨论下面这些方向：

- 增加更多算法封装，例如 DQN、IMPALA、APPO 等
- 增加更多模型组件模板，例如 CNN encoder、Transformer encoder、multi-head critic
- 增加更多 reporter，例如 OpenTelemetry、Kafka、企业内部监控系统
- 增加训练任务模板，例如离线训练、评估任务、批量实验管理
- 增加模型发布流程，例如 checkpoint 转推理服务、模型 registry
- 增加文档和最佳实践，例如不同环境接入范式、生产训练 checklist

## 13. 分享结尾建议

可以用这段话收尾：

> rlframework 不是要替代 RLlib，而是把我们在强化学习项目里反复写的工程能力沉淀下来。它让业务同学仍然用 RLlib 的方式训练模型，同时自动获得统一配置、指标观测、checkpoint 管理、模型扩展、环境封装和存储后端。我们的目标是让每个项目少写 boilerplate，多沉淀可复用能力，让强化学习训练从“能跑”走向“可维护、可观测、可恢复、可推广”。
