# rlframework 统一 Run Layout 方案

## 背景

当前 `examples/01_ppo_cartpole.py` 里，RLlib `UnifiedLogger` 和
`framework_checkpointing` 的路径分别由用户手动维护：

```python
def logger_creator(config):
    logdir = os.path.abspath("./checkpoints/cartpole")
    os.makedirs(logdir, exist_ok=True)
    return UnifiedLogger(config, logdir, loggers=None)

config = (
    CustomPPOConfig()
    .debugging(logger_creator=logger_creator)
    .framework_checkpointing(freq=5, local_dir="./checkpoints/cartpole")
)
```

这种写法能跑，但存在几个问题：

- `UnifiedLogger` 产物和 checkpoint 容易混在同一个目录。
- 每个训练脚本都要重复写 `logger_creator` 和 checkpoint 路径。
- local storage、metrics、checkpoint、RLlib logs 没有统一的实验目录规范。
- 后续迁移到 MinIO/S3 或多实验管理时，路径命名容易冲突。

目标是在 `rlframework` 内部统一维护这些路径，让用户只声明一次实验目录。

## 目标用法

推荐用户侧 API：

```python
config = (
    CustomPPOConfig()
    .framework_run("cartpole", root_dir="./runs")
    .environment("CartPole-v1")
    .training(
        lr=5e-5,
        train_batch_size=4000,
        num_epochs=10,
        minibatch_size=128,
    )
    .env_runners(num_env_runners=2)
    .evaluation(evaluation_interval=5)
    .framework_checkpointing(freq=5)
    .metrics(reporters=["file"])
    .storage(upload_async=True, best_upload_freq=1)
)
```

用户不再需要手写：

```python
.debugging(logger_creator=logger_creator)
.framework_checkpointing(freq=5, local_dir="./checkpoints/cartpole")
FileReporter(filepath="./logs/cartpole_metrics.jsonl")
```

## 目录布局

`framework_run(name="cartpole", root_dir="./runs")` 生成统一实验目录：

```text
runs/
  cartpole/
    rllib_logs/
      result.json
      progress.csv
      params.json
      params.pkl
      events.out.tfevents.*
    checkpoints/
      iter_000005/
      iter_000010/
      best/
    metrics/
      metrics.jsonl
    storage/
      best.tar
```

核心原则：

```text
run_dir
  -> rllib_log_dir
  -> checkpoint_dir
  -> best_checkpoint_dir
  -> metrics_dir
  -> storage_dir
```

`run_dir` 是唯一源头，其它目录都从它派生。

## RunLayout 数据结构

建议新增一个轻量数据结构：

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path
    rllib_log_dir: Path
    checkpoint_dir: Path
    best_checkpoint_dir: Path
    metrics_dir: Path
    storage_dir: Path
```

路径规则：

```python
run_dir = Path(root_dir) / name

if run_id is not None:
    run_dir = Path(root_dir) / name / run_id
```

示例：

```text
./runs/cartpole/
./runs/cartpole/lr_5e-5_seed_1/
```

## API 设计

在 `FrameworkConfigMixin` 中新增：

```python
def framework_run(
    self,
    name: str,
    root_dir: str = "./runs",
    run_id: str | None = None,
    auto_logger: bool = True,
    strict_layout: bool = False,
) -> "FrameworkConfigMixin":
    ...
```

参数含义：

- `name`: 实验名称，例如 `"cartpole"`。
- `root_dir`: 实验根目录，默认 `"./runs"`。
- `run_id`: 可选的二级目录，用于区分多次实验。
- `auto_logger`: 是否自动注入 RLlib logger。
- `strict_layout`: 是否强制所有 framework 管理的路径都落在 layout 下。

## build 阶段注入顺序

`CustomPPOConfig.build()` 当前已经先调用：

```python
self._apply_framework_runtime_config()
```

建议在 `_apply_framework_runtime_config()` 开头解析并应用 layout：

```python
def _apply_framework_runtime_config(self) -> None:
    layout = self._resolve_framework_run_layout()
    self._apply_framework_run_layout(layout)

    self._validate_framework_config()
    reporters = self.build_reporters()
    ckpt_mgr = self.build_checkpoint_manager()
    ...
```

顺序很重要：必须先填充默认路径，再构建 reporters 和 checkpoint manager。

## UnifiedLogger 路径统一

如果用户没有传 `logger_creator`，framework 自动注入：

```python
def _make_default_logger_creator(log_dir: Path):
    log_dir = log_dir.resolve()

    def logger_creator(config):
        log_dir.mkdir(parents=True, exist_ok=True)
        return UnifiedLogger(config, str(log_dir), loggers=None)

    return logger_creator
```

注入规则：

```python
if layout is not None and self._run_auto_logger:
    if getattr(self, "logger_creator", None) is None:
        self.debugging(
            logger_creator=_make_default_logger_creator(layout.rllib_log_dir)
        )
```

这样 `algo.train()` 后，Ray/RLlib 默认 logger 产物会写到：

```text
runs/cartpole/rllib_logs/
```

包括：

```text
result.json
progress.csv
params.json
params.pkl
events.out.tfevents.*
```

## framework_checkpointing 路径统一

建议把 `framework_checkpointing` 的签名改成：

```python
def framework_checkpointing(
    self,
    freq: int = 0,
    local_dir: str | None = None,
) -> "FrameworkConfigMixin":
    ...
```

路径规则：

- 用户显式传 `local_dir`：使用用户路径。
- 用户未传 `local_dir` 且启用了 `framework_run`：使用 `layout.checkpoint_dir`。
- 用户未传 `local_dir` 且未启用 `framework_run`：保持兼容，使用 `"./checkpoints"`。

伪代码：

```python
def framework_checkpointing(self, freq: int = 0, local_dir: str | None = None):
    validators.validate_non_negative_int(freq, "checkpoint_freq")
    self._checkpoint_freq = freq
    self._checkpoint_local_dir = local_dir
    self._checkpoint_local_dir_configured = local_dir is not None
    return self
```

应用 layout 时：

```python
if layout is not None and not self._checkpoint_local_dir_configured:
    self._checkpoint_local_dir = str(layout.checkpoint_dir)
elif self._checkpoint_local_dir is None:
    self._checkpoint_local_dir = "./checkpoints"
```

最终 callback 注入：

```python
FrameworkCallback.with_reporters(
    reporters,
    checkpoint_freq=self._checkpoint_freq,
    checkpoint_local_dir=self._checkpoint_local_dir,
    best_local_dir=str(layout.best_checkpoint_dir),
    ...
)
```

checkpoint 产物：

```text
runs/cartpole/checkpoints/iter_000005/
runs/cartpole/checkpoints/iter_000010/
runs/cartpole/checkpoints/best/
```

## Metrics 路径统一

如果用户启用了 file reporter，但没有指定 `filepath`：

```python
.metrics(reporters=["file"])
```

自动填充：

```python
if layout is not None and "file" in self._metrics_reporters:
    cfg = self._metrics_reporter_configs.setdefault("file", {})
    cfg.setdefault("filepath", str(layout.metrics_dir / "metrics.jsonl"))
```

输出：

```text
runs/cartpole/metrics/metrics.jsonl
```

如果用户显式传了：

```python
.metrics(
    reporters=["file"],
    reporter_configs={"file": {"filepath": "./custom.jsonl"}},
)
```

则尊重用户配置，不覆盖。

## Local storage 路径统一

当前 `.storage()` 默认 local backend 使用：

```python
LocalBackend(root="./checkpoints")
```

这容易和 checkpoint 源目录混在一起。启用 `framework_run` 后建议默认改为：

```python
if (
    layout is not None
    and self._storage_configured
    and self._storage_backend == "local"
):
    self._storage_backend_config.setdefault("root", str(layout.storage_dir))
```

这样 remote/local storage 的目标目录是：

```text
runs/cartpole/storage/
```

周期 checkpoint 的源目录仍是：

```text
runs/cartpole/checkpoints/
```

二者分开，避免 `best/`、`best.tar`、`iter_xxx/` 混杂。

## 自定义 logger_creator 的处理

如果用户直接传：

```python
.debugging(logger_creator=my_logger_creator)
```

framework 无法保证它写到 `layout.rllib_log_dir`。因为路径完全由用户函数内部决定：

```python
def my_logger_creator(config):
    return UnifiedLogger(config, "/tmp/my_logs", loggers=None)
```

因此语义应明确：

- 未传 `logger_creator`：framework 自动注入默认 `UnifiedLogger`，路径受 layout 管理。
- 传了 raw `logger_creator`：用户接管 RLlib logger 路径，framework 不保证其在 run layout 下。
- `strict_layout=False`：发现 raw `logger_creator` 时 warning。
- `strict_layout=True`：发现 raw `logger_creator` 时 raise。

建议错误信息：

```text
Custom logger_creator bypasses framework_run layout. Use framework_logger_creator()
if you want a custom logger under the managed run directory.
```

## Framework-aware logger 扩展点

为了支持自定义 logger，同时保留 framework 管理的 layout，可以新增：

```python
def framework_logger_creator(self, factory):
    self._framework_logger_creator = factory
    return self
```

用户使用：

```python
def make_logger(config, layout):
    return UnifiedLogger(
        config,
        str(layout.rllib_log_dir),
        loggers=None,
    )

config = (
    CustomPPOConfig()
    .framework_run("cartpole")
    .framework_logger_creator(make_logger)
)
```

内部包装成 RLlib 需要的一参函数：

```python
def logger_creator(config):
    return self._framework_logger_creator(config, layout)
```

这样用户能自定义 logger，但路径信息仍由 framework 提供。

## 路径覆盖规则

推荐最终规则：

```text
framework_run enabled
  logger_creator 未设置
    -> 自动设置 UnifiedLogger 到 layout.rllib_log_dir
  logger_creator 已设置
    -> strict_layout=True: raise
    -> strict_layout=False: warning，用户自管

framework_checkpointing(freq=N, local_dir=None)
  -> 使用 layout.checkpoint_dir

framework_checkpointing(freq=N, local_dir="...")
  -> 使用用户路径

metrics(["file"]) 且 filepath 未设置
  -> 使用 layout.metrics_dir / "metrics.jsonl"

storage(backend="local") 且 root 未设置
  -> 使用 layout.storage_dir
```

## 与 algo.train() 调用链的关系

`UnifiedLogger` 的写入发生在 `algo.train()` 后半段：

```text
algo.train()
  -> Trainable.train()
    -> self.step()
      -> Algorithm.step()
        -> training_step()
        -> compile result
    -> self.log_result(result)
      -> Algorithm.log_result(result)
        -> callbacks.on_train_result(...)
           -> FrameworkCallback.on_train_result(...)
              -> FileReporter.report(...)
              -> periodic checkpoint save
        -> Trainable.log_result(self, result)
          -> self._result_logger.on_result(result)
            -> UnifiedLogger.on_result(result)
              -> JsonLogger.on_result(result)
              -> CSVLogger.on_result(result)
              -> TBXLogger.on_result(result)
```

因此路径统一必须在 `config.build()` / `Algorithm.__init__()` 阶段完成。到
`UnifiedLogger.on_result()` 时，logger 对象已经创建好，不能再修改路径。

## 实施步骤

建议按以下顺序落地：

1. 新增 `RunLayout` 数据结构和 `.framework_run(...)` API。
2. 在 `_apply_framework_runtime_config()` 开头解析并应用 layout。
3. 未设置 raw `logger_creator` 时，自动注入 `UnifiedLogger` 到 `layout.rllib_log_dir`。
4. 修改 `framework_checkpointing(local_dir=None)`，默认使用 `layout.checkpoint_dir`。
5. 为 file metrics reporter 自动填充 `layout.metrics_dir / "metrics.jsonl"`。
6. 为 local storage backend 自动填充 `layout.storage_dir`。
7. 增加 raw `logger_creator` 的 warning / strict error。
8. 更新 `examples/01_ppo_cartpole.py`。
9. 增加单测，覆盖默认路径、显式路径不覆盖、自定义 logger 行为。

## 单测建议

重点测试：

- `framework_run("cartpole")` 后自动生成 `rllib_logs`、`checkpoints`、`metrics`、`storage` 路径。
- 未设置 `logger_creator` 时，config 会注入默认 logger creator。
- 已设置 raw `logger_creator` 且 `strict_layout=False` 时不覆盖。
- 已设置 raw `logger_creator` 且 `strict_layout=True` 时报错。
- `framework_checkpointing(freq=5)` 默认使用 `layout.checkpoint_dir`。
- `framework_checkpointing(freq=5, local_dir="/tmp/ckpts")` 不被覆盖。
- `.metrics(reporters=["file"])` 自动填充 metrics filepath。
- `.metrics(... filepath="custom.jsonl")` 不被覆盖。
- `.storage(backend="local")` 自动填充 local storage root。
- `.storage(backend="local", root="/tmp/storage")` 不被覆盖。

## 结论

不要让 `UnifiedLogger` 和 `framework_checkpointing` 互相感知。正确做法是让二者都依赖
`FrameworkConfigMixin` 解析出的同一个 `RunLayout`。

这样路径规则集中在 framework 内部：

- 用户侧 API 更简单。
- checkpoint、metrics、RLlib logs、local storage 有清晰边界。
- 后续 Ray `UnifiedLogger` API 变化时，只需要修改 framework 内部实现。
- 多实验、多 run、对象存储 prefix 扩展也更自然。
