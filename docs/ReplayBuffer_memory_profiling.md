# Replay Buffer 内存对象分析

本文用于分析 `FastSampleEpisodeReplayBuffer` 和
`NumpyIndexedFastSampleEpisodeReplayBuffer` 训练时的内存占用，定位内存由哪类对象、哪行代码分配。

## 启用 tracemalloc

在训练脚本完成 import 后、调用 `config.build()` 前加入：

```python
import gc
import os
import tracemalloc

import numpy as np
import psutil

tracemalloc.start(1)
```

`tracemalloc.start(1)` 中的 `1` 表示保存一层分配调用栈，可以降低分析器自身的内存开销。

## 内存报告函数

将下面的函数放入训练脚本：

```python
def report_memory(algo, label):
    gc.collect()

    rb = algo.local_replay_buffer
    snapshot = tracemalloc.take_snapshot()
    numpy_domain = np.lib.tracemalloc_domain

    numpy_snapshot = snapshot.filter_traces(
        (tracemalloc.DomainFilter(True, numpy_domain),)
    )
    python_snapshot = snapshot.filter_traces(
        (tracemalloc.DomainFilter(False, numpy_domain),)
    )

    process_memory = psutil.Process(os.getpid()).memory_info()
    numpy_bytes = sum(trace.size for trace in numpy_snapshot.traces)
    python_bytes = sum(trace.size for trace in python_snapshot.traces)

    index_bytes = 0
    if hasattr(rb, "_index_episode"):
        index_bytes = rb._index_episode.nbytes + rb._index_timestep.nbytes

    print(f"\n[MEMORY] {label}")
    print(
        {
            "type": type(rb).__name__,
            "stored_timesteps": rb.get_num_timesteps(),
            "stored_episodes": rb.get_num_episodes(),
            "rss_mib": process_memory.rss / 2**20,
            "vms_mib": process_memory.vms / 2**20,
            "traced_python_mib": python_bytes / 2**20,
            "traced_numpy_mib": numpy_bytes / 2**20,
            "tracemalloc_overhead_mib": (
                tracemalloc.get_tracemalloc_memory() / 2**20
            ),
            "numpy_index_mib": index_bytes / 2**20,
            "numpy_index_capacity": getattr(rb, "_index_capacity", None),
            "python_index_count": len(rb._indices),
            "learner_inflight": (
                algo._learner_inflight()
                if hasattr(algo, "_learner_inflight")
                else None
            ),
        }
    )

    print("Top live allocations:")
    for stat in snapshot.statistics("lineno")[:25]:
        print(stat)

    snapshot.dump(f"/tmp/{label}.tracemalloc")
```

该函数会输出：

- Replay Buffer 类型、当前 timestep 数和 episode 数。
- 当前进程的 RSS 和 VMS。
- `tracemalloc` 追踪到的 Python heap 和 NumPy data memory。
- NumPy Replay 索引的预留容量和字节数。
- Python `_indices` 的当前元素数量。
- 异步 Learner 当前 in-flight 请求数。
- 存活内存最大的 25 个分配代码行。
- 可供后续离线分析的 `tracemalloc` snapshot 文件。

## 调用方式

应在两个实验达到相同 sampled env steps 时调用，而不只是比较相同训练时间或相同 iteration。

```python
result = algo.train()

if iteration == 20:
    report_memory(algo, "numpy_step_500000")
```

普通 Fast Buffer 实验可以使用另一个 label：

```python
report_memory(algo, "fast_step_500000")
```

生成的 snapshot 文件为：

```text
/tmp/numpy_step_500000.tracemalloc
/tmp/fast_step_500000.tracemalloc
```

## 结果解读

重点关注 Top live allocations 中的文件和代码行：

- `single_agent_episode.py`、环境代码、`copy.py`：通常对应 episode、observation、action、reward、info 或 episode deepcopy。
- `replay_buffers.py:115`：普通 Fast Buffer 的 Python list-of-tuples 索引。
- `replay_buffers.py:396` 和 `replay_buffers.py:397`：NumPy Buffer 的 episode 和 timestep 索引数组。
- `traced_python_mib` 很大：主要内存在 Python 对象，例如 list、tuple、dict 和 `SingleAgentEpisode`。
- `traced_numpy_mib` 很大：主要内存在 NumPy ndarray data memory。
- `rss_mib` 很大，但 `traced_python_mib + traced_numpy_mib` 很小：继续排查 PyTorch/native allocator、Ray object store、共享内存，或者已经释放但尚未归还操作系统的 allocator 高水位。
- `learner_inflight` 较大：可能同时保留了多个异步 Learner batch。
- `vms_mib` 很大但 `rss_mib` 不大：通常只是 NumPy 预留了虚拟地址空间，不代表对应物理内存已经驻留。

## 公平对比要求

比较两种 Buffer 时至少应保证：

- `capacity` 相同。
- `stored_timesteps` 相同。
- `stored_episodes` 和 episode 长度分布接近。
- sampled env steps 相同。
- observation、action、reward 和 info 的结构与 dtype 相同。
- EnvRunner、Learner 数量以及 `max_requests_in_flight_per_learner` 相同。
- 比较的是同一种进程的 RSS，而不是一边统计 driver、另一边统计整个 Ray 集群。

## 开销注意事项

`tracemalloc` 会为被追踪的内存分配保存元数据。Replay Buffer 中存在大量 Python tuple、dict 或标量对象时，分析器本身也可能占用较多内存。

建议：

1. 先使用 `capacity=100_000` 做诊断运行。
2. 使用 `tracemalloc.start(1)`，不要在第一次诊断时保存很深的调用栈。
3. 使用 profiler 输出定位对象，不要用开启 profiler 后的 RSS 重新判断两种 Buffer 的最终内存比例。
4. 在相同 sampled env steps 处分别保存 Fast 和 NumPy snapshot。
5. 如果问题只在超大容量下出现，考虑使用能分析 native allocation peak 的工具，而不是长期启用 `tracemalloc`。

