"""监督学习数据加载与预处理工具.

支持多种数据格式和环境：
- RLlib 离线数据格式 (JSON/JSONL/NumPy)
- Parquet（需 pandas + pyarrow）
- HuggingFace Dataset
- 自定义 Dataset 类
"""

import json
from pathlib import Path
from typing import Any, Iterator


class SupervisedDataset:
    """监督学习数据集封装.

    支持任意格式的 (observations, actions) 对，
    提供统一的迭代接口。

    Usage::

        dataset = SupervisedDataset.load("expert_demos.jsonl")
        for batch in dataset.batch_iter(batch_size=256, shuffle=True):
            obs, actions = batch["obs"], batch["actions"]
            ...

        # 训练/验证集划分
        train_ds, val_ds = dataset.split(train_ratio=0.9)
    """

    def __init__(
        self,
        observations,
        actions,
        rewards=None,
        terminals=None,
        weights=None,
        metadata: dict | None = None,
    ):
        import numpy as np

        assert len(observations) == len(actions)
        self.observations = np.array(observations, dtype=np.float32)
        self.actions = np.array(actions)
        self.rewards = np.array(rewards) if rewards is not None else None
        self.terminals = np.array(terminals) if terminals is not None else None
        self.weights = np.array(weights) if weights is not None else None
        self.metadata = metadata or {}
        self._n = len(self.observations)

    def __len__(self) -> int:
        return self._n

    @classmethod
    def load(
        cls,
        path: str,
        obs_key: str = "observations",
        action_key: str = "actions",
        reward_key: str | None = "rewards",
        terminal_key: str | None = "dones",
        weight_key: str | None = "weights",
        normalize_obs: bool = False,
        filter_terminals: bool = False,
    ) -> "SupervisedDataset":
        """从文件加载数据集。

        自动检测格式：JSON, JSONL, Parquet, NumPy .npz

        Args:
            path: 数据文件路径
            obs_key / action_key: 字段名
            reward_key / terminal_key / weight_key: 可选字段
            normalize_obs: 是否对观测做标准化
            filter_terminals: 是否过滤 terminal=True 之后的数据
        """
        p = Path(path)

        if p.suffix == ".jsonl":
            data = cls._load_jsonl(path)
        elif p.suffix == ".json":
            data = cls._load_json(path)
        elif p.suffix == ".parquet":
            data = cls._load_parquet(path)
        elif p.suffix == ".npz":
            data = cls._load_npz(path)
        else:
            raise ValueError(f"Unsupported format: {p.suffix}")

        obs = cls._to_float32(data[obs_key])
        actions = data[action_key]

        rewards = cls._to_float32(data[reward_key]) if reward_key and reward_key in data else None
        terminals = data[terminal_key] if terminal_key and terminal_key in data else None
        weights = cls._to_float32(data[weight_key]) if weight_key and weight_key in data else None

        if normalize_obs:
            mean = obs.mean(axis=0)
            std = obs.std(axis=0) + 1e-8
            obs = (obs - mean) / std

        if filter_terminals and terminals is not None:
            mask = np.concatenate([[True], terminals[:-1]])
            obs = obs[mask]
            actions = actions[mask]
            if rewards is not None:
                rewards = rewards[mask]

        return cls(
            observations=obs,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            weights=weights,
            metadata={"source": str(path), "size": len(obs)},
        )

    @staticmethod
    def _to_float32(arr) -> Any:
        import numpy as np
        return np.array(arr, dtype=np.float32)

    @staticmethod
    def _load_jsonl(path: str) -> dict:
        samples: dict[str, list] = {"observations": [], "actions": []}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                obs = item.get("obs", item.get("observation"))
                if isinstance(obs, list):
                    obs = [float(x) for x in obs]
                samples["observations"].append(obs)
                samples["actions"].append(item.get("action"))
        return samples

    @staticmethod
    def _load_json(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_parquet(path: str) -> dict:
        try:
            import pandas as pd
            df = pd.read_parquet(path)
            return {col: df[col].values for col in df.columns}
        except ImportError as exc:
            raise ImportError(
                "pandas + pyarrow are required for Parquet support. "
                "Install with: pip install pandas pyarrow"
            ) from exc

    @staticmethod
    def _load_npz(path: str) -> dict:
        import numpy as np
        data = np.load(path)
        return {k: data[k].tolist() for k in data.files}

    def batch_iter(
        self,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        device: str | None = None,
    ) -> Iterator[dict]:
        """生成 batch 迭代器。

        Args:
            batch_size: 批大小
            shuffle: 是否在每个 epoch 前打乱数据
            drop_last: 是否丢弃最后不完整的 batch
            device: 自动将数据移到指定设备（"cuda:0" 等）
        """
        import numpy as np
        import torch

        n = self._n
        indices = np.random.permutation(n) if shuffle else np.arange(n)

        for i in range(0, n, batch_size):
            batch_indices = indices[i : i + batch_size]
            if drop_last and len(batch_indices) < batch_size:
                continue

            obs = self.observations[batch_indices]
            acts = self.actions[batch_indices]

            batch: dict[str, Any] = {
                "obs": torch.FloatTensor(obs),
                "actions": torch.LongTensor(acts)
                if acts.dtype.kind in ("i", "u")
                else torch.FloatTensor(acts),
            }

            if self.rewards is not None:
                batch["rewards"] = torch.FloatTensor(self.rewards[batch_indices])
            if self.terminals is not None:
                batch["terminals"] = torch.BoolTensor(self.terminals[batch_indices])
            if self.weights is not None:
                batch["weights"] = torch.FloatTensor(self.weights[batch_indices])

            batch["indices"] = batch_indices

            if device:
                batch = {
                    k: v.to(device) if hasattr(v, "to") else v
                    for k, v in batch.items()
                }

            yield batch

    def split(
        self,
        train_ratio: float = 0.9,
        shuffle: bool = True,
    ) -> tuple["SupervisedDataset", "SupervisedDataset"]:
        """按比例划分训练集 / 验证集。"""
        import numpy as np

        n = len(self)
        split_idx = int(n * train_ratio)
        perm = np.random.permutation(n) if shuffle else np.arange(n)
        train_idx, val_idx = perm[:split_idx], perm[split_idx:]

        def subset(idx) -> "SupervisedDataset":
            return SupervisedDataset(
                observations=self.observations[idx],
                actions=self.actions[idx],
                rewards=self.rewards[idx] if self.rewards is not None else None,
                terminals=self.terminals[idx] if self.terminals is not None else None,
                weights=self.weights[idx] if self.weights is not None else None,
                metadata={**self.metadata},
            )

        return subset(train_idx), subset(val_idx)


def load_supervised_dataset(
    path: str,
    **kwargs,
) -> SupervisedDataset:
    """便捷函数：加载监督数据集。

    等价于 ``SupervisedDataset.load(path, **kwargs)``
    """
    return SupervisedDataset.load(path, **kwargs)


__all__ = [
    "SupervisedDataset",
    "load_supervised_dataset",
]
