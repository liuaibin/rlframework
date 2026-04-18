"""SupervisedAlgorithmMixin - 监督学习算法的框架钩子扩展点.

继承自 FrameworkAlgorithmMixin，额外增加监督学习特有的扩展点：
- 数据加载与预处理
- 损失函数构建
- 评估指标
"""

from typing import Any

from ray.rllib.utils.typing import ResultDict

from rlframework.algorithms.base import FrameworkAlgorithmMixin


class SupervisedAlgorithmMixin(FrameworkAlgorithmMixin):
    """Mixin for supervised learning algorithms (BC, MARWIL).

    在 CustomBC / CustomMARWIL 中使用，与 FrameworkAlgorithmMixin
    的 hook 完全兼容。

    典型用法::

        class MyBC(SupervisedAlgorithmMixin, BC):
            def build_extra_model_config(self):
                return {"custom_model": "my_supervised_model"}

            def load_supervised_data(self, data_path):
                return load_offline_dataset(data_path)

            def compute_supervised_loss(self, batch, module):
                # 自定义监督损失
                return F.cross_entropy(logits, labels)
    """

    # ------------------------------------------------------------------
    # 数据相关钩子
    # ------------------------------------------------------------------

    def load_supervised_data(self, data_path: str | None) -> Any:
        """加载监督学习数据集。

        Args:
            data_path: 数据路径，支持：
                - JSON / JSONL 文件（离线 RL 数据集）
                - Directory（多文件数据集）
                - Parquet 文件
                - HuggingFace Dataset URI

        Returns:
            数据集对象（格式由子类决定）

        Raises:
            FileNotFoundError: 数据路径不存在
            ValueError: 数据格式不支持
        """
        if data_path is None:
            return None
        import os
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Supervised data not found: {data_path}"
            )
        return self._default_load(data_path)

    def _default_load(self, path: str) -> dict:
        """默认数据加载：自动检测格式。

        支持格式:
            - .json / .jsonl: RLlib 离线数据格式
            - .npz: NumPy 压缩格式
        """
        import json

        if path.endswith(".jsonl"):
            return self._load_jsonl(path)
        elif path.endswith(".json"):
            return self._load_json(path)
        elif path.endswith(".npz"):
            return self._load_npz(path)
        raise ValueError(f"Unsupported data format: {path}")

    def _load_jsonl(self, path: str) -> dict:
        """加载 JSONL 格式（每行一个样本）。"""
        import json

        samples = {"observations": [], "actions": [], "rewards": []}
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                obs = item.get("obs", item.get("observation"))
                act = item.get("action")
                rew = item.get("reward")
                if isinstance(obs, list):
                    obs = [float(x) for x in obs]
                samples["observations"].append(obs)
                samples["actions"].append(act)
                if rew is not None:
                    samples["rewards"].append(rew)
        return samples

    def _load_json(self, path: str) -> dict:
        """加载 JSON 格式。"""
        import json

        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _load_npz(self, path: str) -> dict:
        """加载 NumPy .npz 压缩格式。"""
        import numpy as np

        data = np.load(path)
        result = {k: data[k].tolist() for k in data.files}
        return result

    def preprocess_supervised_batch(self, batch: dict) -> dict:
        """预处理单个 batch（如归一化、数据增强）。

        Args:
            batch: 原始 batch dict，应包含:
                - observations: (B, *obs_shape)
                - actions: (B,) 或 (B, *action_shape)
                - 可选: rewards, terminals, weights 等

        Returns:
            预处理后的 batch dict
        """
        return batch

    def create_supervised_iterator(
        self,
        data_path: str | None,
        batch_size: int,
    ):
        """创建数据迭代器。

        默认实现使用 PyTorch DataLoader。
        子类可覆写以支持其他数据加载方式。

        Args:
            data_path: 数据路径
            batch_size: 批大小

        Returns:
            可迭代对象，每次 yield 一个 batch dict
        """
        from torch.utils.data import DataLoader, TensorDataset

        data = self.load_supervised_data(data_path)
        if data is None:
            return iter([])

        import numpy as np
        import torch

        obs = data.get("observations")
        actions = data.get("actions")

        if obs is None or actions is None:
            return iter([])

        obs_arr = np.array(obs, dtype=np.float32)
        act_arr = np.array(actions)

        # 动作类型：离散用 long，连续用 float
        if act_arr.dtype.kind in ("i", "u"):
            act_tensor = torch.LongTensor(act_arr)
        else:
            act_tensor = torch.FloatTensor(act_arr)

        dataset = TensorDataset(
            torch.FloatTensor(obs_arr),
            act_tensor,
        )
        return iter(
            DataLoader(dataset, batch_size=batch_size, shuffle=True)
        )

    # ------------------------------------------------------------------
    # 损失函数钩子
    # ------------------------------------------------------------------

    def compute_supervised_loss(
        self,
        batch: dict,
        module,
        **kwargs,
    ):
        """计算监督损失。

        Args:
            batch: 当前 batch dict
            module: RLModule 实例

        Returns:
            (loss_value, loss_components_dict)

        默认实现根据动作空间自动选择：
            - 离散动作 -> CrossEntropyLoss
            - 连续动作 -> MSELoss
        """
        import torch
        import torch.nn.functional as F

        obs = batch["obs"]
        actions = batch["actions"]

        # Forward pass through RLModule
        logits = self._forward_supervised(module, obs)

        # 离散动作：交叉熵
        if actions.dtype in (torch.int64, torch.long):
            loss = F.cross_entropy(logits, actions)
            loss_components = {"cross_entropy_loss": float(loss)}
        else:
            # 连续动作：MSE
            if logits.shape == actions.shape:
                loss = F.mse_loss(logits, actions)
                loss_components = {"mse_loss": float(loss)}
            else:
                raise ValueError(
                    f"Shape mismatch: logits {logits.shape} vs "
                    f"actions {actions.shape}. Override compute_supervised_loss() "
                    "for custom behavior."
                )

        return float(loss), loss_components

    def _forward_supervised(self, module, obs):
        """通过 RLModule 执行前向传播，返回策略 logits。

        适配不同的 RLModule 架构。
        """
        import torch

        if hasattr(module, "get_inference_actions"):
            return module.get_inference_actions(obs)

        if hasattr(module, "encoder") and hasattr(module, "pi"):
            enc_out = module.encoder({"obs": obs})
            latent = enc_out.get("encoder_out", {})
            if isinstance(latent, dict):
                latent = latent.get("actor", latent.get("critic"))
            if isinstance(latent, tuple):
                latent = latent[0]
            return module.pi(latent)

        raise NotImplementedError(
            f"Cannot compute supervised loss for module type {type(module)}. "
            "Override _forward_supervised() to provide custom logic."
        )

    # ------------------------------------------------------------------
    # 评估指标钩子
    # ------------------------------------------------------------------

    def compute_supervised_metrics(
        self,
        batch: dict,
        module,
        predictions,
    ) -> dict[str, float]:
        """计算监督学习评估指标。

        Args:
            batch: 当前 batch
            module: RLModule
            predictions: 模型输出

        Returns:
            指标 dict
        """
        import torch

        actions = batch["actions"]
        logits = predictions

        metrics = {}

        if logits.dtype in (torch.int64, torch.long):
            preds = torch.argmax(logits, dim=-1)
            accuracy = (preds == actions).float().mean().item()
            metrics["sup_accuracy"] = accuracy

            # Top-3 准确率（多分类，>=3 个类别）
            if logits.shape[-1] >= 3:
                _, top3 = torch.topk(logits, k=3, dim=-1)
                top3_acc = (
                    (top3 == actions.unsqueeze(-1)).any(dim=-1).float().mean().item()
                )
                metrics["sup_top3_accuracy"] = top3_acc

        else:
            mse = torch.nn.functional.mse_loss(logits, actions).item()
            metrics["sup_mse"] = mse

        return metrics

    # ------------------------------------------------------------------
    # 训练步钩子（扩展 FrameworkAlgorithmMixin）
    # ------------------------------------------------------------------

    def on_after_supervised_step(
        self,
        loss_components: dict,
        metrics: dict,
    ) -> None:
        """每次监督学习训练步完成后的回调。

        可用于:
            - 日志记录
            - 学习率调度
            - 早停检查

        Args:
            loss_components: 损失分量 dict
            metrics: 评估指标 dict
        """
        for key, value in {**loss_components, **metrics}.items():
            if isinstance(value, (int, float)):
                self.metrics.log_value(f"supervised/{key}", value, window=1)
