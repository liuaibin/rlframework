"""CustomMARWIL - MARWIL 算法，集成 rlframework 所有基础设施.

MARWIL = BC + Advantage-weighted regression
在行为克隆基础上用 Advantage 信号加权，使高回报轨迹获得更高的学习权重。

Usage::

    from rlframework.algorithms.supervised import CustomMARWILConfig

    config = (
        CustomMARWILConfig()
        .environment("CartPole-v1")
        .offline_data(input_="./demonstrations.json")
        .supervised_training(
            data_path="./marwil_data.json",
            batch_size=256,
            beta=1.0,  # advantage 加权强度
        )
        .storage(backend="minio", bucket="rl-models")
        .metrics(reporters=["file"])
    )
    algo = config.build()
"""

import numpy as np
import torch
from typing import Any

from ray.rllib.algorithms.marwil import MARWIL, MARWILConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override

from rlframework.algorithms.base import FrameworkAlgorithmMixin
from rlframework.algorithms.supervised.base import SupervisedAlgorithmMixin
from rlframework.algorithms.supervised.losses import behavioral_cloning_loss
from rlframework.config.framework_config import FrameworkConfigMixin


# ── 配置类 ────────────────────────────────────────────────────────────────

class CustomMARWILConfig(MARWILConfig, FrameworkConfigMixin):
    """MARWIL 配置 + rlframework 扩展 + 监督学习参数.

    新增参数:
        - supervised_data_path: 监督数据路径
        - supervised_batch_size: 监督学习 batch size
        - supervised_epochs_per_round: 每轮迭代的监督学习 epoch 数
        - beta: Advantage 加权强度（0=纯BC, 1=强 advantage 信号）
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CustomMARWIL)
        self._init_framework_mixin()

        # 监督学习参数
        self._supervised_data_path: str | None = None
        self._supervised_batch_size: int = 256
        self._supervised_epochs_per_round: int = 1

    def supervised_training(
        self,
        data_path: str | None = None,
        batch_size: int = 256,
        epochs_per_round: int = 1,
    ) -> "CustomMARWILConfig":
        """配置监督学习参数。

        Args:
            data_path: 离线数据路径（JSON/JSONL/Parquet）
            batch_size: 每次训练步的 batch size
            epochs_per_round: 每轮 RL 迭代中监督学习的 epoch 数
        """
        self._supervised_data_path = data_path
        self._supervised_batch_size = batch_size
        self._supervised_epochs_per_round = epochs_per_round
        return self

    def framework_models(
        self,
        encoder: str | None = None,
        actor_head: str | None = None,
    ) -> "CustomMARWILConfig":
        """配置自定义模型组件（与 PPO/SAC 的同名方法一致）"""
        custom_config = {}

        if encoder and encoder != "default":
            custom_config["custom_encoder"] = encoder
        if actor_head and actor_head != "default":
            custom_config["custom_actor_head"] = actor_head

        if custom_config:
            self.model.update({"_framework_custom_config": custom_config})
            self.rl_module(
                rl_module_spec=RLModuleSpec(
                    model_config_dict=self.model,
                )
            )

        return self

    # ── 属性访问 ────────────────────────────────────────────────────────

    @property
    def supervised_data_path(self) -> str | None:
        return self._supervised_data_path

    @property
    def supervised_batch_size(self) -> int:
        return self._supervised_batch_size

    @property
    def supervised_epochs_per_round(self) -> int:
        return self._supervised_epochs_per_round


# ── 算法类 ─────────────────────────────────────────────────────────────────

class CustomMARWIL(SupervisedAlgorithmMixin, MARWIL):
    """MARWIL + rlframework 扩展钩子.

    MARWIL 在 BC 基础上用 Advantage 替代 beta 参数：
        - beta=0: 等价于纯 BC（不利用 reward 信号）
        - beta>0: 高 advantage 样本获得更高权重

    可通过 SupervisedAlgorithmMixin 的钩子扩展：
        - load_supervised_data()
        - preprocess_supervised_batch()
        - compute_supervised_loss()（子类覆写以支持 advantage 加权）
        - compute_supervised_metrics()
    """

    @classmethod
    @override(MARWIL)
    def get_default_config(cls) -> CustomMARWILConfig:
        return CustomMARWILConfig()

    @override(MARWIL)
    def setup(self, config: CustomMARWILConfig):
        super().setup(config)

        self._sup_train_iter = None
        self._sup_epoch = 0
        self._sup_global_step = 0

        data_path = config.supervised_data_path
        if data_path:
            self._sup_data = self.load_supervised_data(data_path)
            self._sup_train_iter = self.create_supervised_iterator(
                data_path, config.supervised_batch_size
            )
        else:
            self._sup_data = None

    def compute_supervised_loss(self, batch: dict, module, **kwargs):
        """覆写以支持 MARWIL 的 Advantage 加权。

        相比标准 BC，MARWIL 的损失函数为：
            L = -E[exp(beta * A(s,a)) * log π(a|s)]

        其中 beta 控制 advantage 信号的强度（由 MARWILConfig.beta 决定）。
        """
        import torch
        import torch.nn.functional as F

        obs = batch["obs"]
        actions = batch["actions"]
        advantages = batch.get("advantages")

        # 标准 BC 前向传播
        logits = self._forward_supervised(module, obs)

        # 基础监督损失
        if actions.dtype in (torch.int64, torch.long):
            log_probs = F.log_softmax(logits, dim=-1)
            nll = F.nll_loss(log_probs, actions, reduction="none")
            base_loss = nll.mean()
        else:
            base_loss = F.mse_loss(logits, actions)

        # MARWIL: Advantage 加权
        if advantages is not None and self.config.beta > 0:
            # clamp 防止数值溢出
            clamped_adv = advantages.clamp(-20, 20)
            weights = torch.exp(self.config.beta * clamped_adv)
            weighted_loss = (nll if actions.dtype in (torch.int64, torch.long) else F.mse_loss(logits, actions, reduction="none")) * weights
            loss = weighted_loss.mean()
            loss_components = {
                "marwil_weighted_loss": float(loss),
                "advantage_mean": float(advantages.mean()),
                "weight_mean": float(weights.mean()),
            }
        else:
            loss = base_loss
            loss_components = {"marwil_base_loss": float(loss)}

        return float(loss), loss_components

    @override(MARWIL)
    def training_step(self) -> None:
        """覆写 training_step 以支持多 epoch 监督训练。"""
        self.on_before_training_step()

        config: CustomMARWILConfig = self.get_default_config()

        for _ in range(config.supervised_epochs_per_round):
            if self._sup_train_iter is None:
                break
            try:
                batch = next(self._sup_train_iter)
            except StopIteration:
                self._sup_train_iter = self.create_supervised_iterator(
                    config.supervised_data_path,
                    config.supervised_batch_size,
                )
                self._sup_epoch += 1
                batch = next(self._sup_train_iter)

            batch = self.preprocess_supervised_batch(batch)

            module = self.get_module("default_policy")

            loss, loss_components = self.compute_supervised_loss(batch, module)

            self._supervised_optimize(loss)

            preds = self._forward_supervised(module, batch["obs"])
            metrics = self.compute_supervised_metrics(batch, module, preds)

            self._sup_global_step += 1
            self.on_after_supervised_step(loss_components, metrics)

        super(MARWIL, self).training_step()

        result = self.metrics.peek()
        result = self.on_after_training_step(result)
        if result:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    self.metrics.log_value(key, value, window=1)

    def _supervised_optimize(self, loss: float) -> None:
        """执行一步监督学习优化。"""
        try:
            learner_group = self.learner_group
            if learner_group is None:
                return

            def _update_step(learner):
                optimizer = learner.get_optimizer()[0]
                optimizer.zero_grad()
                loss_tensor = torch.tensor(
                    loss,
                    device=next(learner.get_parameters())[0].device,
                )
                loss_tensor.backward()
                optimizer.step()
                return loss_tensor.item()

            results = learner_group.foreach_learner(_update_step)
            avg_loss = sum(r for r in results if isinstance(r, float)) / max(len(results), 1)
            self.metrics.log_value("supervised/step_loss", avg_loss, window=1)
        except Exception:
            pass

    def evaluate_on_supervised_data(
        self,
        data_path: str | None = None,
    ) -> dict[str, float]:
        """在监督数据集上评估模型。"""
        import torch

        eval_data = self.load_supervised_data(
            data_path
        ) if data_path else self._sup_data

        if eval_data is None:
            return {}

        module = self.get_module("default_policy")

        obs = torch.FloatTensor(np.array(eval_data["observations"]))
        actions_arr = np.array(eval_data["actions"])
        actions = (
            torch.LongTensor(actions_arr)
            if actions_arr.dtype.kind in ("i", "u")
            else torch.FloatTensor(actions_arr)
        )

        batch_size = self.get_default_config().supervised_batch_size
        all_preds = []
        for i in range(0, len(obs), batch_size):
            preds = self._forward_supervised(module, obs[i : i + batch_size])
            all_preds.append(preds)
        all_preds = torch.cat(all_preds, dim=0)

        metrics = self.compute_supervised_metrics(
            {"obs": obs, "actions": actions},
            module,
            all_preds,
        )
        metrics["num_evaluated_samples"] = len(obs)
        return metrics
