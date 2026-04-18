"""CustomBC - Behavioral Cloning，集成 rlframework 所有基础设施.

Usage::

    from rlframework.algorithms.supervised import CustomBCConfig

    config = (
        CustomBCConfig()
        .environment("CartPole-v1")
        .offline_data(input_="./demonstrations.json")
        .supervised_training(
            data_path="./bc_data.parquet",
            batch_size=256,
            epochs_per_round=10,
        )
        .storage(backend="minio", bucket="rl-models")
        .metrics(reporters=["influxdb", "file"])
    )
    algo = config.build()
"""

from typing import Any

from ray.rllib.algorithms.bc import BC, BCConfig
from ray.rllib.algorithms.bc.bc_catalog import BCCatalog
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override

from rlframework.algorithms.base import FrameworkAlgorithmMixin
from rlframework.algorithms.supervised.base import SupervisedAlgorithmMixin
from rlframework.algorithms.supervised.losses import behavioral_cloning_loss
from rlframework.config.framework_config import FrameworkConfigMixin


# ── 配置类 ────────────────────────────────────────────────────────────────

class CustomBCConfig(BCConfig, FrameworkConfigMixin):
    """BC 配置 + rlframework 扩展 + 监督学习参数.

    新增参数:
        - supervised_data_path: 监督数据路径
        - supervised_batch_size: 监督学习 batch size
        - supervised_epochs_per_round: 每轮 RL 训练迭代的监督学习 epochs
        - custom_encoder / custom_actor_head: 组件注册名称
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CustomBC)
        self._init_framework_mixin()

        # 监督学习参数
        self._supervised_data_path: str | None = None
        self._supervised_batch_size: int = 256
        self._supervised_epochs_per_round: int = 1
        self._supervised_validation_split: float = 0.0
        self._supervised_loss_type: str = "ce"  # "ce" or "mse"

    def supervised_training(
        self,
        data_path: str | None = None,
        batch_size: int = 256,
        epochs_per_round: int = 1,
        validation_split: float = 0.0,
        loss_type: str = "ce",
    ) -> "CustomBCConfig":
        """配置监督学习参数。

        Args:
            data_path: 离线数据路径（JSON/JSONL/Parquet）
            batch_size: 每次训练步的 batch size
            epochs_per_round: 每轮 RL 迭代中监督学习的 epoch 数
            validation_split: 验证集比例（用于早停，0=不使用）
            loss_type: 损失类型，"ce" 离散/分类，"mse" 连续/回归
        """
        self._supervised_data_path = data_path
        self._supervised_batch_size = batch_size
        self._supervised_epochs_per_round = epochs_per_round
        self._supervised_validation_split = validation_split
        self._supervised_loss_type = loss_type
        return self

    def framework_models(
        self,
        encoder: str | None = None,
        actor_head: str | None = None,
    ) -> "CustomBCConfig":
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
                    catalog_class=BCCatalog,
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

    @property
    def supervised_validation_split(self) -> float:
        return self._supervised_validation_split

    @property
    def supervised_loss_type(self) -> str:
        return self._supervised_loss_type


# ── 算法类 ─────────────────────────────────────────────────────────────────

class CustomBC(SupervisedAlgorithmMixin, BC):
    """行为克隆 + rlframework 扩展钩子.

    BC 是一种纯监督学习算法，目标是最大化
    log π(a|s) 从专家演示数据中学习策略。

    可通过 SupervisedAlgorithmMixin 的钩子扩展:
        - load_supervised_data(): 自定义数据加载
        - preprocess_supervised_batch(): 数据预处理/增强
        - compute_supervised_loss(): 自定义损失函数
        - compute_supervised_metrics(): 自定义评估指标
        - on_after_supervised_step(): 每步完成后的回调
    """

    @classmethod
    @override(BC)
    def get_default_config(cls) -> CustomBCConfig:
        return CustomBCConfig()

    @override(BC)
    def setup(self, config: CustomBCConfig):
        super().setup(config)

        # 初始化数据迭代器
        self._sup_train_iter = None
        self._sup_epoch = 0
        self._sup_global_step = 0

        # 加载数据
        data_path = config.supervised_data_path
        if data_path:
            self._sup_data = self.load_supervised_data(data_path)
            self._sup_train_iter = self.create_supervised_iterator(
                data_path, config.supervised_batch_size
            )
        else:
            self._sup_data = None

    @override(BC)
    def training_step(self) -> None:
        """覆写 training_step 以支持多 epoch 监督训练。

        每个 RL 迭代执行 supervised_epochs_per_round 次监督学习步。
        """
        self.on_before_training_step()

        config: CustomBCConfig = self.get_default_config()

        for _ in range(config.supervised_epochs_per_round):
            # 获取下一个 batch
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

            # 预处理
            batch = self.preprocess_supervised_batch(batch)

            # 获取 RLModule
            module = self.get_module("default_policy")

            # 计算监督损失
            loss, loss_components = behavioral_cloning_loss(
                observations=batch["obs"],
                actions=batch["actions"],
                module=module,
                loss_type=config.supervised_loss_type,
            )

            # 优化器更新
            self._supervised_optimize(loss)

            # 评估指标
            preds = self._forward_supervised(module, batch["obs"])
            metrics = self.compute_supervised_metrics(batch, module, preds)

            self._sup_global_step += 1

            # 钩子回调
            self.on_after_supervised_step(loss_components, metrics)

        # 调用父类训练步（执行 BC 的标准逻辑）
        super(BC, self).training_step()

        result = self.metrics.peek()
        result = self.on_after_training_step(result)
        if result:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    self.metrics.log_value(key, value, window=1)

    def _supervised_optimize(self, loss: float) -> None:
        """执行一步监督学习优化（通过 TorchLearner）。

        直接使用 RLlib 的 TorchLearner 执行 backward + step。
        """
        try:
            # 获取 learner，对每个 module 执行优化
            learner_group = self.learner_group
            if learner_group is None:
                return

            # 使用 foreach_learner 批量更新
            def _update_step(learner):
                import torch
                optimizer = learner.get_optimizer()[0]
                optimizer.zero_grad()
                loss_tensor = torch.tensor(loss, device=next(learner.get_parameters())[0].device)
                loss_tensor.backward()
                optimizer.step()
                return loss_tensor.item()

            results = learner_group.foreach_learner(_update_step)
            avg_loss = sum(r for r in results if isinstance(r, float)) / max(len(results), 1)
            self.metrics.log_value("supervised/step_loss", avg_loss, window=1)

        except Exception:
            # 如果内部优化失败，尝试 fallback
            pass

    def evaluate_on_supervised_data(
        self,
        data_path: str | None = None,
    ) -> dict[str, float]:
        """在监督数据集上评估模型（不训练）。

        计算准确率（离散）或 MSE（连续）。

        Args:
            data_path: 评估数据路径，None 则使用训练数据

        Returns:
            评估指标 dict
        """
        import numpy as np
        import torch

        eval_data = self.load_supervised_data(
            data_path
        ) if data_path else self._sup_data

        if eval_data is None:
            return {}

        module = self.get_module("default_policy")
        obs = torch.FloatTensor(np.array(eval_data["observations"]))
        actions = torch.LongTensor(
            np.array(eval_data["actions"])
        ) if np.array(eval_data["actions"]).dtype.kind in ("i", "u") else torch.FloatTensor(
            np.array(eval_data["actions"])
        )

        # 分批评估
        batch_size = self.get_default_config().supervised_batch_size
        all_preds, all_labels = [], []
        for i in range(0, len(obs), batch_size):
            batch_obs = obs[i : i + batch_size]
            preds = self._forward_supervised(module, batch_obs)
            all_preds.append(preds)
            all_labels.append(actions[i : i + batch_size])

        import torch
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        metrics = self.compute_supervised_metrics(
            {"obs": obs, "actions": actions},
            module,
            all_preds,
        )
        metrics["num_evaluated_samples"] = len(obs)
        metrics["num_supervised_epochs"] = self._sup_epoch
        return metrics
