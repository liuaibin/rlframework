"""rlframework.algorithms.supervised - 监督学习算法模块.

提供 BC (Behavioral Cloning) 和 MARWIL 算法的框架集成版本，
继承所有 rlframework 基础设施（存储、日志、模型组件注册）。

Usage::

    from rlframework.algorithms.supervised import CustomBC, CustomBCConfig

    config = (
        CustomBCConfig()
        .environment("CartPole-v1")
        .offline_data(input_="./expert_demos.json")
        .supervised_training(
            data_path="./bc_data.json",
            batch_size=256,
            epochs_per_round=10,
        )
        .storage(backend="minio", bucket="models")
        .metrics(reporters=["file"])
    )
    algo = config.build()

    for epoch in range(100):
        algo.train()
        # 评估
        metrics = algo.evaluate_on_supervised_data()
        print(f"accuracy: {metrics.get('sup_accuracy', 'N/A')}")
"""

from rlframework.algorithms.supervised.base import SupervisedAlgorithmMixin
from rlframework.algorithms.supervised.bc import CustomBC, CustomBCConfig
from rlframework.algorithms.supervised.marwil import CustomMARWIL, CustomMARWILConfig
from rlframework.algorithms.supervised.losses import (
    behavioral_cloning_loss,
    cross_entropy_loss,
    huber_loss,
    mixup_supervised_loss,
    mse_loss,
    weighted_supervised_loss,
)
from rlframework.algorithms.supervised.data_utils import (
    SupervisedDataset,
    load_supervised_dataset,
)

__all__ = [
    # Mixin
    "SupervisedAlgorithmMixin",
    # BC
    "CustomBC",
    "CustomBCConfig",
    # MARWIL
    "CustomMARWIL",
    "CustomMARWILConfig",
    # 损失函数
    "behavioral_cloning_loss",
    "cross_entropy_loss",
    "huber_loss",
    "mse_loss",
    "mixup_supervised_loss",
    "weighted_supervised_loss",
    # 数据工具
    "SupervisedDataset",
    "load_supervised_dataset",
]
