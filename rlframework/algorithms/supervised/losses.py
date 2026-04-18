"""监督学习损失函数库.

提供常用的监督损失函数，支持自定义组合和加权。
"""

import torch
import torch.nn.functional as F
from typing import Literal


def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weight: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """交叉熵损失（支持样本权重和标签平滑）。

    Args:
        logits: (B, num_classes) 模型输出的未归一化 logit
        targets: (B,) 目标类别索引
        weight: (B,) 每个样本的可选权重（用于 Importance Sampling）
        label_smoothing: 标签平滑系数 [0, 1)

    Returns:
        标量损失张量
    """
    return F.cross_entropy(logits, targets, weight=weight, label_smoothing=label_smoothing)


def mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    """均方误差损失（用于连续动作/回归任务）。

    Args:
        predictions: (B, *) 模型预测
        targets: (B, *) 目标值
        reduction: 归约方式

    Returns:
        标量损失张量
    """
    return F.mse_loss(predictions, targets, reduction=reduction)


def huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Huber 损失（对异常值更稳健的回归损失）。

    Args:
        predictions: (B, *) 模型预测
        targets: (B, *) 目标值
        delta: Huber 损失的拐点

    Returns:
        标量损失张量
    """
    return F.smooth_l1_loss(predictions, targets, beta=delta, reduction="mean")


def weighted_supervised_loss(
    rl_loss: torch.Tensor,
    sup_loss: torch.Tensor,
    sup_weight: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """将监督损失与 RL 损失加权合并。

    Args:
        rl_loss: 强化学习损失（策略 + 价值损失之和）
        sup_loss: 监督学习损失（交叉熵/MSE/Huber）
        sup_weight: 监督损失的相对权重
        normalize: 是否对合并后的损失做归一化

    Returns:
        combined_loss: 加权合并后的总损失

    数学形式:
        combined = rl_loss + sup_weight * sup_loss

    当 normalize=True 时:
        combined = (rl_loss + sup_weight * sup_loss) / (1 + sup_weight)
    """
    combined = rl_loss + sup_weight * sup_loss
    if normalize:
        combined = combined / (1 + sup_weight)
    return combined


def mixup_supervised_loss(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    forward_fn,
    alpha: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MixUp 数据增强 + 监督损失。

    MixUp 通过混合样本和标签来正则化模型，提升泛化能力。
    适用于监督学习和行为克隆。

    Args:
        batch_x: (B, *obs_shape) 输入观测
        batch_y: (B,) 标签（离散动作索引）
        forward_fn: 可调用对象，接收 batch_x 返回 logits
        alpha: Beta 分布参数，控制混合强度

    Returns:
        (loss, mixed_labels) — MixUp 损失和混合标签（用于日志记录）
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().to(batch_x.device)
    else:
        lam = 1.0

    batch_size = batch_x.size(0)
    index = torch.randperm(batch_size, device=batch_x.device)

    mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
    y_a, y_b = batch_y, batch_y[index]

    logits = forward_fn(mixed_x)

    loss = lam * F.cross_entropy(logits, y_a) + (1 - lam) * F.cross_entropy(logits, y_b)

    return loss, (lam, y_a, y_b)


def behavioral_cloning_loss(
    observations: torch.Tensor,
    actions: torch.Tensor,
    module,
    loss_type: Literal["ce", "mse"] = "ce",
    action_space_type: Literal["discrete", "continuous", "auto"] = "auto",
) -> tuple[torch.Tensor, dict]:
    """标准行为克隆损失。

    根据动作空间类型自动选择损失函数：
        - 离散动作空间: CrossEntropy
        - 连续动作空间: MSE

    Args:
        observations: (B, *obs_shape)
        actions: (B,) 离散 / (B, action_dim) 连续
        module: 包含 encoder + pi 的 RLModule
        loss_type: 强制指定损失类型（覆盖自动推断）
        action_space_type: 强制指定动作空间类型

    Returns:
        (loss, components_dict)
    """
    # Forward
    if hasattr(module, "encoder") and hasattr(module, "pi"):
        enc_out = module.encoder({"obs": observations})
        latent = enc_out.get("encoder_out", {})
        if isinstance(latent, dict):
            latent = latent.get("actor", latent.get("critic"))
        if isinstance(latent, tuple):
            latent = latent[0]
        logits = module.pi(latent)
    else:
        raise NotImplementedError(
            f"Module {type(module)} does not support standard BC loss. "
            "Provide a custom compute_supervised_loss() implementation."
        )

    # 自动推断动作空间类型
    if action_space_type == "auto":
        if actions.dtype in (torch.int64, torch.long):
            action_space_type = "discrete"
        else:
            action_space_type = "continuous"

    # 计算损失
    if loss_type == "ce" or action_space_type == "discrete":
        loss = F.cross_entropy(logits, actions)
        components = {"bc_cross_entropy_loss": float(loss)}
    elif loss_type == "mse" or action_space_type == "continuous":
        loss = F.mse_loss(logits, actions)
        components = {"bc_mse_loss": float(loss)}
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss, components


__all__ = [
    "behavioral_cloning_loss",
    "cross_entropy_loss",
    "huber_loss",
    "mixup_supervised_loss",
    "mse_loss",
    "weighted_supervised_loss",
]
