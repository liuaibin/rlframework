# RLlib SAC 训练采样动作链路分析

本文记录 RLlib new API stack 下，SAC 在训练采样时如何从策略网络输出动作，并转换成 `env.step()` 可用的环境动作。

分析基于本机 RLlib 源码：

```text
/Users/lab/Library/Python/3.9/lib/python/site-packages/ray/rllib/
```

## 总览

SAC 训练采样时的动作链路可以简化为：

```text
SAC.train()
  -> DQN._training_step_new_api_stack()
  -> synchronous_parallel_sample()
  -> SingleAgentEnvRunner.sample()
  -> SingleAgentEnvRunner._sample()
  -> env-to-module connector: obs -> tensor batch
  -> SAC RLModule.forward_exploration()
  -> policy head 输出 ACTION_DIST_INPUTS
  -> module-to-env connector: GetActions
  -> TorchSquashedGaussian.sample()
  -> tanh 得到 normalized action [-1, 1]
  -> NormalizeAndClipActions
  -> unsquash_action: [-1, 1] -> env.action_space.low/high
  -> env.step(actions_for_env)
  -> episode/replay buffer 保存 normalized action，而不是 env_action
```

核心结论：

- SAC 连续动作策略内部使用 `TorchSquashedGaussian`。
- 策略采样动作会先经过 `tanh`，得到 `[-1, 1]` 的归一化动作。
- 如果 `config.normalize_actions=True`，RLlib 会在送入环境前把 `[-1, 1]` 映射到真实 `env.action_space` 范围。
- `env.step()` 收到的是映射后的 `actions_for_env`。
- episode/replay buffer 中保存的是策略分布产生的归一化动作 `Columns.ACTIONS`。

## 1. SAC 训练入口复用 DQN training_step

RLlib 的 SAC 继承自 DQN：

```python
class SAC(DQN):
```

源码位置：

```text
ray/rllib/algorithms/sac/sac.py:559
```

new API stack 下，SAC 训练采样主流程复用 DQN：

```python
def training_step(self) -> None:
    if not self.config.enable_env_runner_and_connector_v2:
        return self._training_step_old_api_stack()

    return self._training_step_new_api_stack()
```

源码位置：

```text
ray/rllib/algorithms/dqn/dqn.py:624
```

在 `_training_step_new_api_stack()` 中采样：

```python
episodes, env_runner_results = synchronous_parallel_sample(
    worker_set=self.env_runner_group,
    concat=True,
    sample_timeout_s=self.config.sample_timeout_s,
    _uses_new_env_runners=True,
    _return_metrics=True,
)
```

源码位置：

```text
ray/rllib/algorithms/dqn/dqn.py:646
ray/rllib/algorithms/dqn/dqn.py:654
```

这说明 SAC 训练时与环境交互的动作来自 EnvRunner 采样流程。

## 2. 训练采样默认使用 explore=True

EnvRunner 采样时，如果没有显式传入 `explore`，会使用配置中的 `config.explore`：

```python
if explore is None:
    explore = self.config.explore
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:204
```

默认配置中：

```python
self.explore = True
```

源码位置：

```text
ray/rllib/algorithms/algorithm_config.py:436
```

SAC 配置中的 exploration 类型是随机采样：

```python
self.exploration_config = {
    "type": "StochasticSampling",
}
```

源码位置：

```text
ray/rllib/algorithms/sac/sac.py:50
```

因此训练采样阶段通常是随机动作采样，而不是确定性动作。

## 3. env-to-module connector 构造模型输入

`SingleAgentEnvRunner._sample()` 会在需要 reset 时调用：

```python
self._reset_envs(episodes, shared_data, explore)
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:270
```

reset 后，EnvRunner 会运行 env-to-module connector，把环境 observation 变成 RLModule 输入：

```python
self._cached_to_module = self._env_to_module(
    rl_module=self.module,
    episodes=episodes,
    explore=explore,
    shared_data=shared_data,
    metrics=self.metrics,
    metrics_prefix_key=(ENV_TO_MODULE_CONNECTOR,),
)
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:761
```

默认 env-to-module pipeline 在 `AlgorithmConfig.build_env_to_module_connector()` 中构造：

```python
pipeline.append(AddObservationsFromEpisodesToBatch())
pipeline.append(AddTimeDimToBatchAndZeroPad())
pipeline.append(BatchIndividualItems(multi_agent=self.is_multi_agent))
pipeline.append(NumpyToTensor(device=device))
```

源码位置：

```text
ray/rllib/algorithms/algorithm_config.py:1006
ray/rllib/algorithms/algorithm_config.py:1091
ray/rllib/algorithms/algorithm_config.py:1111
```

简化理解：

```text
obs_from_env
  -> AddObservationsFromEpisodesToBatch
  -> BatchIndividualItems
  -> NumpyToTensor
  -> {"obs": obs_tensor_batch}
```

## 4. SAC RLModule 前向输出 ACTION_DIST_INPUTS

在采样循环中，EnvRunner 取出 connector 缓存的模型输入：

```python
to_module = self._cached_to_module
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:297
```

训练采样默认 `explore=True`，因此调用：

```python
to_env = self.module.forward_exploration(
    to_module, t=global_env_steps_lifetime
)
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:303
ray/rllib/env/single_agent_env_runner.py:311
```

SAC Torch RLModule 中，exploration 前向直接复用 inference 前向：

```python
def _forward_exploration(self, batch: Dict, **kwargs) -> Dict[str, Any]:
    return self._forward_inference(batch)
```

源码位置：

```text
ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py:51
```

实际策略网络前向：

```python
pi_encoder_outs = self.pi_encoder(batch)
output[Columns.ACTION_DIST_INPUTS] = self.pi(pi_encoder_outs[ENCODER_OUT])
return output
```

源码位置：

```text
ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py:38
ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py:43
ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py:47
```

这里需要注意：

```text
ACTION_DIST_INPUTS 不是最终 action，而是动作分布参数。
```

连续动作 SAC 下，`ACTION_DIST_INPUTS` 可以理解为：

```text
[mu, log_std]
```

维度通常是：

```text
2 * action_dim
```

## 5. 连续动作 SAC 使用 TorchSquashedGaussian

SAC 的 action distribution 在 `SACCatalog.get_action_dist_cls()` 中决定：

```python
def get_action_dist_cls(self, framework: str) -> Distribution:
    assert framework == "torch"

    if isinstance(self.action_space, gym.spaces.Box):
        return TorchSquashedGaussian
    elif isinstance(self.action_space, gym.spaces.Discrete):
        return TorchCategorical
```

源码位置：

```text
ray/rllib/algorithms/sac/sac_catalog.py:266
ray/rllib/algorithms/sac/sac_catalog.py:273
```

因此当环境动作空间是连续 `gym.spaces.Box` 时，SAC 使用：

```python
TorchSquashedGaussian
```

## 6. module-to-env connector 从分布参数采样 action

RLModule 前向返回 `Columns.ACTION_DIST_INPUTS` 后，EnvRunner 继续调用 module-to-env connector：

```python
to_env = self._module_to_env(
    rl_module=self.module,
    batch=to_env,
    episodes=episodes,
    explore=explore,
    shared_data=shared_data,
    metrics=self.metrics,
    metrics_prefix_key=(MODULE_TO_ENV_CONNECTOR,),
)
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:318
```

默认 module-to-env pipeline 在 `AlgorithmConfig.build_module_to_env_connector()` 中构造：

```python
pipeline.prepend(GetActions())
pipeline.prepend(TensorToNumpy())
pipeline.prepend(UnBatchToIndividualItems())
pipeline.append(
    NormalizeAndClipActions(
        normalize_actions=self.normalize_actions,
        clip_actions=self.clip_actions,
    )
)
pipeline.append(ListifyDataForVectorEnv())
```

源码位置：

```text
ray/rllib/algorithms/algorithm_config.py:1208
ray/rllib/algorithms/algorithm_config.py:1211
ray/rllib/algorithms/algorithm_config.py:1214
ray/rllib/algorithms/algorithm_config.py:1219
```

默认 pipeline 注释为：

```text
[
    GetActions,
    TensorToNumpy,
    UnBatchToIndividualItems,
    ModuleToAgentUnmapping,
    RemoveSingleTsTimeRankFromBatch,
    NormalizeAndClipActions,
    ListifyDataForVectorEnv,
]
```

源码位置：

```text
ray/rllib/connectors/module_to_env/get_actions.py:21
ray/rllib/connectors/module_to_env/normalize_and_clip_actions.py:28
```

## 7. GetActions 负责真正采样策略动作

`GetActions` 的核心逻辑：

```python
if Columns.ACTION_DIST_INPUTS in batch:
    if explore:
        action_dist_class = sa_rl_module.get_exploration_action_dist_cls()
    else:
        action_dist_class = sa_rl_module.get_inference_action_dist_cls()

    action_dist = action_dist_class.from_logits(
        batch[Columns.ACTION_DIST_INPUTS],
    )

    if not explore:
        action_dist = action_dist.to_deterministic()

    actions = action_dist.sample()
    batch[Columns.ACTIONS] = actions
```

源码位置：

```text
ray/rllib/connectors/module_to_env/get_actions.py:66
ray/rllib/connectors/module_to_env/get_actions.py:73
ray/rllib/connectors/module_to_env/get_actions.py:74
ray/rllib/connectors/module_to_env/get_actions.py:78
ray/rllib/connectors/module_to_env/get_actions.py:81
ray/rllib/connectors/module_to_env/get_actions.py:85
```

训练采样时 `explore=True`，所以：

```python
action_dist = TorchSquashedGaussian.from_logits(action_dist_inputs)
actions = action_dist.sample()
```

不会走：

```python
action_dist.to_deterministic()
```

## 8. TorchSquashedGaussian.from_logits 拆出 mu 和 log_std

`TorchSquashedGaussian.from_logits()`：

```python
@classmethod
def from_logits(cls, logits: TensorType, low: float = -1.0, high: float = 1.0, **kwargs):
    loc, log_std = logits.chunk(2, dim=-1)
    log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
    scale = log_std.exp()
    return cls(loc=loc, scale=scale, low=low, high=high, **kwargs)
```

源码位置：

```text
ray/rllib/core/distribution/torch/torch_distribution.py:323
ray/rllib/core/distribution/torch/torch_distribution.py:328
ray/rllib/core/distribution/torch/torch_distribution.py:331
ray/rllib/core/distribution/torch/torch_distribution.py:332
ray/rllib/core/distribution/torch/torch_distribution.py:337
```

等价理解：

```python
mu, log_std = policy_output.chunk(2, dim=-1)
std = exp(clamp(log_std))
dist = Normal(mu, std)
```

默认参数：

```python
low = -1.0
high = 1.0
```

## 9. TorchSquashedGaussian.sample 内部做 tanh

`TorchSquashedGaussian.sample()`：

```python
def sample(self, *, sample_shape=None):
    sample = super().sample(
        sample_shape=sample_shape if sample_shape is not None else torch.Size()
    )
    return self._squash(sample)
```

源码位置：

```text
ray/rllib/core/distribution/torch/torch_distribution.py:262
ray/rllib/core/distribution/torch/torch_distribution.py:267
ray/rllib/core/distribution/torch/torch_distribution.py:271
```

`super().sample()` 调用的是底层 `torch.distributions.Normal.sample()`：

```python
sample = self._dist.sample(...)
```

源码位置：

```text
ray/rllib/core/distribution/torch/torch_distribution.py:49
```

然后 `_squash()`：

```python
def _squash(self, sample: TensorType) -> TensorType:
    sample = ((torch.tanh(sample) + 1.0) / 2.0) * (self.high - self.low) + self.low
    return torch.clamp(sample, self.low, self.high)
```

源码位置：

```text
ray/rllib/core/distribution/torch/torch_distribution.py:304
ray/rllib/core/distribution/torch/torch_distribution.py:306
```

由于默认 `low=-1.0, high=1.0`，公式：

```python
((tanh(z) + 1.0) / 2.0) * (1.0 - (-1.0)) + (-1.0)
```

可以化简成：

```python
normalized_action = tanh(z)
```

因此 `GetActions` 生成的 `Columns.ACTIONS` 是归一化动作：

```text
[-1, 1]
```

## 10. NormalizeAndClipActions 生成 ACTIONS_FOR_ENV

采样出的动作还不能保证直接匹配环境动作范围。默认 pipeline 后续会进入：

```python
NormalizeAndClipActions
```

源码位置：

```text
ray/rllib/connectors/module_to_env/normalize_and_clip_actions.py:20
```

这个 connector 会复制一份动作，专门给环境使用：

```python
batch[Columns.ACTIONS_FOR_ENV] = copy.deepcopy(batch[Columns.ACTIONS])
self.foreach_batch_item_change_in_place(
    batch=batch,
    column=Columns.ACTIONS_FOR_ENV,
    func=_unsquash_or_clip,
)
```

源码位置：

```text
ray/rllib/connectors/module_to_env/normalize_and_clip_actions.py:137
ray/rllib/connectors/module_to_env/normalize_and_clip_actions.py:139
ray/rllib/connectors/module_to_env/normalize_and_clip_actions.py:140
```

实际转换逻辑：

```python
if self.normalize_actions:
    return unsquash_action(action_for_env, struct)
else:
    return clip_action(action_for_env, struct)
```

源码位置：

```text
ray/rllib/connectors/module_to_env/normalize_and_clip_actions.py:130
```

默认配置中：

```python
self.normalize_actions = True
self.clip_actions = False
```

源码位置：

```text
ray/rllib/algorithms/algorithm_config.py:314
ray/rllib/algorithms/algorithm_config.py:315
```

SAC 又显式设置：

```python
self.clip_actions = False
```

源码位置：

```text
ray/rllib/algorithms/sac/sac.py:84
```

因此默认走的是：

```python
unsquash_action(...)
```

## 11. unsquash_action 的真实映射公式

`unsquash_action()` 在这里：

```text
ray/rllib/utils/spaces/space_utils.py:446
```

关键代码：

```python
def unsquash_action(action, action_space_struct):
    def map_(a, s):
        if (
            isinstance(s, gym.spaces.Box)
            and np.all(s.bounded_below)
            and np.all(s.bounded_above)
        ):
            if s.dtype == np.float32 or s.dtype == np.float64:
                a = s.low + (a + 1.0) * (s.high - s.low) / 2.0
                a = np.clip(a, s.low, s.high)
        return a

    return tree.map_structure(map_, action, action_space_struct)
```

源码位置：

```text
ray/rllib/utils/spaces/space_utils.py:467
ray/rllib/utils/spaces/space_utils.py:473
ray/rllib/utils/spaces/space_utils.py:476
ray/rllib/utils/spaces/space_utils.py:479
```

也就是：

```python
env_action = low + (normalized_action + 1.0) * (high - low) / 2.0
env_action = clip(env_action, low, high)
```

如果环境动作空间是：

```python
gym.spaces.Box(low=0.0, high=100.0, shape=(1,), dtype=np.float32)
```

那么：

```python
env_action = 0.0 + (normalized_action + 1.0) * (100.0 - 0.0) / 2.0
```

即：

```python
env_action = (normalized_action + 1.0) * 50.0
```

对应关系：

```text
normalized_action = -1.0 -> env_action = 0.0
normalized_action =  0.0 -> env_action = 50.0
normalized_action =  1.0 -> env_action = 100.0
```

## 12. EnvRunner 传给 env.step 的是 ACTIONS_FOR_ENV

module-to-env connector 处理完成后，EnvRunner 取出两个动作字段：

```python
actions = to_env.pop(Columns.ACTIONS)
actions_for_env = to_env.pop(Columns.ACTIONS_FOR_ENV, actions)
results = self._try_env_step(actions_for_env)
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:334
ray/rllib/env/single_agent_env_runner.py:335
ray/rllib/env/single_agent_env_runner.py:337
```

源码注释强调：

```text
actions are fully ready ... to be sent to the environment
and might not be identical to the actions produced by the RLModule/distribution,
which are the ones stored permanently in the episode objects.
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:329
```

含义：

```text
Columns.ACTIONS          = RLModule/distribution 产生的归一化动作，通常 [-1, 1]
Columns.ACTIONS_FOR_ENV  = 映射/裁剪后的环境动作，匹配 env.action_space
```

`env.step()` 使用的是：

```python
actions_for_env
```

## 13. episode/replay buffer 保存的是 normalized action

环境 step 后，EnvRunner 保存 episode transition：

```python
observations, rewards, terminateds, truncateds, infos = results
observations, actions = unbatch(observations), unbatch(actions)
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:346
ray/rllib/env/single_agent_env_runner.py:347
```

然后：

```python
episodes[env_index].add_env_step(
    observation=observations[env_index],
    action=actions[env_index],
    reward=rewards[env_index],
    infos=infos[env_index],
    terminated=terminateds[env_index],
    truncated=truncateds[env_index],
    extra_model_outputs=extra_model_output,
)
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:368
```

这里传入的是：

```python
action=actions[env_index]
```

而 `actions` 是前面取出的 `Columns.ACTIONS`，不是 `Columns.ACTIONS_FOR_ENV`。

因此 episode/replay buffer 中保存的是：

```text
normalized action [-1, 1]
```

而不是环境实际收到的 action。

这和 `AlgorithmConfig` 对 `normalize_actions` 的注释一致：

```python
normalize_actions: If True, RLlib learns entirely inside a normalized
action space (0.0 centered with small stddev; only affecting Box
components). RLlib unsquashes actions (and clip, just in case) to the
bounds of the env's action space before sending actions back to the env.
```

源码位置：

```text
ray/rllib/algorithms/algorithm_config.py:1784
```

## 14. [0, 100] 动作空间示例

假设环境动作空间：

```python
action_space = gym.spaces.Box(
    low=np.array([0.0], dtype=np.float32),
    high=np.array([100.0], dtype=np.float32),
    dtype=np.float32,
)
```

策略网络输出：

```python
mu = 0.3
std = 0.5
```

训练采样时：

```python
z = Normal(mu, std).sample()
```

假设采样得到：

```python
z = 0.7
```

`TorchSquashedGaussian.sample()` 内部：

```python
normalized_action = tanh(0.7) = 0.604
```

`GetActions` 写入：

```python
Columns.ACTIONS = 0.604
```

`NormalizeAndClipActions` 复制并映射：

```python
Columns.ACTIONS_FOR_ENV = 0.0 + (0.604 + 1.0) * (100.0 - 0.0) / 2.0
                        = 80.2
```

EnvRunner 调用：

```python
env.step(80.2)
```

但 episode/replay buffer 保存：

```python
action = 0.604
```

后续 SAC learner 训练 Q 网络时，Q 网络输入的 action 也是这个归一化动作。

## 15. 和推理/评估动作的区别

训练采样默认：

```text
explore=True
```

因此 `GetActions` 直接：

```python
actions = action_dist.sample()
```

这会从 `Normal(mu, std)` 随机采样，再经过 `tanh`。

如果评估或手动推理设置：

```python
explore=False
```

`GetActions` 会先：

```python
action_dist = action_dist.to_deterministic()
```

再：

```python
actions = action_dist.sample()
```

源码位置：

```text
ray/rllib/connectors/module_to_env/get_actions.py:81
ray/rllib/connectors/module_to_env/get_actions.py:85
```

但不管随机采样还是确定性推理，只要默认 module-to-env connector 存在，并且 `normalize_actions=True`，最终送给环境前都会走：

```text
NormalizeAndClipActions -> unsquash_action -> env.action_space 范围
```

## 结论

RLlib SAC 训练时，模型推理到 `env.step()` 可用 action 的关键点是：

```text
1. RLModule 输出 ACTION_DIST_INPUTS，也就是 [mu, log_std]
2. GetActions 根据 ACTION_DIST_INPUTS 构造 TorchSquashedGaussian
3. TorchSquashedGaussian.sample() 内部执行 tanh，得到 [-1, 1] 归一化动作
4. NormalizeAndClipActions 复制 Columns.ACTIONS 到 Columns.ACTIONS_FOR_ENV
5. unsquash_action 把 [-1, 1] 映射到 env.action_space.low/high
6. SingleAgentEnvRunner 调用 env.step(actions_for_env)
7. episode/replay buffer 保存的是 Columns.ACTIONS，即归一化动作
```

因此，如果环境动作空间是 `[0, 100]`，RLlib SAC 内部学习的是 `[-1, 1]` 动作，环境实际收到的是：

```python
env_action = (normalized_action + 1.0) * 50.0
```

## 16. forward_exploration 输出如何关联到 get_action_dist_cls

`forward_exploration()` 的输出不会直接调用 `SACCatalog.get_action_dist_cls()`。这两者之间的关系分为两个阶段：

```text
模块初始化阶段：
SACCatalog.get_action_dist_cls()
  -> 返回 TorchSquashedGaussian / TorchCategorical
  -> 缓存到 RLModule.action_dist_cls
  -> 同时用于决定 pi_head 的输出维度

训练采样阶段：
forward_exploration()
  -> 输出 ACTION_DIST_INPUTS
  -> GetActions connector 调 module.get_exploration_action_dist_cls()
  -> 返回初始化时缓存的 action_dist_cls
  -> action_dist_cls.from_logits(ACTION_DIST_INPUTS)
  -> action_dist.sample()
```

也就是说：

```text
forward_exploration() 只负责输出分布参数；
get_action_dist_cls() 负责在模块构建时决定这些参数应该被哪个 Distribution 解释。
```

### 16.1 RLModule 初始化时缓存 action_dist_cls

`RLModule.__init__()` 中有这段逻辑：

```python
self.action_dist_cls = None
if self.catalog is not None:
    self.action_dist_cls = self.catalog.get_action_dist_cls(
        framework=self.framework
    )
```

源码位置：

```text
ray/rllib/core/rl_module/rl_module.py:454
ray/rllib/core/rl_module/rl_module.py:456
```

对于 SAC，`self.catalog` 是 `SACCatalog`，所以这里会调用：

```python
SACCatalog.get_action_dist_cls(framework="torch")
```

SACCatalog 的实现：

```python
def get_action_dist_cls(self, framework: str) -> Distribution:
    assert framework == "torch"

    if isinstance(self.action_space, gym.spaces.Box):
        return TorchSquashedGaussian
    elif isinstance(self.action_space, gym.spaces.Discrete):
        return TorchCategorical
```

源码位置：

```text
ray/rllib/algorithms/sac/sac_catalog.py:266
ray/rllib/algorithms/sac/sac_catalog.py:273
```

如果动作空间是连续 `Box`，则缓存结果是：

```python
self.action_dist_cls = TorchSquashedGaussian
```

### 16.2 构建 pi_head 时也使用 get_action_dist_cls

`SACCatalog.build_pi_head()` 中也会先获取 action distribution class：

```python
action_distribution_cls = self.get_action_dist_cls(framework=framework)
```

源码位置：

```text
ray/rllib/algorithms/sac/sac_catalog.py:187
```

然后根据这个 distribution class 计算 policy head 需要输出多少维：

```python
required_output_dim = action_distribution_cls.required_input_dim(
    space=self.action_space,
    model_config=self._model_config_dict,
)
```

源码位置：

```text
ray/rllib/algorithms/sac/sac_catalog.py:220
```

`TorchSquashedGaussian.required_input_dim()` 返回：

```python
return int(np.prod(space.shape, dtype=np.int32) * 2)
```

因此连续动作下：

```text
action_dim = 1 -> pi_head 输出 2 维：[mu, log_std]
action_dim = 3 -> pi_head 输出 6 维：[mu1, mu2, mu3, log_std1, log_std2, log_std3]
```

这说明 `get_action_dist_cls()` 在模型构建阶段已经决定了两个事情：

```text
1. 策略输出应该被哪个 Distribution 解释
2. 策略 head 最后一层应该输出多少维
```

### 16.3 forward_exploration 只输出 ACTION_DIST_INPUTS

训练采样时，EnvRunner 调用：

```python
to_env = self.module.forward_exploration(
    to_module, t=global_env_steps_lifetime
)
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:311
```

SAC Torch RLModule 中：

```python
def _forward_exploration(self, batch: Dict, **kwargs) -> Dict[str, Any]:
    return self._forward_inference(batch)
```

源码位置：

```text
ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py:51
```

`_forward_inference()` 的核心逻辑：

```python
pi_encoder_outs = self.pi_encoder(batch)
output[Columns.ACTION_DIST_INPUTS] = self.pi(pi_encoder_outs[ENCODER_OUT])
return output
```

源码位置：

```text
ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py:43
ray/rllib/algorithms/sac/torch/default_sac_torch_rl_module.py:47
```

因此 `forward_exploration()` 的输出大致是：

```python
{
    Columns.ACTION_DIST_INPUTS: action_dist_inputs,
}
```

连续 SAC 下可以理解为：

```python
action_dist_inputs = torch.cat([mu, log_std], dim=-1)
```

这一步只产出分布参数，不产出最终 action，也不会直接构造 `TorchSquashedGaussian`。

### 16.4 GetActions connector 把 ACTION_DIST_INPUTS 变成 Distribution

`forward_exploration()` 的输出会作为 `batch` 传给 module-to-env connector：

```python
to_env = self._module_to_env(
    rl_module=self.module,
    batch=to_env,
    episodes=episodes,
    explore=explore,
    ...
)
```

源码位置：

```text
ray/rllib/env/single_agent_env_runner.py:318
```

`GetActions` connector 看到 batch 中有 `Columns.ACTION_DIST_INPUTS` 后，才开始构造分布并采样：

```python
if Columns.ACTION_DIST_INPUTS in batch:
    if explore:
        action_dist_class = sa_rl_module.get_exploration_action_dist_cls()
    else:
        action_dist_class = sa_rl_module.get_inference_action_dist_cls()

    action_dist = action_dist_class.from_logits(
        batch[Columns.ACTION_DIST_INPUTS],
    )

    if not explore:
        action_dist = action_dist.to_deterministic()

    actions = action_dist.sample()
    batch[Columns.ACTIONS] = actions
```

源码位置：

```text
ray/rllib/connectors/module_to_env/get_actions.py:73
ray/rllib/connectors/module_to_env/get_actions.py:74
ray/rllib/connectors/module_to_env/get_actions.py:78
ray/rllib/connectors/module_to_env/get_actions.py:81
ray/rllib/connectors/module_to_env/get_actions.py:85
```

训练采样时 `explore=True`，所以它调用：

```python
action_dist_class = sa_rl_module.get_exploration_action_dist_cls()
```

### 16.5 get_exploration_action_dist_cls 返回缓存的 action_dist_cls

`TorchRLModule` 中：

```python
def get_exploration_action_dist_cls(self) -> Type[TorchDistribution]:
    return self.get_inference_action_dist_cls()
```

源码位置：

```text
ray/rllib/core/rl_module/torch/torch_rl_module.py:159
```

`get_inference_action_dist_cls()` 中优先返回初始化时缓存的 `self.action_dist_cls`：

```python
def get_inference_action_dist_cls(self) -> Type[TorchDistribution]:
    if self.action_dist_cls is not None:
        return self.action_dist_cls
    elif isinstance(self.action_space, gym.spaces.Discrete):
        return TorchCategorical
    elif isinstance(self.action_space, gym.spaces.Box):
        return TorchDiagGaussian
```

源码位置：

```text
ray/rllib/core/rl_module/torch/torch_rl_module.py:137
ray/rllib/core/rl_module/torch/torch_rl_module.py:138
```

对于 SAC，`self.action_dist_cls` 已经在初始化时被 `SACCatalog.get_action_dist_cls()` 设置为：

```python
TorchSquashedGaussian
```

因此 `GetActions` 中拿到的是：

```python
action_dist_class = TorchSquashedGaussian
```

### 16.6 from_logits 使用 forward_exploration 的输出构造分布

`GetActions` 接着执行：

```python
action_dist = action_dist_class.from_logits(
    batch[Columns.ACTION_DIST_INPUTS],
)
```

等价于：

```python
action_dist = TorchSquashedGaussian.from_logits(action_dist_inputs)
```

`TorchSquashedGaussian.from_logits()` 中：

```python
loc, log_std = logits.chunk(2, dim=-1)
log_std = torch.clamp(log_std, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT)
scale = log_std.exp()
return cls(loc=loc, scale=scale, low=low, high=high, **kwargs)
```

因此：

```text
forward_exploration 输出的 ACTION_DIST_INPUTS
  -> 被 TorchSquashedGaussian.from_logits 拆成 mu/log_std
  -> 构造 Normal(mu, std)
  -> sample() 时再 tanh squash
```

### 16.7 精确调用关系

把这段关系串起来：

```text
模块初始化：
RLModule.__init__()
  -> self.catalog.get_action_dist_cls(framework="torch")
  -> SACCatalog.get_action_dist_cls()
  -> Box action_space 返回 TorchSquashedGaussian
  -> self.action_dist_cls = TorchSquashedGaussian

模型构建：
SACCatalog.build_pi_head()
  -> self.get_action_dist_cls(framework="torch")
  -> TorchSquashedGaussian.required_input_dim(action_space)
  -> pi_head 输出维度 = action_dim * 2

训练采样：
SingleAgentEnvRunner._sample()
  -> self.module.forward_exploration(to_module, ...)
  -> {ACTION_DIST_INPUTS: pi_head(obs)}
  -> self._module_to_env(..., batch=to_env, explore=True)
  -> GetActions._get_actions()
  -> module.get_exploration_action_dist_cls()
  -> module.get_inference_action_dist_cls()
  -> return self.action_dist_cls
  -> TorchSquashedGaussian.from_logits(ACTION_DIST_INPUTS)
  -> TorchSquashedGaussian.sample()
  -> Columns.ACTIONS
```

核心点：

```text
forward_exploration() 不负责选择分布类；
分布类在 RLModule 初始化时由 Catalog 决定并缓存；
GetActions connector 在采样时把 forward_exploration 的 ACTION_DIST_INPUTS 和缓存的分布类组合起来，生成最终 action。
```

## 17. 连续动作 SAC 线上部署推理代码

本节采用的部署口径是：线上使用稳定的确定性策略，同时保持 SAC 训练时的
squash 语义。

也就是显式复刻 `TorchSquashedGaussian._squash()`，不要在部署代码里只写化简后的
`torch.tanh(mu)`：

```python
dist_cls = module.get_inference_action_dist_cls()
dist = dist_cls.from_logits(action_dist_inputs)
mu = dist.to_deterministic().sample()

low = torch.as_tensor(dist.low, dtype=mu.dtype, device=mu.device)
high = torch.as_tensor(dist.high, dtype=mu.dtype, device=mu.device)
policy_action = ((torch.tanh(mu) + 1.0) / 2.0) * (high - low) + low
policy_action = torch.clamp(policy_action, low, high)

policy_action_np = policy_action.squeeze(0).detach().cpu().numpy()
env_action = unsquash_action(policy_action_np, action_space)
```

当前 RLlib `GetActions` 调用 `from_logits()` 时不会额外传 `low/high`，所以
`TorchSquashedGaussian` 默认仍然是 `low=-1.0, high=1.0`。这里保留完整公式的目的，
是让部署代码和 `_squash()` 语义一致，避免把这个默认值写死到 `torch.tanh(mu)` 里。

如果以后你自定义了 distribution bounds，不再是 `[-1, 1]`，还要同时确认
`NormalizeAndClipActions` / `unsquash_action()` 是否仍然适用，因为
`unsquash_action()` 默认按 `[-1, 1] -> env.action_space` 这套归一化语义做映射。

这不是 RLlib 默认 `explore=False` connector 当前行为的逐行复刻；它是更符合
SAC 连续动作策略语义的线上确定性部署方式。

### 17.1 为什么线上推荐 squash(mu)

SAC 连续动作训练采样时，策略动作来自 `TorchSquashedGaussian.sample()`：

```text
ACTION_DIST_INPUTS = [mu, log_std]
z ~ Normal(mu, std)
policy_action = tanh(z)
env_action = unsquash_action(policy_action, action_space)
```

因此训练过程中，actor / replay buffer / Q 网络面对的是 squashed 后的策略动作，
也就是通常位于：

```text
[-1, 1]
```

线上如果想用确定性动作，不再从 `Normal(mu, std)` 随机采样，最自然的确定性版本是：

```text
policy_action = squash(mu)
```

这里的 `squash` 指 `TorchSquashedGaussian._squash()` 的同形公式：

```python
policy_action = ((torch.tanh(mu) + 1.0) / 2.0) * (high - low) + low
policy_action = torch.clamp(policy_action, low, high)
```

然后再映射到环境动作空间：

```text
env_action = low + (policy_action + 1.0) * (high - low) / 2.0
```

这和训练采样的差异只有一处：

```text
训练：squash(Normal(mu, std).sample())
线上：squash(mu)
```

也就是说，线上不用随机采样，但保留了训练时的完整 squash 语义。

### 17.2 完整线上推理类

```python
from pathlib import Path

import numpy as np
import torch
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.spaces.space_utils import clip_action, unsquash_action


class SACSquashedMeanPolicy:
    """SAC continuous online policy: action = squash(mu), then map to env space."""

    def __init__(
        self,
        checkpoint_dir: str,
        *,
        normalize_actions: bool = True,
        clip_actions: bool = False,
        device: str = "cpu",
    ):
        ckpt = Path(checkpoint_dir).expanduser().resolve()

        # algorithm.save_to_path() 保存的是完整 Algorithm checkpoint。
        # 单 agent RLModule 通常在这个子目录。
        module_dir = ckpt / "learner_group" / "learner" / "rl_module" / "default_policy"
        if module_dir.exists():
            ckpt = module_dir

        self.module = RLModule.from_checkpoint(ckpt.as_posix())
        self.module.to(device)
        self.module.eval()

        self.device = device
        self.normalize_actions = normalize_actions
        self.clip_actions = clip_actions
        self.action_space = self.module.action_space

    @staticmethod
    def _squash_with_dist_bounds(sample: torch.Tensor, dist: object) -> torch.Tensor:
        # Same formula as TorchSquashedGaussian._squash(), without calling a private API.
        low = torch.as_tensor(
            getattr(dist, "low", -1.0),
            dtype=sample.dtype,
            device=sample.device,
        )
        high = torch.as_tensor(
            getattr(dist, "high", 1.0),
            dtype=sample.dtype,
            device=sample.device,
        )
        squashed = ((torch.tanh(sample) + 1.0) / 2.0) * (high - low) + low
        return torch.clamp(squashed, low, high)

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.inference_mode():
            out = self.module.forward_inference({Columns.OBS: obs_tensor})
            action_dist_inputs = out[Columns.ACTION_DIST_INPUTS]

            # SAC continuous ACTION_DIST_INPUTS = [mu, log_std]。
            # 线上确定性部署采用训练采样分布的 squashed mean。
            # 不直接化简成 torch.tanh(mu)，而是保留 TorchSquashedGaussian._squash() 同形公式。
            dist_cls = self.module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(action_dist_inputs)
            mu = dist.to_deterministic().sample()
            policy_action_tensor = self._squash_with_dist_bounds(mu, dist)

        # policy_action 是 RLlib 内部 squashed action。
        # squeeze(0) 去掉 batch 维；detach().cpu().numpy() 兼容 GPU/MPS 推理。
        policy_action = policy_action_tensor.squeeze(0).detach().cpu().numpy()

        # 对齐训练 config 的 action 后处理。
        # 默认 normalize_actions=True 时，送环境前要 unsquash 到 env.action_space。
        if self.normalize_actions:
            env_action = unsquash_action(policy_action, self.action_space)
        elif self.clip_actions:
            env_action = clip_action(policy_action, self.action_space)
        else:
            env_action = policy_action

        return np.asarray(env_action, dtype=self.action_space.dtype)
```

这里有两个容易写错的点：

```python
policy_action = policy_action_tensor.squeeze(0).detach().cpu().numpy()
```

- `squeeze(0)`：去掉前面 `obs_tensor.unsqueeze(0)` 加出来的 batch 维。
- `detach()`：断开 autograd 关系；虽然 `inference_mode()` 下通常已经无梯度，保留更安全。
- `cpu()`：如果推理在 GPU/MPS 上，必须先搬回 CPU 才能 `.numpy()`。
- `numpy()`：`unsquash_action()` / `clip_action()` 使用 NumPy / gym space 语义处理动作。

使用示例：

```python
import gymnasium as gym

env = gym.make("Pendulum-v1")
policy = SACSquashedMeanPolicy(
    "runs/xxx/checkpoints/best",
    normalize_actions=True,  # 必须对齐训练 config.normalize_actions，RLlib 默认 True。
    clip_actions=False,      # 必须对齐训练 config.clip_actions，SAC 默认 False。
)

obs, info = env.reset()
terminated = truncated = False

while not (terminated or truncated):
    action = policy.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
```

### 17.3 和 RLlib 默认 explore=False evaluation 的区别

RLlib 默认 `GetActions(explore=False)` 当前行为不是 `tanh(mu)`。

精确链路是：

```text
module.forward_inference(...)
  -> ACTION_DIST_INPUTS = [mu, log_std]
  -> TorchSquashedGaussian.from_logits(...)
  -> dist.to_deterministic()
  -> TorchDeterministic(loc=mu)
  -> sample() 返回 mu
  -> NormalizeAndClipActions
  -> unsquash_action(mu, action_space)
  -> env.step(env_action)
```

关键点：

```text
RLlib 默认 explore=False evaluation: mu -> unsquash_action()
线上推荐 SAC deterministic:       squash(mu) -> unsquash_action()
```

所以，如果线上按本节推荐的 `squash(mu)` 部署，那么 RLlib 默认
`evaluation_config.explore=False` 的指标只能作为参考，不是线上行为的精确评估。

### 17.4 训练、默认评估、线上部署三者对比

| 场景 | 策略动作 | 是否随机 | 是否有 tanh | 用途 |
| --- | --- | --- | --- | --- |
| 训练采样 `explore=True` | `tanh(Normal(mu, std).sample())` | 是 | 是 | 探索、收集 replay buffer |
| RLlib 默认评估 `explore=False` | `mu` | 否 | 否 | RLlib connector 当前默认确定性行为 |
| 线上推荐确定性部署 | `squash(mu)` | 否 | 是 | 稳定部署，保留 SAC squash 语义 |
| 随机策略评估 `explore=True` | `tanh(Normal(mu, std).sample())` | 是 | 是 | 评估训练采样策略本身 |

因此，想让评估指标代表线上效果时，不要只依赖 RLlib 默认 `explore=False`。
应该让 evaluation 也走和线上完全相同的 `squash(mu) + unsquash_action()` 逻辑。

### 17.5 如何让 evaluation 和线上部署一致

如果线上使用：

```python
dist_cls = module.get_inference_action_dist_cls()
dist = dist_cls.from_logits(action_dist_inputs)
mu = dist.to_deterministic().sample()
low = torch.as_tensor(dist.low, dtype=mu.dtype, device=mu.device)
high = torch.as_tensor(dist.high, dtype=mu.dtype, device=mu.device)
policy_action = ((torch.tanh(mu) + 1.0) / 2.0) * (high - low) + low
policy_action = torch.clamp(policy_action, low, high)
policy_action_np = policy_action.squeeze(0).detach().cpu().numpy()
env_action = unsquash_action(policy_action_np, action_space)
```

那么训练期间选择 best checkpoint 的 evaluation 也应该使用同一套动作逻辑。
否则会出现：

```text
best checkpoint 是按 mu -> unsquash_action() 选出来的，
线上却按 squash(mu) -> unsquash_action() 执行。
```

这时 evaluation 指标和线上效果不是同一个策略口径。

可选方案：

1. 写自定义 evaluation function，内部使用 `SACSquashedMeanPolicy.act()` 跑评估 episode。
2. 或单独写离线评估脚本，用 `SACSquashedMeanPolicy` 对每个 checkpoint 评估并选 best。
3. 或如果决定线上也严格复刻 RLlib 默认 evaluation，就不要手动 squash，而是使用 `dist.to_deterministic().sample()`。

推荐口径是：

```text
训练：RLlib SAC 正常训练
评估 best：squash(mu) + unsquash_action()
线上部署：squash(mu) + unsquash_action()
```

### 17.6 自定义 evaluation 的动作核心

自定义 evaluation 不一定要完整复刻 RLlib EnvRunner，只要确保 observation 输入处理一致，
action 部分可以复用下面的核心逻辑：

```python
with torch.inference_mode():
    out = module.forward_inference({Columns.OBS: obs_tensor})
    action_dist_inputs = out[Columns.ACTION_DIST_INPUTS]
    dist_cls = module.get_inference_action_dist_cls()
    dist = dist_cls.from_logits(action_dist_inputs)
    mu = dist.to_deterministic().sample()
    low = torch.as_tensor(dist.low, dtype=mu.dtype, device=mu.device)
    high = torch.as_tensor(dist.high, dtype=mu.dtype, device=mu.device)
    policy_action_tensor = ((torch.tanh(mu) + 1.0) / 2.0) * (high - low) + low
    policy_action_tensor = torch.clamp(policy_action_tensor, low, high)

policy_action = policy_action_tensor.squeeze(0).detach().cpu().numpy()
env_action = unsquash_action(policy_action, module.action_space)
```

如果训练配置不是默认值，则仍然要对齐：

```text
normalize_actions=True  -> unsquash_action(policy_action, action_space)
normalize_actions=False and clip_actions=True -> clip_action(policy_action, action_space)
normalize_actions=False and clip_actions=False -> policy_action
```

### 17.7 注意 observation 预处理也要一致

上面的代码只覆盖 action 生成与 action 后处理。

如果使用了这些能力：

```text
自定义 env-to-module connector
observation normalization
frame stack
action mask
多智能体 module mapping
自定义 observation wrapper
```

那么线上部署和自定义 evaluation 也必须复刻同样的 observation 预处理。
否则即使 action 逻辑都是 `squash(mu) + unsquash_action()`，模型输入也不同，评估和线上仍然不会一致。

### 17.8 如果要严格复刻 RLlib 默认 explore=False

如果目标不是采用 `squash(mu)`，而是严格复刻 RLlib 默认 `explore=False` connector，
则动作核心应写成：

```python
dist_cls = module.get_inference_action_dist_cls()
dist = dist_cls.from_logits(action_dist_inputs)
policy_action_tensor = dist.to_deterministic().sample()
```

然后继续按配置做：

```python
if normalize_actions:
    env_action = unsquash_action(policy_action, action_space)
elif clip_actions:
    env_action = clip_action(policy_action, action_space)
else:
    env_action = policy_action
```

这条链路和 RLlib 默认 evaluation 一致，但不包含手动 squash。

### 17.9 关键纠正：to_deterministic 后不会再走 _squash

容易把下面两条链路混在一起：

```text
action_dist = TorchSquashedGaussian.from_logits(...)
action_dist = action_dist.to_deterministic()
action = action_dist.sample()
```

和：

```text
action = TorchSquashedGaussian.sample()
```

它们不是一回事。

训练 / explore=True 时：

```text
action_dist = TorchSquashedGaussian.from_logits(ACTION_DIST_INPUTS)
action = action_dist.sample()
  -> Normal(mu, std).sample()
  -> TorchSquashedGaussian._squash(sample)
  -> tanh(sample)
env_action = unsquash_action(action, action_space)
```

RLlib 默认 explore=False 时：

```text
action_dist = TorchSquashedGaussian.from_logits(ACTION_DIST_INPUTS)
action_dist = action_dist.to_deterministic()
  -> TorchDeterministic(loc=mu)
action = action_dist.sample()
  -> TorchDeterministic.sample()
  -> mu
env_action = unsquash_action(action, action_space)
```

所以，严格来说，RLlib 默认 explore=False 链路里没有：

```text
TorchSquashedGaussian._squash(mu)
```

如果线上希望使用 SAC squashed deterministic 语义，应该显式做：

```python
dist_cls = module.get_inference_action_dist_cls()
dist = dist_cls.from_logits(action_dist_inputs)
mu = dist.to_deterministic().sample()
low = torch.as_tensor(dist.low, dtype=mu.dtype, device=mu.device)
high = torch.as_tensor(dist.high, dtype=mu.dtype, device=mu.device)
policy_action = ((torch.tanh(mu) + 1.0) / 2.0) * (high - low) + low
policy_action = torch.clamp(policy_action, low, high)
policy_action_np = policy_action.squeeze(0).detach().cpu().numpy()
env_action = unsquash_action(policy_action_np, action_space)
```

也可以理解成“手动对 `mu` 做 `_squash`”。不要直接调用私有方法
`_squash()`，因为它不是稳定 API；在部署代码里保留同形公式更稳妥，
尤其是以后 distribution bounds 不再是 `[-1, 1]` 时。
