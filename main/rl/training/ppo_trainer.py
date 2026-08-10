from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.optim as optim
from omegaconf import MISSING
from pfrl import experiments

from rjt_rl.rjt.vocab import Vocab, load_vocab
from rjt_rl.rl.agents.rjt_ppo import RJTPPO, RJTPPOConfig
from rjt_rl.rl.datasets import get_dataset_class
from rjt_rl.rl.datasets.expert_dataset_collator import (
    PretrainDatasetCollator,
    RLDatasetCollator,
)
from rjt_rl.rl.envs.mol_env_base import MolEnvBase
# from rjt_rl.rl.models.reverse_RL_reward_net import RewardNetRNN, RewardNetRNNConfig 
from rjt_rl.utils.config_wrapper import obj_from_config

from .snapshot import SnapshotHook, load_snap_file
from .utils import setup_random_seed

logger = logging.getLogger(__name__)


import torch.nn as nn

# def get_recent_states_from_env(env: MolEnvBase, n: int = 128) -> torch.Tensor:
#     """
#     从环境中提取最近 n 个状态，转换为 IRL 模型的输入张量。
#     假设 env.state_history 是一个包含状态对象的列表，
#     并且 env 提供了 get_state_feature 方法将状态转换为数值型特征列表。
#     """
#     recent_states = env.state_history[-n:]
#     # 如果状态数量不足 n，直接使用所有状态
#     if len(recent_states) == 0:
#         raise ValueError("环境状态历史为空，无法提取最近状态。")
#         # 将 features 转换为 tensor
#     return torch.tensor(recent_states, dtype=torch.float32)


# class IRLTrainer:
#     """
#     IRLTrainer 用于更新奖励网络，使得专家数据得到较高奖励，
#     而 agent 生成的数据奖励较低。目标函数形式为：
#         loss = mean(r_expert) - log(mean(exp(r_agent)))
#     然后取负值进行最小化。
#     """
#     def __init__(self, reward_net: nn.Module, optimizer: optim.Optimizer, device: torch.device):
#         """
#         :param reward_net: 奖励网络（例如 RewardNetRNN），输入状态，输出奖励值，形状 (batch_size, 1)
#         :param optimizer: 优化器，用于更新 reward_net 参数
#         :param device: torch.device 对象
#         """
#         self.reward_net = reward_net
#         self.optimizer = optimizer
#         self.device = device

#     def fit(self, expert_states: torch.Tensor, agent_states: torch.Tensor) -> float:
#         """
#         利用专家状态数据和 agent 状态数据更新奖励网络。
        
#         :param expert_states: Tensor，形状 (N_expert, state_dim)，专家样本状态表示
#         :param agent_states: Tensor，形状 (N_agent, state_dim)，agent生成的状态表示
#         :return: 当前批次的 IRL 损失（标量）
#         """
#         # 确保模型处于训练模式
#         self.reward_net.train()
#         self.optimizer.zero_grad()

#         # 将输入移动到指定设备上
#         expert_states = expert_states.to(self.device)
#         agent_states = agent_states.to(self.device)

#         # 前向传播计算奖励
#         # expert_rewards, agent_rewards 均为 (batch_size, 1)
#         expert_rewards = self.reward_net(expert_states)
#         agent_rewards = self.reward_net(agent_states)

#         # 计算目标函数
#         # 我们希望专家样本的奖励较高，agent样本的奖励较低
#         loss = torch.mean(expert_rewards) - torch.log(torch.mean(torch.exp(agent_rewards)) + 1e-8)
#         # 取负值，以最小化 loss
#         loss = -loss

#         loss.backward()
#         self.optimizer.step()

#         return loss.item()


@dataclass
class PPOTrainerConfig:
    # output directory
    outdir: str = "result"

    # GPU ID (negative value indicates CPU)
    gpu: int = 0

    # Random seed [0, 2 ** 32)
    seed: int = 0

    # Number of env instances run in parallel
    num_envs: int = 1

    # torch's debug flag
    autograd_detect_anomaly: bool = False

    # Total time steps for training
    steps: int = 10**6

    # Learning rate
    learning_rate: float = 1e-3
    final_learning_rate: float = 0.0

    snap_freq: Optional[int] = None
    snap_file_name: str = "last_snap.pt"
    resume: Optional[str] = None
    load_model: Optional[str] = None

    # vocab (required)
    vocab: str = MISSING

    pretrained_model: Optional[str] = None

    # Expert minibatch size
    expert_batch_size: int = 128
    dataloader_shuffle: bool = True

    input_pkl: Optional[str] = None
    use_moltree_dataset: bool = False
    use_depth_first: bool = False

    dataset: Any = MISSING

    agent: RJTPPOConfig = MISSING


class PPOTrainer:
    @staticmethod
    def get_config_class() -> type[PPOTrainerConfig]:
        return PPOTrainerConfig

    @classmethod
    def from_config(cls, config: PPOTrainerConfig) -> "PPOTrainer":
        return cls(config)

    def __init__(self, config: PPOTrainerConfig):
        self.outdir = config.outdir
        self.device_id = config.gpu
        self.seed = config.seed
        self.num_envs = config.num_envs
        self.autograd_detect_anomaly = config.autograd_detect_anomaly
        self.steps = config.steps
        self.lr = config.learning_rate
        self.final_lr = config.final_learning_rate
        self.snap_freq = config.snap_freq
        self.snap_file_name = config.snap_file_name
        self.resume = config.resume
        self.vocab = config.vocab

        self.pretrained_model = config.pretrained_model

        self.input_pkl = config.input_pkl
        self.use_moltree_dataset = config.use_moltree_dataset
        self.use_depth_first = config.use_depth_first

        self.config = config

    def create_env(self, idx: int) -> MolEnvBase:
        raise NotImplementedError()

    def create_model(self, vocab: Vocab, env: MolEnvBase) -> torch.nn.Module:
        raise NotImplementedError()

    def _create_env_wrapper(self, idx: int, env_seed: int, vocab: Vocab) -> MolEnvBase:
        logger.info(f"create env: idx={idx}, seed={env_seed}")
        env = self.create_env(idx)
        env.init(vocab)
        env.seed(env_seed)
        return env

    def setup_ppo_agent(self) -> tuple[RJTPPO, Vocab]:
        vocab = load_vocab(self.vocab)

        expert_loadfn = functools.partial(self.expert_dataset_loader, vocab)

        sample_env = self._create_env_wrapper(idx=0, env_seed=0, vocab=vocab)
        model = self.create_model(vocab, sample_env)
        del sample_env

        if self.device_id is None or self.device_id < 0:
            device_s = "cpu"
        else:
            device_s = f"cuda:{self.device_id}"
        device = torch.device(device_s)

        if self.pretrained_model is not None:
            logger.info(f"Load pretrained model from: {self.pretrained_model}")
            state = torch.load(self.pretrained_model, map_location=device)  # type: ignore
            logger.debug(state.keys())
            if "models" in state:
                logger.info(state["models"]["main"].keys())
                new_state = OrderedDict()
                nskip = len("policy.")
                for k, v in state["models"]["main"].items():
                    new_state[k[nskip:]] = v
                model.load_state_dict(new_state)
            else:
                model.load_state_dict(state)

        model.to(device)

        opt = optim.Adam(model.parameters())

        batch_molenv_states = RLDatasetCollator(vocab)

        agent = RJTPPO.from_config(
            model,
            opt,
            self.config.agent,
            gpu=device.index,
            batch_states=batch_molenv_states,
            dataset_loader=expert_loadfn,
        )

        return agent, vocab

    # def setup_rewardnet(self):
    #     # --- 初始化 IRL 部分 ---
    #     # 假设状态表示维度为 128（根据实际情况调整）
    #     vocab = load_vocab(self.vocab)
    #     state_dim = 128  
    #     # 配置 IRL 奖励网络（使用 RewardNetRNN）
    #     irl_config = RewardNetRNNConfig(hidden_size=state_dim, mlp_hidden_sizes=[state_dim, state_dim])
    #     self.irl_reward_net = RewardNetRNN(vocab, irl_config).to(device)
    #     irl_opt = optim.Adam(self.irl_reward_net.parameters(), lr=1e-3)
    #     self.irl_trainer = IRLTrainer(self.irl_reward_net, irl_opt, device)
    #     if self.device_id is None or self.device_id < 0:
    #         device_s = "cpu"
    #     else:
    #         device_s = f"cuda:{self.device_id}"
    #     device = torch.device(device_s)
    #     self.device = device  # 保存设备引用
    #     # ----------------------------------

    def train(self) -> None:
        rank: Optional[int] = None
        if rank is None:
            local_seed = self.seed
        else:
            local_seed = self.seed + rank
        setup_random_seed(local_seed)

        if self.autograd_detect_anomaly:
            torch.autograd.set_detect_anomaly(True)

        agent, vocab = self.setup_ppo_agent()
        expert_loader = self.expert_dataset_loader(vocab)
        expert_iter = iter(expert_loader)

        def lr_setter(env: MolEnvBase, agent: RJTPPO, value: float) -> None:
            agent.optimizer.alpha = value

        def agent_step_hook(env: MolEnvBase, agent: RJTPPO, t: int) -> None:
            agent.current_step = t
            agent.current_episode = env.episode

            # 自定义 IRL 更新 hook：每隔 1000 步更新一次 IRL 模块
        # def irl_update_hook(env: MolEnvBase, agent: RJTPPO, t: int) -> None:
        #     if t % 1000 == 0:
        #         nonlocal expert_iter
        #         try:
        #             expert_batch = next(expert_iter)
        #         except StopIteration:
        #             expert_iter = iter(expert_loader)
        #             expert_batch = next(expert_iter)
        #         # expert_batch 预处理后得到专家状态张量，格式需与你的 IRL 网络输入匹配
        #         expert_states = expert_batch
        #         # 假设 agent 提供 get_recent_states() 接口获取最近采样的状态表示
        #         agent_states = get_recent_states_from_env(env, n=128)  
        #         irl_loss = self.irl_trainer.fit(expert_states, agent_states)
        #         logger.info(f"IRL update at step {t}: loss = {irl_loss}")

        step_hooks = [
            experiments.LinearInterpolationHook(
                self.steps, self.lr, self.final_lr, lr_setter
            ),
            SnapshotHook(
                freq=self.snap_freq, file_name=self.snap_file_name, out_dir=self.outdir
            ),
            agent_step_hook,
            # irl_update_hook,
        ]

        if self.num_envs != 1:
            raise RuntimeError("num_envs != 1 not supported")

        env = self._create_env_wrapper(
            0,
            env_seed=local_seed,
            vocab=vocab,
        )

        step_offset = 0
        if self.resume is not None:
            load_path = Path(self.resume)
            if load_path.exists():
                step_offset = load_snap_file(env, agent, self.resume)
            else:
                logger.info(f"resume file {self.resume} not found --> ignored")
        logger.info(f"starting from step: {step_offset}")

        try:
            experiments.train_agent(
                agent=agent,
                env=env,
                outdir=self.outdir,
                steps=self.steps,
                step_hooks=step_hooks,
                step_offset=step_offset,
            )
        except (Exception, KeyboardInterrupt):
            env.flush_history()
            raise

        env.flush_history()

    def expert_dataset_loader(self, vocab: Vocab) -> torch.utils.data.DataLoader:
        dataset_config = self.config.dataset
        expert_ds, _ = obj_from_config(get_dataset_class, dataset_config)

        shuffle = self.config.dataloader_shuffle
        if isinstance(expert_ds, torch.utils.data.IterableDataset):
            if shuffle:
                logger.warning(
                    "dataloader shuffling is not supported"
                    " by IterableDataset. The option is disabled."
                )
                shuffle = False

        collator = PretrainDatasetCollator()
        expert_iter = torch.utils.data.DataLoader(
            expert_ds,
            batch_size=self.config.expert_batch_size,
            shuffle=shuffle,
            collate_fn=collator,
        )
        return expert_iter
