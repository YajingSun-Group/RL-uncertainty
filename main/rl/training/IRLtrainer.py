import torch
import torch.nn as nn
import torch.optim as optim

class IRLTrainer:
    """
    IRLTrainer 用于更新奖励网络，使得专家数据得到较高奖励，
    而 agent 生成的数据奖励较低。目标函数形式为：
        loss = mean(r_expert) - log(mean(exp(r_agent)))
    然后取负值进行最小化。
    """
    def __init__(self, reward_net: nn.Module, optimizer: optim.Optimizer, device: torch.device):
        """
        :param reward_net: 奖励网络（例如 RewardNetRNN），输入状态，输出奖励值，形状 (batch_size, 1)
        :param optimizer: 优化器，用于更新 reward_net 参数
        :param device: torch.device 对象
        """
        self.reward_net = reward_net
        self.optimizer = optimizer
        self.device = device

    def fit(self, expert_states: torch.Tensor, agent_states: torch.Tensor) -> float:
        """
        利用专家状态数据和 agent 状态数据更新奖励网络。
        
        :param expert_states: Tensor，形状 (N_expert, state_dim)，专家样本状态表示
        :param agent_states: Tensor，形状 (N_agent, state_dim)，agent生成的状态表示
        :return: 当前批次的 IRL 损失（标量）
        """
        # 确保模型处于训练模式
        self.reward_net.train()
        self.optimizer.zero_grad()

        # 将输入移动到指定设备上
        expert_states = expert_states.to(self.device)
        agent_states = agent_states.to(self.device)

        # 前向传播计算奖励
        # expert_rewards, agent_rewards 均为 (batch_size, 1)
        expert_rewards = self.reward_net(expert_states)
        agent_rewards = self.reward_net(agent_states)

        # 计算目标函数
        # 我们希望专家样本的奖励较高，agent样本的奖励较低
        loss = torch.mean(expert_rewards) - torch.log(torch.mean(torch.exp(agent_rewards)) + 1e-8)
        # 取负值，以最小化 loss
        loss = -loss

        loss.backward()
        self.optimizer.step()

        return loss.item()
