from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pfrl.nn.mlp import MLP
from torch.distributions import Categorical

from rjt_rl.nn.site_tree.embed_site_info import EmbedSiteInfoBidir1
from rjt_rl.nn.site_tree.site_tree_encoder import SiteTreeEncoder
from rjt_rl.rjt.mol_tree import MolTree
from rjt_rl.rjt.utils import (
    Module,
    check_node_idx,
    filter_logits,
    index_tensor,
    mask_invalid_nodes,
    set_batch_node_id,
)
from rjt_rl.rjt.vocab import Vocab
from rjt_rl.rl.datasets.expert_dataset_collator import ListFromMolTree
from rjt_rl.rl.envs.mol_action_distr import MolActionDistr3

logger = logging.getLogger(__name__)

@dataclass
class RewardNetRNNConfig:
    hidden_size: int = 512
    mlp_hidden_sizes: Optional[List[int]] = None  # 用于最终输出奖励的 MLP
    # 如果需要更多参数，可以继续添加

class RewardNetRNN(nn.Module):

    def __init__(self, vocab, config: RewardNetRNNConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.node_feature_size = config.hidden_size  # 这里简单采用相同尺寸
        self.edge_feature_size = config.hidden_size

        vocab_size = len(vocab)
        self.embedding = nn.Embedding(vocab_size + 1, self.node_feature_size, padding_idx=vocab_size)
        self.embed_slot = EmbedSiteInfoBidir1(out_size=self.edge_feature_size)
        self.tree_encoder = SiteTreeEncoder(vocab, self.hidden_size, self.node_feature_size, self.edge_feature_size, self.embedding, self.embed_slot)
        hidden_sizes = config.mlp_hidden_sizes
        self.reward_mlp = MLP(in_size=self.hidden_size, out_size=1, hidden_sizes=hidden_sizes)

    def forward(self, batch):
        encoder_data = batch.get_encoder_data()  
        _, aggr = self.tree_encoder(encoder_data, aggregate_all=True)
        rewards = self.reward_mlp(aggr)  
        return rewards
