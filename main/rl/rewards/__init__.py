from __future__ import annotations

from rjt_rl.utils.config_wrapper import load_class
from .penalized_logp_reward import similarityReward,TADFscoreReward,PropReward,TPSAReward,SAReward,PropReward2  # NOQA
from .similarity_reward import SimilarityReward  # NOQA
from .MPNN_reward import MPNNReward,MPNNReward0,MPNNReward_ale_epi,MPNNReward_ale_epi_2,MPNNReward_T2,MPNNReward_small_est

def get_reward_class(clsnm: str) -> type:
    return load_class(clsnm, default_module_name="rjt_rl.rl.rewards")
