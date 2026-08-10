from __future__ import annotations

import logging
from typing import Optional

from rjt_rl.rl.envs.states import State

logger = logging.getLogger(__name__)


class BaseReward(object):
    known_smiles: dict[str, int]

    def __init__(self, score_until_dup_count: Optional[int] = None):
        self.score_until_dup_count = score_until_dup_count
        self.known_smiles = {}
        logger.info(f"score_until_dup_count: {self.score_until_dup_count}")

    def step_reward(self, current_state: State, next_state: Optional[State]) -> float:
        raise NotImplementedError()

    def final_reward(self, state: State) -> float:
        raise NotImplementedError()

    def calc_dup_penalty(
        self, smiles: str, score: float, default_score: float = 0.0
    ) -> float:
        #表示允许重复生成该分子的最大次数。如果超过这个次数，就会触发惩罚。
        if self.score_until_dup_count is None:
            return default_score
        known_smiles = self.known_smiles
        cnt = known_smiles[smiles] = known_smiles.get(smiles, 0) + 1
        if cnt > self.score_until_dup_count:
            logger.info(f"NOT new mol cnt={cnt}: {smiles}")
            after_score = -max(0.0, score)
            logger.info(f"calc dup {score} --> {after_score}")
            return after_score
        else:
            logger.info(f"new mol cnt={cnt}: {smiles}")
            return default_score


class DummyReward(BaseReward):
    def step_reward(self, current_state: State, next_state: Optional[State]) -> float:
        return 0.0

    def final_reward(self, state: State) -> float:
        return 0.0
