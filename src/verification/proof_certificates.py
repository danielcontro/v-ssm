from abc import ABC
from typing import Optional

from numpy.typing import NDArray
import numpy as np

from src.environments.product.deterministic import DProductMDP
from src.environments.product import Policy
from src.automata.dsa import StreetPair
from src.verification.bscc import bottom_bscc_labels
from src.verification.policy_evaluation import (
    compute_value_function_linear_solver,
)

type ProofCertificate = NDArray


class ProofCertBuilder(ABC):
    def __init__(self, product: DProductMDP, discount: float):
        self.product = product
        self.discount = discount
        self.state_dependent_discount = False

    def _reward_matrix(
        self, streett_pair: StreetPair, actions_map: Policy
    ) -> Optional[NDArray]:
        return None

    def build(self, actions_map: Policy, streett_pair: StreetPair) -> ProofCertificate:
        reward_matrix = self._reward_matrix(streett_pair, actions_map)
        V = compute_value_function_linear_solver(
            self.product,
            actions_map,
            self.discount,
            reward_matrix=reward_matrix,
            state_dependent_discount=self.state_dependent_discount,
        )
        max_value = 1 / (1 - self.discount)
        certificate = max_value - V
        return certificate


class BSCCsProofCertBuilder(ProofCertBuilder):
    def _reward_matrix(
        self, streett_pair: StreetPair, actions_map: Policy
    ) -> Optional[NDArray]:
        spec_states = int(self.product.specification.states)
        mdp_dims = tuple(
            int(x) for x in self.product.mdp.observation_space.nvec.tolist()
        )
        shape = (spec_states, *mdp_dims)
        n = int(np.prod(shape))
        P = np.zeros((n, n))
        for i, s in enumerate(np.ndindex(shape)):
            a = int(actions_map[s])
            for prob, succ in self.product.successors_distribution(list(s), a):
                j = np.ravel_multi_index(
                    tuple(int(x) for x in succ), shape, mode="raise"
                )
                P[i, j] += float(prob)

        bscc_labels = bottom_bscc_labels(P)
        u = np.unique(bscc_labels)
        n_bsccs = int(np.sum(u >= 0))

        fin_set, inf_set = streett_pair

        # BSCCs outside both fin_set and inf_set: no automaton state in the BSCC is in either set
        bscc_spec_states: dict[int, set[int]] = {}
        for i in range(n):
            comp_id = bscc_labels[i]
            if comp_id < 0:
                continue
            q = int(np.unravel_index(i, shape)[0])
            bscc_spec_states.setdefault(comp_id, set()).add(q)
        outside_bscc_ids = {
            comp_id
            for comp_id, qs in bscc_spec_states.items()
            if not (qs & fin_set) and not (qs & inf_set)
        }

        # Reward = 1 when automaton state in inf_set OR product state is in an outside BSCC
        reward_by_q = np.array(
            [1.0 if q in inf_set else 0.0 for q in range(spec_states)],
            dtype=float,
        )
        reward_matrix = np.broadcast_to(
            reward_by_q.reshape((spec_states,) + (1,) * len(mdp_dims)),
            shape,
        ).copy()
        for i in range(n):
            if bscc_labels[i] in outside_bscc_ids:
                reward_matrix.flat[i] = 1.0
        return reward_matrix


class StateDepDiscountProofCertBuilder(ProofCertBuilder):
    def __init__(
        self,
        product: DProductMDP,
        discount: float,
        positive_reward: float = 1.0,
        negative_reward: float = 1.0,
    ):
        super().__init__(product, discount)
        self.state_dependent_discount = True
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward

    def _reward_matrix(
        self, streett_pair: StreetPair, actions_map: Policy
    ) -> Optional[NDArray]:
        fin_set, inf_set = streett_pair
        spec_states = int(self.product.specification.states)
        mdp_dims = tuple(
            int(x) for x in self.product.mdp.observation_space.nvec.tolist()
        )

        reward_by_q = np.zeros(spec_states, dtype=float)
        for q in inf_set:
            reward_by_q[q] = self.positive_reward
        for q in fin_set - inf_set:
            reward_by_q[q] = -self.negative_reward
        shape = (spec_states, *mdp_dims)
        return np.broadcast_to(
            reward_by_q.reshape((spec_states,) + (1,) * len(mdp_dims)),
            shape,
        ).copy()
