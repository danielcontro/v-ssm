import numpy as np
from numpy.typing import NDArray

from src.environments.mdp import MDP
from src.automata.base import DAutomaton


from typing import Any, Optional

from src.environments.product import Policy, ProductMDP


class DProductMDP(ProductMDP):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 5,
    }

    def __init__(
        self,
        mdp: MDP,
        spec: DAutomaton,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__(mdp, spec, max_steps, render_mode)

        self.specification = spec

    def reachable_states(self, policy: NDArray, initial_states: NDArray) -> NDArray:
        """
        Compute the set of product states reachable under a given deterministic
        action-map policy, starting from a set of initial states.

        Parameters
        ----------
        policy:
            NDArray of shape
            (spec_states, *mdp.observation_space.nvec) containing integer
            action IDs for each product state (an action map).
        initial_states:
            Boolean NDArray of the same shape as `policy` indicating which
            product states are initially reachable.

        Returns
        -------
        NDArray[bool]:
            Boolean NDArray of the same shape as `policy` where an entry is
            True iff the corresponding product state is reachable from
            `initial_states` by repeatedly following `policy`.
        """
        state_shape = (self.specification.states, *self.mdp.observation_space.nvec)

        if policy.shape != state_shape:
            raise ValueError(
                f"Policy array shape {policy.shape} is invalid. Expected action map "
                f"shape {state_shape}."
            )

        if initial_states.shape != state_shape:
            raise ValueError(
                "Initial states mask has invalid shape "
                f"{initial_states.shape}. Expected {state_shape}."
            )

        reachable = initial_states.astype(bool, copy=True)
        frontier = reachable.copy()

        # Fixed-point expansion of reachable set
        while np.any(frontier):
            next_frontier = np.zeros_like(frontier, dtype=bool)

            for idx in np.argwhere(frontier):
                idx_tuple = tuple(int(i) for i in idx)
                q = idx_tuple[0]
                mdp_state_indices = list(idx_tuple[1:])
                state = [q, *mdp_state_indices]

                action = int(policy[idx_tuple])

                for prob, succ_state in self.successors_distribution(state, action):
                    if prob <= 0.0:
                        continue

                    succ_q = succ_state[0]
                    succ_mdp = succ_state[1:]
                    succ_idx = (succ_q, *succ_mdp)

                    if not reachable[succ_idx]:
                        reachable[succ_idx] = True
                        next_frontier[succ_idx] = True

            frontier = next_frontier

        return reachable

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed, options=options)

        reset = True
        while reset:
            # NOTE: Assuming that the reset methods of mdp and spec safely initialize the state
            s0, info = self.mdp.reset(seed=seed, options=options)
            self.specification.reset(
                randomize=options.get("randomize_spec", False) if options else False
            )

            self.current_label = self.mdp.get_label(s0)
            q0 = self.specification.step(self.current_label)

            # Assure that we don't end up in a sink state right away
            if q0 not in self.specification.rejecting_states:
                # info["priority"] = self.specification.coloring_function[q0]
                self.current_state = np.array([q0] + list(s0))
                reset = False

        self.step_count = 0

        return self.current_state, info

    def step(self, action):
        truncated = False

        mdp_state, _, terminated, _, info = self.mdp.step(action)
        self.current_label = self.mdp.get_label(mdp_state)
        spec_state = self.specification.step(self.current_label)
        reward = self.specification.get_reward(self.specification.current_state)
        info["label"] = self.current_label
        self.current_state = [spec_state] + list(mdp_state)

        self.step_count += 1

        if self.specification.shortcircuit:
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        return self.current_state, reward, terminated, truncated, info

    def render(self):
        return self.mdp.render()

    def close(self) -> None:
        return self.mdp.close()

    def successors_distribution(
        self, state: list[int], action: int
    ) -> list[tuple[float, list[int]]]:
        return [
            (
                prob,
                [self.specification.transition(state[0], self.mdp.get_label(mdp_succ))]
                + list(mdp_succ),
            )
            for prob, mdp_succ in self.mdp.successors_distribution(
                tuple(state[1:]), action
            )
        ]

    def _get_expected_qtable_values(self, max_q: NDArray, actions: NDArray) -> NDArray:
        # TODO: make expectation computation general and more efficient by vectorizing it
        return np.array(
            [
                [
                    [
                        sum(
                            (
                                max_q[succ_q, succ_x, succ_y] * prob
                                for prob, (
                                    succ_q,
                                    succ_x,
                                    succ_y,
                                ) in self.successors_distribution(
                                    [q, x, y], actions[q, x, y]
                                )
                            ),
                            0.0,
                        )
                        for y in range(self.mdp.render_shape[1])
                    ]
                    for x in range(self.mdp.render_shape[0])
                ]
                for q in range(self.specification.states)
            ]
        )

    def plot_post_expectation(self, policy: Policy, save_dir: Optional[str] = None):
        max_q = np.max(policy, axis=-1) if isinstance(policy, np.ndarray) else None
        actions = np.argmax(policy, axis=-1) if isinstance(policy, np.ndarray) else None
        expected_values, actions = (
            self._get_expected_qtable_values(max_q, actions),
            actions,
        )
        self.plot_heatmap(
            expected_values,
            actions,
            save_dir,
            "Expected Q-Values Heatmap",
            "qtable_expected_values.svg",
        )

    def plot_post_expectation_diff(
        self, policy: Policy, save_dir: Optional[str] = None
    ):
        max_q = np.max(policy, axis=-1) if isinstance(policy, np.ndarray) else None
        actions = np.argmax(policy, axis=-1) if isinstance(policy, np.ndarray) else None
        expected_values, actions = (
            self._get_expected_qtable_values(max_q, actions),
            actions,
        )
        diff = expected_values - max_q
        self.plot_heatmap(
            diff,
            actions,
            save_dir,
            "Diff Expected Q-Values Heatmap",
            "diff_qtable_expected_values.svg",
        )
        self.plot_heatmap(
            diff,
            actions,
            save_dir,
            "Sign Diff Expected Q-Values Heatmap",
            "sign_diff_qtable_expected_values.svg",
            sign=True,
        )
