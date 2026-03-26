from typing import Optional
from src.automata.base import (
    Alphabet,
    AutomatonState,
    DAutomaton,
    TransitionFunction,
)

type StreetPair = tuple[set[AutomatonState], set[AutomatonState]]


class DSA(DAutomaton):
    def __init__(
        self,
        states: int,
        alphabet: Alphabet,
        transition_function: TransitionFunction,
        streett_pairs: list[StreetPair],
        initial_state: AutomatonState = 0,
        seed: int = 0,
        epsilon: float = 1,
        m: float = 1e2,
        sink_penalty: float = 1e3,
        name: Optional[str] = None,
    ):
        super().__init__(
            states,
            initial_state,
            alphabet,
            transition_function,
            seed,
            name,
        )

        self.streett_pairs = streett_pairs
        self.epsilon = epsilon
        self.m = m
        self.sink_penalty = sink_penalty

    @property
    def rejecting_states(self) -> set[AutomatonState]:
        return set(
            q
            for q in range(self.states)
            if all(self.transitions[q, a] == q for a in self.alphabet)
            and any(q in pair[0] for pair in self.streett_pairs)
            and all(q not in pair[1] for pair in self.streett_pairs)
        )

    def get_reward(self, state: AutomatonState) -> float:
        if state in self.rejecting_states:
            return -self.sink_penalty
        if any(state in pair[0] for pair in self.streett_pairs):
            return -self.epsilon
        elif any(state in pair[1] for pair in self.streett_pairs):
            return self.m if self.m is not None else 0
        else:
            return 0
