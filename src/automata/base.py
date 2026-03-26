import random

from typing import Any, Optional
from abc import ABC, abstractmethod

type AutomatonState = int
type Symbol = str
type Alphabet = set[Symbol]
type TransitionRelation = dict[tuple[AutomatonState, Symbol], list[AutomatonState]]
type TransitionFunction = dict[tuple[AutomatonState, Symbol], AutomatonState]


class Automaton(ABC):
    def __init__(
        self,
        states: int,
        initial_state: AutomatonState,
        alphabet: Alphabet,
        transitions: Any,
        seed: int = 0,
        name: Optional[str] = "",
    ):
        self.seed = seed
        self.rng = random.Random(seed)
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.initial_state = initial_state
        self.current_state = initial_state
        self.name = name

    @property
    @abstractmethod
    def rejecting_states(self) -> set[AutomatonState]:
        pass

    @property
    def shortcircuit(self) -> bool:
        return False

    @property
    def n_transitions(self) -> int:
        raise NotImplementedError("Automaton must implement n_transitions method")

    def reset(self, randomize: bool = False) -> AutomatonState:
        if randomize:
            while (
                state := self.rng.randint(0, self.states - 1)
            ) in self.rejecting_states:
                continue
            self.current_state = state
        else:
            self.current_state = self.initial_state
        return self.current_state

    @abstractmethod
    def get_reward(self, state: AutomatonState) -> float:
        raise NotImplementedError("Automaton must implement get_reward method")


class DAutomaton(Automaton):
    def __init__(
        self,
        states: int,
        initial_state: AutomatonState,
        alphabet: Alphabet,
        transition_function: TransitionFunction,
        seed: int = 0,
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
        self.transitions: TransitionFunction = transition_function

    @property
    def n_transitions(self) -> int:
        return sum(successors for successors in self.transitions.values())

    def step(self, symbol: Symbol) -> AutomatonState:
        self.current_state = self.transition(self.current_state, symbol)
        return self.current_state

    def transition(self, state: AutomatonState, symbol: Symbol) -> AutomatonState:
        if symbol not in self.alphabet:
            symbol = ""
        successor = self.transitions.get((state, symbol), None)
        assert successor is not None, (
            f"No transition defined for state {state} and symbol '{symbol}'"
        )
        return successor


class NDAutomaton(Automaton):
    def __init__(
        self,
        states: int,
        initial_state: AutomatonState,
        alphabet: Alphabet,
        transition_relation: TransitionRelation,
        seed: int = 0,
    ):
        super().__init__(
            states,
            initial_state,
            alphabet,
            transition_relation,
            seed,
        )
        self.transitions: TransitionRelation = transition_relation

    @property
    def action_space(self) -> int:
        return max(len(successors) for successors in self.transitions.values())

    @property
    def n_transitions(self) -> int:
        return sum(len(successors) for successors in self.transitions.values())

    def n_available_actions(self, state: AutomatonState, symbol: Symbol) -> int:
        return len(self.transition(state, symbol))

    def step(self, symbol: Symbol, action: int) -> AutomatonState:
        successors = self.transition(self.current_state, symbol)
        assert len(successors) > 0, (
            f"No successors for state {self.current_state} and symbol '{symbol}'"
        )
        if action >= len(successors):
            print(
                f"Warning: action {action} is out of bounds for state {self.current_state} and symbol '{symbol}', selecting action 0 out of {len(successors)}"
            )
            self.current_state = successors[0]
        else:
            self.current_state = successors[action]

        return self.current_state

    def transition(self, state: AutomatonState, symbol: Symbol) -> list[AutomatonState]:
        if symbol not in self.alphabet:
            symbol = ""
        successors = self.transitions.get((state, symbol), None)
        assert successors is not None, (
            f"No transition defined for state {state} and symbol '{symbol}'"
        )
        return successors
