from abc import ABC, abstractmethod
from gymnasium import Env

from typing import Any, Optional, TypedDict

from matplotlib.axes import Axes
from numpy.typing import NDArray
import numpy as np


class MDPInfo(TypedDict):
    label: str


class MDP[ObsType, ActType](Env[ObsType, ActType], ABC):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 5,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.fig = None
        self.step_count = 0
        self.initial_state = np.array([0, 0])

    @property
    def render_shape(self) -> tuple[int, ...]:
        raise NotImplementedError("MDP must implement render_shape property")

    @property
    def render_extent(self) -> tuple[int, ...]:
        raise NotImplementedError("MDP must implement render_extent property")

    @property
    def n_transitions(self) -> int:
        raise NotImplementedError("MDP must implement n_transitions property")

    @property
    def n_actions(self) -> int:
        raise NotImplementedError("MDP must implement n_transitions property")

    @property
    def excluded_states(self) -> set[ObsType]:
        return set()

    @property
    @abstractmethod
    def states(self) -> NDArray:
        raise NotImplementedError("MDP must implement states method")

    @abstractmethod
    def successors_distribution(self, state, action) -> list[tuple[float, Any]]:
        raise NotImplementedError("MDP must implement successors_distribution method")

    @abstractmethod
    def get_label(self, state) -> str:
        raise NotImplementedError("MDP must implement get_label method")

    @abstractmethod
    def outline(self, ax: Axes, only_edges: bool = False):
        raise NotImplementedError("MDP must implement outline method")

    @abstractmethod
    def draw_actions(
        self,
        ax: Axes,
        actions: NDArray,
        arrow_color: Optional[str] = None,
        skip: Optional[NDArray[bool]] = None,
    ):
        raise NotImplementedError("MDP must implement draw_actions method")

    @abstractmethod
    def draw_trajectory(self, ax: Axes, trajectory: NDArray):
        raise NotImplementedError("MDP must implement draw_trajectory method")
