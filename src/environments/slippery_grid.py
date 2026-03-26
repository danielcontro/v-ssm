from enum import Enum

from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.spaces import Discrete, MultiDiscrete
from numpy.typing import NDArray

from src.environments.mdp import MDP
from src.environments.utils import RectangularRegion, Wall

from typing import Any, Optional


class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4


type StateType = NDArray[np.integer]
type ActionType = np.integer


class SlipperyGrid(MDP[StateType, ActionType]):
    def __init__(
        self,
        width: int,
        height: int,
        initial_state: StateType = np.zeros(shape=(2,), dtype=int),
        slip_prob: float = 0.3,
        render_mode: Optional[str] = None,
    ):
        # NOTE: The state is represented as (x, y) coordinates, where (0, 0) is the bottom-left corner of the grid.
        super().__init__(render_mode)

        self.render_mode = "human"

        self.width = width
        self.height = height

        self.action_space = Discrete(len(Action))
        self.observation_space = MultiDiscrete([self.width, self.height])

        self.slip_prob = slip_prob
        self.initial_state = initial_state
        self.current_state = initial_state
        self.deterministic_states: list[StateType] = []
        self.h_walls: list[Wall] = []
        self.v_walls: list[Wall] = []

    @property
    def render_shape(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def render_extent(self) -> tuple[int, int, int, int]:
        return (0, self.width, 0, self.height)

    @property
    def n_transitions(self) -> int:
        return (
            self.width * self.height * len(Action) * (3 if self.slip_prob > 0 else 1)
        )  # Slippy transitions: 1 for intended action, 2 for slip actions

    @property
    def n_actions(self) -> int:
        return self.action_space.n

    @property
    def states(self) -> NDArray:
        return np.indices((self.width, self.height)).transpose(1, 2, 0).reshape(-1, 2)

    def deterministic_ring_around_region(
        self, region: RectangularRegion
    ) -> list[StateType]:
        deterministic_ring = []
        # Bottom edge
        if region.y > 0:
            deterministic_ring.extend(
                [
                    (x, region.y - 1)
                    for x in range(
                        max(0, region.x - 1),
                        min(self.width, region.x + region.w + 1),
                    )
                ]
            )

        # Top edge
        if region.y + region.h < self.height:
            deterministic_ring.extend(
                [
                    (x, region.y + region.h)
                    for x in range(
                        max(0, region.x - 1),
                        min(self.width, region.x + region.w + 1),
                    )
                ]
            )

        # Left edge
        if region.x > 0:
            deterministic_ring.extend(
                [
                    (region.x - 1, y)
                    for y in range(
                        max(0, region.y),
                        min(self.height, region.y + region.h),
                    )
                ]
            )

        # Right edge
        if region.x + region.w < self.width:
            deterministic_ring.extend(
                [
                    (region.x + region.w, y)
                    for y in range(
                        max(0, region.y),
                        min(self.height, region.y + region.h),
                    )
                ]
            )

        return deterministic_ring

    def _move_up(self, state: StateType) -> StateType:
        return np.array(
            [
                state[0],
                min(self.height - 1, state[1] + 1),
            ]
        )

    def _move_down(self, state: StateType) -> StateType:
        return np.array(
            [
                state[0],
                max(0, state[1] - 1),
            ]
        )

    def _move_left(self, state: StateType) -> StateType:
        return np.array(
            [
                max(0, state[0] - 1),
                state[1],
            ]
        )

    def _move_right(self, state: StateType) -> StateType:
        return np.array(
            [
                min(self.width - 1, state[0] + 1),
                state[1],
            ]
        )

    def move(self, state: StateType, action: Action) -> StateType:
        match action:
            case Action.UP:
                if any(wall.is_above_of(state) for wall in self.h_walls):
                    return state
                return self._move_up(state)
            case Action.DOWN:
                if any(wall.is_below_of(state) for wall in self.h_walls):
                    return state
                return self._move_down(state)
            case Action.LEFT:
                if any(wall.is_left_of(state) for wall in self.v_walls):
                    return state
                return self._move_left(state)
            case Action.RIGHT:
                if any(wall.is_right_of(state) for wall in self.v_walls):
                    return state
                return self._move_right(state)
            case Action.STAY:
                return state
            case _:
                raise ValueError(f"Unknown action: {action}")

    def slip_action_set(self, action: Action) -> list[int]:
        match action:
            case Action.UP | Action.DOWN:
                return [Action.LEFT.value, Action.RIGHT.value]
            case Action.LEFT | Action.RIGHT:
                return [Action.UP.value, Action.DOWN.value]
            case Action.STAY:
                return [Action.STAY.value]
            case _:
                raise ValueError(f"Unknown action: {action}")

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[StateType, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.step_count = 0
        self.current_state = self.initial_state
        return self.current_state, {}

    def step(
        self, action: ActionType
    ) -> tuple[StateType, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        terminated = False
        truncated = False
        chosen_action = action
        if (slip := (self.np_random.random() < self.slip_prob)) and (
            self.current_state.tolist() not in self.deterministic_states
        ):
            action = self.np_random.choice(list(self.slip_action_set(Action(action))))

        self.current_state = self.move(self.current_state, Action(action))

        return (
            self.current_state,
            0.0,
            terminated,
            truncated,
            {
                "slipped": slip,
                "chosen_action": Action(chosen_action),
                "performed_action": action,
            },
        )

    def get_label(self, state: StateType) -> str:
        return ""

    def _rgb_render(self):
        # Create RGB grid
        grid = np.ones((self.width, self.height, 3), dtype=np.uint8) * 255
        # Draw agent (blue)
        grid[self.current_state[0], self.current_state[1]] = [0, 0, 255]
        return grid

    def render(self):
        if self.render_mode is None:
            return

        grid = self._rgb_render()

        if self.render_mode == "rgb_array":
            return grid

        if self.render_mode == "human":
            if self.fig is None:
                plt.ion()
                self.fig, self.ax = plt.subplots()
                self.ax.set_xticks(np.arange(-0.5, self.width, 1))
                self.ax.set_yticks(np.arange(-0.5, self.height, 1))
                self.ax.set_xticklabels([])
                self.ax.set_yticklabels([])
                self.ax.grid(True)

            assert self.ax is not None
            self.ax.clear()
            self.ax.imshow(
                grid,
                origin="lower",
            )
            self.ax.set_xticks(np.arange(-0.5, self.width, 1))
            self.ax.set_yticks(np.arange(-0.5, self.height, 1))
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
            self.ax.grid(True)

            plt.title(f"State: {self.current_state}, step: {self.step_count}")
            plt.draw()
            plt.pause(1.0 / self.metadata["render_fps"])

    def close(self):
        if self.fig is not None:
            plt.ioff()
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def outline(self, ax: Axes, only_edges: bool = False):
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks(np.arange(0, self.width + 1, 1))
        ax.set_yticks(np.arange(0, self.height + 1, 1))
        ax.set_xticklabels(np.arange(0, self.width + 1, 1))
        ax.set_yticklabels(np.arange(0, self.height + 1, 1))
        ax.grid(True)

        # Set the grid world limits
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        return

    def draw_actions(
        self,
        ax: Axes,
        actions: NDArray,
        arrow_color: Optional[str] = None,
        skip: Optional[NDArray[bool]] = None,
    ):
        # Grid of cell centers: (0,0) bottom-left, x right, y up. actions is indexed [x, y].
        # Use indexing='ij' so X, Y have shape (width, height) and X[x,y]=x+0.5, Y[x,y]=y+0.5.
        X, Y = np.meshgrid(
            np.arange(self.width) + 0.5,
            np.arange(self.height) + 0.5,
            indexing="ij",
        )

        # Initialize arrow components
        dx = np.zeros_like(actions, dtype=float)
        dy = np.zeros_like(actions, dtype=float)

        arrow_length = 0.7  # Length of the arrows
        # Map actions to dx, dy
        dx[actions == 2] = -1 * arrow_length  # left
        dx[actions == 3] = 1 * arrow_length  # right
        dy[actions == 0] = 1 * arrow_length  # up
        dy[actions == 1] = -1 * arrow_length  # down

        # Mask for arrows (exclude stay)
        arrow_mask = actions != 4

        # Mask for stay actions
        stay_mask = actions == 4

        if skip is not None:
            sk = np.asarray(skip, dtype=bool)
            arrow_mask = arrow_mask & ~sk
            stay_mask = stay_mask & ~sk

        color = arrow_color if arrow_color is not None else "white"
        ax.quiver(
            X[arrow_mask] - (dx[arrow_mask] / 2),
            Y[arrow_mask] - (dy[arrow_mask] / 2),
            dx[arrow_mask],
            dy[arrow_mask],
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
        )

        # Plot stay actions as red dots
        ax.scatter(
            X[stay_mask],
            Y[stay_mask],
            color="red",
            s=50,
            marker="o",
        )

        return

    def draw_trajectory(self, ax: Axes, trajectory: NDArray):
        xs = trajectory[:, 0] + 0.5
        ys = trajectory[:, 1] + 0.5

        ax.plot(
            xs,
            ys,
            color="blue",
            linewidth=2,
            marker="o",
            markersize=6,
            zorder=1,
            alpha=0.5,
        )
        ax.scatter(xs[0], ys[0], color="green", s=150, label="Start", zorder=2)
        ax.scatter(xs[-1], ys[-1], color="red", s=150, label="End", zorder=2)

        return

    def successors_distribution(
        self, state: StateType, action: ActionType
    ) -> list[tuple[float, Any]]:
        if state in self.deterministic_states:
            return [(1.0, self.move(state, Action(action)))]
        if Action(action) == Action.STAY:
            return [(1.0, self.move(state, Action(action)))]

        return [(1.0 - self.slip_prob, self.move(state, Action(action)))] + [
            (self.slip_prob / 2, self.move(state, Action(slip_action)))
            for slip_action in self.slip_action_set(Action(action))
        ]
