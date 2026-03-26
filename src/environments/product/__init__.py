from abc import ABC, abstractmethod
import os
from typing import Any, Optional
from gymnasium import Env
from gymnasium.spaces import MultiDiscrete
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from numpy import integer, ndarray
from numpy.typing import NDArray
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.animation import FuncAnimation

from src.automata.base import Automaton
from src.environments.mdp import MDP

type ActionType = integer
type StateType = NDArray[integer]

# NDArray: Q-table (spec_states, *mdp_states, n_actions) or
# action map (spec_states, *mdp_states).
type QValue = NDArray
type Policy = NDArray


class ProductMDP(Env[StateType, ActionType], ABC):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 5,
    }

    def __init__(
        self,
        mdp: MDP[StateType, ActionType],
        spec: Automaton,
        max_steps=1000,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.render_mode = render_mode

        self.mdp = mdp
        self.specification = spec

        self.action_space = mdp.action_space
        self.observation_space = MultiDiscrete(
            [self.specification.states] + self.mdp.observation_space.nvec.tolist(),
            start=[0] + self.mdp.observation_space.start.tolist(),
        )
        self.current_state = np.array([spec.initial_state] + list(mdp.reset()[0]))

        self.max_steps = max_steps
        self.step_count = 0

    @property
    def n_transitions(self):
        return self.specification.n_transitions * self.mdp.n_transitions

    @property
    def action_map_shape(self):
        return (
            self.specification.states,
            *self.mdp.observation_space.nvec,
        )

    @property
    def q_table_shape(self):
        return (
            self.specification.states,
            *self.mdp.observation_space.nvec,
            self.action_space.n,
        )

    def generate_trajectory(
        self,
        policy: Policy,
        max_length: int,
        seed: Optional[int] = None,
    ) -> NDArray:

        state = self.reset(seed=seed, options={"deterministic_init": True})[0]
        trajectory = []

        if isinstance(policy, ndarray):
            if policy.shape == self.action_map_shape:
                actions = policy
            elif policy.shape == self.q_table_shape:
                actions = np.argmax(policy, axis=-1)
            else:
                raise ValueError("Wrong policy shape given")

        for _ in range(max_length):
            action = actions[tuple(self.current_state)]
            next_state, _, terminated, truncated, _ = self.step(action)
            trajectory.append(state)

            if terminated or truncated:
                break
            state = next_state

        return np.array(trajectory)

    def draw_actions(
        self,
        ax: Axes,
        actions: NDArray,
        arrow_color: Optional[str] = None,
        skip: Optional[NDArray[bool]] = None,
    ):
        self.mdp.draw_actions(ax, actions, arrow_color, skip=skip)

    def draw_outline(self, ax: Axes, only_edges: bool = False):
        self.mdp.outline(ax, only_edges)

    def _blackout_cells(self, ax: Axes, cell_mask: NDArray[bool]) -> None:
        """Draw solid black unit squares where cell_mask[x, y] is True (MDP grid indices)."""
        obs = self.mdp.observation_space
        if not isinstance(obs, MultiDiscrete) or len(obs.nvec) != 2:
            return
        low = np.asarray(obs.start, dtype=int)
        nvec = np.asarray(obs.nvec, dtype=int)
        mask = np.asarray(cell_mask, dtype=bool)
        if mask.shape != tuple(nvec):
            return
        for xi in range(nvec[0]):
            for yi in range(nvec[1]):
                if not mask[xi, yi]:
                    continue
                x, y = low[0] + xi, low[1] + yi
                ax.add_patch(
                    Rectangle(
                        (x, y),
                        1,
                        1,
                        facecolor="black",
                        edgecolor="black",
                        linewidth=0,
                        zorder=25,
                    )
                )

    def plot_trajectories(
        self,
        policy: Policy,
        num_trajectories: int,
        max_length: int,
        save_dir: Optional[str] = None,
    ):
        trajectories = [
            self.generate_trajectory(policy, max_length)
            for _ in range(num_trajectories)
        ]
        fig, ax = plt.subplots(figsize=(20, 20))
        self.draw_outline(ax)
        for trajectory in trajectories:
            self.mdp.draw_trajectory(
                ax, trajectory[:, 1:]
            )  # Remove the automaton state from the trajectory for plotting

        plt.title("Policies Heatmap")
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, "trajectories.svg"))
        else:
            plt.show()
        plt.close(fig)

    def plot_trajectory_animation(
        self,
        policy: Policy,
        max_length: int,
        save_dir: Optional[str] = None,
    ) -> None:
        trajectory = self.generate_trajectory(policy, max_length)
        # Remove the automaton state from the trajectory for plotting
        trajectory = trajectory[:, 1:]

        fig, ax = plt.subplots(figsize=(20, 20))
        self.draw_outline(ax)

        # Create a line and points for trajectory, but don't plot them initially
        (trajectory_line,) = ax.plot(
            [], [], color="blue", linewidth=2, marker="o", markersize=6, zorder=1
        )
        start_dot = ax.scatter([], [], color="green", s=200, label="Start", zorder=2)
        end_dot = ax.scatter([], [], color="blue", s=200, label="End", zorder=2)

        # ax.invert_yaxis()  # Invert the y-axis so the origin is at the bottom-left
        ax.set_aspect("equal")
        ax.legend()

        def init():
            start_dot.set_offsets(trajectory[0] + 0.5)
            return start_dot, end_dot, trajectory_line

        # Animation update function
        def update(frame):
            # Update the trajectory (plot the path up to the current frame)
            trajectory_fragment = trajectory[: frame + 1] + 0.5
            trajectory_line.set_data(
                trajectory_fragment[:, 0],
                trajectory_fragment[:, 1],
            )

            # Update the position of the start and end dots (they stay in the same place, but need to be updated)
            end_dot.set_offsets(
                trajectory[frame] + 0.5,
            )

            return start_dot, end_dot, trajectory_line

        # Create the animation
        ani = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=min(max_length, len(trajectory)),
            interval=200,
            blit=True,
            repeat=True,
        )
        plt.title("Policy Trajectory Animation")
        if save_dir is not None:
            ani.save(
                os.path.join(save_dir, "trajectory_animation.gif"),
                writer="pillow",
                fps=10,
            )
        plt.close(fig)

    def plot_heatmap(
        self,
        max_q: Optional[NDArray],
        actions: NDArray,
        save_dir: Optional[str] = None,
        title: str = "Q-Net Heatmap",
        filename: str = "qnet_heatmap.svg",
        sign: bool = False,
        invariant_states: Optional[NDArray[bool]] = None,
        automaton_state: Optional[int] = None,
    ):
        # TODO: find smarter way to assign figsize based on the number of
        # automaton states and the shape of the MDP observation space
        n_spec = int(self.specification.states)
        if automaton_state is not None:
            if automaton_state < 0 or automaton_state >= n_spec:
                raise ValueError(
                    f"automaton_state {automaton_state} out of range [0, {n_spec})."
                )
            state_indices = [automaton_state]
        else:
            state_indices = list(range(n_spec))

        n_panels = len(state_indices)
        # With a fixed (20, 20) figure, one panel spans the full width and the same
        # point-sized fonts look tiny relative to the grid. Scale the figure so each
        # panel matches the size of one column in the all-states layout.
        base_w, base_h = 20.0, 20.0
        scale = n_panels / n_spec
        fig = plt.figure(figsize=(base_w * scale, base_h * scale))
        grid = AxesGrid(
            fig,
            111,
            nrows_ncols=(1, n_panels),
            cbar_mode="single" if max_q is not None else None,
            cbar_location="right",
            cbar_pad=0.2,
            axes_pad=0.2,
            share_all=True,
        )

        im = None
        arrow_color = "black" if max_q is None else None
        for grid_idx, i in enumerate(state_indices):
            ax = grid[grid_idx]
            self.draw_outline(ax, True)
            skip = None
            if invariant_states is not None:
                skip = ~np.asarray(invariant_states[i], dtype=bool)
            self.draw_actions(ax, actions[i], arrow_color=arrow_color, skip=skip)
            if max_q is not None:
                if not sign:
                    data = max_q[i]
                    base_cmap = "viridis"
                    vmin = np.min(max_q)
                    vmax = np.max(max_q)
                else:
                    data = np.sign(max_q[i])
                    base_cmap = "bwr"
                    vmin = -1
                    vmax = 1

                cmap = base_cmap
                if invariant_states is not None:
                    mask_i = ~invariant_states[i]
                    data = np.ma.array(data, mask=mask_i)
                    cmap = plt.get_cmap(base_cmap).copy()
                    cmap.set_bad("black")

                # data is indexed [x, y], but imshow expects [y, x].
                # See: https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.imshow.html
                plot_data = np.transpose(data)

                im = ax.imshow(
                    plot_data,
                    cmap=cmap,
                    origin="lower",
                    vmin=vmin,
                    vmax=vmax,
                    extent=self.mdp.render_extent,
                )

            if max_q is None and invariant_states is not None:
                self._blackout_cells(ax, ~invariant_states[i])

            ax.set_title(
                f"q={i}",
                fontsize=8,
                pad=2,
            )

        if im is not None:
            grid.cbar_axes[0].colorbar(im)

        plt.title(title)
        if save_dir is not None:
            plt.savefig(os.path.join(save_dir, filename))
        else:
            plt.show()
        plt.close(fig)

    def plot_policy(
        self,
        policy: Policy,
        save_dir: Optional[str] = None,
        *,
        invariant_states: Optional[NDArray[bool]] = None,
        automaton_state: Optional[int] = None,
    ):
        if policy.shape == self.action_map_shape:
            title = "Action Map"
            filename = "action_map.svg"
        elif policy.shape == self.q_table_shape:
            title = "Q-Table Heatmap"
            filename = "qtable_heatmap.svg"
        else:
            raise ValueError(
                f"Policy array shape {policy.shape} is invalid. "
                f"Expected action map shape {self.action_map_shape} or "
                f"Q-table shape {self.q_table_shape}."
            )

        if automaton_state is not None:
            stem, dot, ext = filename.rpartition(".")
            filename = f"{stem}_q{automaton_state}{dot}{ext}" if dot else filename

        if isinstance(policy, np.ndarray) and policy.shape == self.action_map_shape:
            self.plot_heatmap(
                None,
                policy,
                save_dir,
                title=title,
                filename=filename,
                invariant_states=invariant_states,
                automaton_state=automaton_state,
            )

        return

    @abstractmethod
    def plot_post_expectation(self, policy: Policy, save_dir: Optional[str] = None):
        raise NotImplementedError(
            "ProductMDP must implement plot_post_expectation method"
        )

    @abstractmethod
    def plot_post_expectation_diff(
        self, policy: Policy, save_dir: Optional[str] = None
    ):
        raise NotImplementedError(
            "ProductMDP must implement plot_post_expectation_diff method"
        )

    def render(self):
        return self.mdp.render()


def average_initial_qvalue(trajectories: list[list[NDArray]], policy: Policy) -> float:
    max_q = np.max(policy, axis=-1) if isinstance(policy, ndarray) else None

    return sum(
        max_q[trajectory[0][0]] if max_q is not None else max(policy(trajectory[0][0]))
        for trajectory in trajectories
    ) / len(trajectories)


def average_episode_reward(trajectories_infos: list[list[dict[str, Any]]]) -> float:
    return sum(
        sum(info["reward"] for info in trajectory_infos)
        for trajectory_infos in trajectories_infos
    ) / len(trajectories_infos)
