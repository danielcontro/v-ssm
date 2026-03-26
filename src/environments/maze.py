from typing import Any
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
from src.environments.slippery_grid import SlipperyGrid, StateType
from src.environments.utils import RectangularRegion, Wall


class Maze(SlipperyGrid):
    def __init__(self, scale: float = 1):
        super().__init__(
            width=int(scale * 10),
            height=int(scale * 10),
        )

        self.h_walls = [
            Wall((int(scale * 0), int(scale * 3)), (int(scale * 7), int(scale * 3))),
            Wall((int(scale * 1), int(scale * 5)), (int(scale * 2), int(scale * 5))),
            Wall((int(scale * 5), int(scale * 5)), (int(scale * 7), int(scale * 5))),
            Wall((int(scale * 3), int(scale * 7)), (int(scale * 10), int(scale * 7))),
            Wall((int(scale * 7), int(scale * 8)), (int(scale * 9), int(scale * 8))),
            Wall((int(scale * 0), int(scale * 9)), (int(scale * 2), int(scale * 9))),
        ]
        self.v_walls = [
            Wall((int(scale * 2), int(scale * 0)), (int(scale * 2), int(scale * 2))),
            Wall((int(scale * 8), int(scale * 0)), (int(scale * 8), int(scale * 2))),
            Wall((int(scale * 4), int(scale * 1)), (int(scale * 4), int(scale * 3))),
            Wall((int(scale * 1), int(scale * 5)), (int(scale * 1), int(scale * 8))),
            Wall((int(scale * 5), int(scale * 5)), (int(scale * 5), int(scale * 7))),
            Wall((int(scale * 8), int(scale * 5)), (int(scale * 8), int(scale * 7))),
            Wall((int(scale * 3), int(scale * 8)), (int(scale * 3), int(scale * 10))),
            Wall((int(scale * 7), int(scale * 8)), (int(scale * 7), int(scale * 10))),
        ]

        self.initial_states = [
            np.array([0, 0], dtype=int),
            np.array([0, 1], dtype=int),
            np.array([1, 0], dtype=int),
            np.array([1, 1], dtype=int),
        ]

        self.regions = {
            "a": RectangularRegion(
                int(scale * 0), int(scale * 0), int(scale * 2), int(scale * 2)
            ),
            "b": RectangularRegion(
                int(scale * 7), int(scale * 8), int(scale * 2), int(scale * 2)
            ),
            "c": RectangularRegion(
                int(scale * 8), int(scale * 0), int(scale * 2), int(scale * 2)
            ),
            "d": RectangularRegion(
                int(scale * 0), int(scale * 9), int(scale * 2), int(scale * 1)
            ),
            "h1": RectangularRegion(
                int(scale * 3), int(scale * 2), int(scale * 1), int(scale * 1)
            ),
            "h2": RectangularRegion(
                int(scale * 0), int(scale * 3), int(scale * 2), int(scale * 2)
            ),
            "h3": RectangularRegion(
                int(scale * 8), int(scale * 6), int(scale * 2), int(scale * 1)
            ),
            "e1": RectangularRegion(
                int(scale * 0), int(scale * 5), int(scale * 1), int(scale * 1)
            ),
            "e2": RectangularRegion(
                int(scale * 5), int(scale * 5), int(scale * 2), int(scale * 2)
            ),
            "e3": RectangularRegion(
                int(scale * 8), int(scale * 5), int(scale * 2), int(scale * 1)
            ),
        }

        self.unsafe = []

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[StateType, dict[str, Any]]:
        self.step_count = 0
        if options and options.get("deterministic_init"):
            self.current_state = self.initial_state
        else:
            reset = True
            low = np.array([0, 0], dtype=int)
            high = np.array([self.width, self.height], dtype=int)
            while reset:
                self.current_state = self.np_random.integers(
                    low=low, high=high, size=(2,), dtype=int
                )
                reset = (
                    self.regions["h1"].contains(tuple(list(self.current_state)))
                    or self.regions["h2"].contains(tuple(list(self.current_state)))
                    or self.regions["h3"].contains(tuple(list(self.current_state)))
                )
        return self.current_state, {"label": self.get_label(self.current_state)}

    def get_label(self, state: StateType) -> str:
        if self.regions["a"].contains(state):
            return "A"
        elif self.regions["b"].contains(state):
            return "B"
        elif self.regions["c"].contains(state):
            return "C"
        elif self.regions["d"].contains(state):
            return "D"
        elif self.regions["e1"].contains(state):
            return "E"
        elif self.regions["e2"].contains(state):
            return "E"
        elif self.regions["e3"].contains(state):
            return "E"
        elif self.regions["h1"].contains(state):
            return "H"
        elif self.regions["h2"].contains(state):
            return "H"
        elif self.regions["h3"].contains(state):
            return "H"
        else:
            return ""

    def outline(self, ax: Axes, only_edges: bool = False):
        super().outline(ax, only_edges)

        lines = []
        stroke = 5
        for wall in self.h_walls + self.v_walls:
            lines.append([wall.p1, wall.p2])

        ax.add_collection(
            LineCollection(lines, colors="black", linewidths=stroke, zorder=10)
        )
        for name, region in self.regions.items():
            match name:
                case "a":
                    color = "green"
                case "b":
                    color = "lightgreen"
                case "c":
                    color = "blue"
                case "d":
                    color = "lightblue"
                case "e1":
                    color = "orange"
                case "e2":
                    color = "orange"
                case "e3":
                    color = "orange"
                case "h1":
                    color = "red"
                case "h2":
                    color = "red"
                case "h3":
                    color = "red"
                case _:
                    raise ValueError()

            ax.add_patch(
                Rectangle(
                    (region.x, region.y),
                    region.w,
                    region.h,
                    color=color,
                    fill=(not only_edges),
                    alpha=0.5 if not only_edges else 1,
                    linewidth=2,
                    zorder=20,
                )
            )

        return
