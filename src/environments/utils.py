from dataclasses import dataclass
from typing import Iterable


@dataclass
class RectangularRegion:
    x: int
    y: int
    w: int
    h: int

    def contains(self, point: tuple[int, int]) -> bool:
        return (
            self.x <= point[0] < self.x + self.w
            and self.y <= point[1] < self.y + self.h
        )

    def positions(self) -> Iterable[tuple[int, int]]:
        for x in range(self.x, self.x + self.w):
            for y in range(self.y, self.y + self.h):
                yield (x, y)


@dataclass
class Wall:
    p1: tuple[int, int]
    p2: tuple[int, int]

    @property
    def is_vertical(self) -> bool:
        return self.p1[0] == self.p2[0]

    @property
    def is_horizontal(self) -> bool:
        return self.p1[1] == self.p2[1]

    def is_above_of(self, point: tuple[int, int]) -> bool:
        return (
            self.is_horizontal
            and self.p1[1] == point[1] + 1
            and self.p1[0] <= point[0] < self.p2[0]
        )

    def is_below_of(self, point: tuple[int, int]) -> bool:
        return (
            self.is_horizontal
            and self.p1[1] == point[1]
            and self.p1[0] <= point[0] < self.p2[0]
        )

    def is_left_of(self, point: tuple[int, int]) -> bool:
        return (
            self.is_vertical
            and self.p1[0] == point[0]
            and self.p1[1] <= point[1] < self.p2[1]
        )

    def is_right_of(self, point: tuple[int, int]) -> bool:
        return (
            self.is_vertical
            and self.p1[0] == point[0] + 1
            and self.p1[1] <= point[1] < self.p2[1]
        )
