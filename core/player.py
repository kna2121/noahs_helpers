from __future__ import annotations
from abc import ABC, abstractmethod
from typing import final

from math import hypot

import pygame

from core.action import Action
from core.animal import Animal
from core.message import Message
from core.ui.utils import write_at
from core.views.player_view import PlayerView
from core.snapshots import HelperSurroundingsSnapshot

import core.constants as c


class Player(ABC):
    def __init__(self, id: int, ark_x: int, ark_y: int):
        self.id = id
        self.ark_position = (ark_x, ark_y)
        self.position = (float(ark_x), float(ark_y))
        self.flock: set[Animal] = set()

    def __str__(self) -> str:
        return f"{self.__module__.split('.')[-1]}"

    def __repr__(self) -> str:
        return str(self)

    @final
    def distance(self, other: Player) -> float:
        # this is not inherently bad, but just might
        # be an indicator that our calling logic is bad
        if self.id == other.id:
            raise Exception(f"{self}: Calculating distance with myself?")

        x1, y1 = self.position
        x2, y2 = other.position

        # pytharogas
        return (abs(x2 - x1) ** 2 + abs(y2 - y1) ** 2) ** 0.5

    @final
    def get_view(self) -> PlayerView:
        return PlayerView(self.id)

    @final
    def is_in_ark(self) -> bool:
        return (
            int(self.position[0]) == self.ark_position[0]
            and int(self.position[1]) == self.ark_position[1]
        )

    @final
    def is_message_valid(self, msg: int) -> bool:
        return 0 <= msg < c.ONE_BYTE

    @final
    def can_move_to(self, x: float, y: float) -> bool:
        if not (0 <= x < c.X and 0 <= y < c.Y):
            return False

        curr_x, curr_y = self.position
        return (abs(curr_x - x) ** 2 + abs(curr_y - y) ** 2) * 0.5 <= c.MAX_DISTANCE_KM

    @final
    def is_flock_full(self) -> bool:
        return len(self.flock) == c.MAX_FLOCK_SIZE

    @final
    def is_flock_empty(self) -> bool:
        return len(self.flock) == 0

    @final
    def move_towards(self, x: float, y: float) -> tuple[float, float]:
        # let's be conservative, we don't want to move too far
        step_size = c.MAX_DISTANCE_KM * 0.99

        cx, cy = self.position

        dx = x - cx
        dy = y - cy
        dist = hypot(dx, dy)

        if dist == 0:
            return cx, cy

        if dist <= step_size:
            return cx + dx, cy + dy

        scale = step_size / dist

        return cx + dx * scale, cy + dy * scale

    @final
    def draw(
        self, screen: pygame.Surface, font: pygame.font.Font, pos: tuple[int, int]
    ):
        text = font.render(f"h{self.id}", True, c.HELPER_COLOR)
        rect = text.get_rect(center=pos)
        screen.blit(text, rect)

    @final
    def draw_flock(
        self, screen: pygame.Surface, font: pygame.font.Font, start_pos: tuple[int, int]
    ):
        x, y = start_pos
        flist = list(self.flock) + [None] * (c.MAX_FLOCK_SIZE - len(self.flock))
        for i in range(c.MAX_FLOCK_SIZE):
            fi = flist[i]
            pos = (x, y)

            if fi is None:
                write_at(screen, font, "_", pos)
            else:
                fi.draw(screen, font, pos)

            x += 40

    @abstractmethod
    def check_surroundings(self, snapshot: HelperSurroundingsSnapshot) -> int:
        raise Exception("not implemented")

    @abstractmethod
    def get_action(self, messages: list[Message]) -> Action:
        raise Exception("not implemented")
