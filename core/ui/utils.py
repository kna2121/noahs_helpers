import pygame
from typing import Literal


def write_at(
    screen: pygame.Surface,
    font: pygame.font.Font,
    line: str,
    coord: tuple[int, int],
    align: Literal["left", "center", "right"] = "center",
    color=(0, 0, 0),
):
    text = font.render(line, True, color)

    # get rectangle to center the text
    match align:
        case x if x == "left":
            rect = text.get_rect()
            rect.midleft = coord
        case x if x == "center":
            rect = text.get_rect(center=coord)
        case x if x == "right":
            rect = text.get_rect()
            rect.midright = coord
        case _:
            raise Exception(f"invalid value for `align`: {align}")

    screen.blit(text, rect)
