from typing import Literal

from PIL import Image
from pydantic import BaseModel


class Point(BaseModel):
    x: int
    y: int


class Box(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class MaskLabeledPoint(Point):
    label: Literal[0, 1] = 1


class MaskLabeledBox(Box):
    label: Literal[0, 1] = 1


class GeneratedMask(BaseModel):
    image: Image.Image
    score: float

    class Config:
        arbitrary_types_allowed = True
