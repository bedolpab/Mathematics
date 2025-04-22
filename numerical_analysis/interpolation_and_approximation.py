"""Python modeling for Chapter 4 of (ISBN 0-471-43337-3)."""

import math
import numpy as np
from typing import List, Union


def linear_interpolation(
    x: Union[float, int],
    x_points: List[Union[int, float]],
    y_points: List[Union[int, float]],
) -> float:
    """Linear polynomial interpolation of two points."""
    numerator = (x_points[1] - x) * y_points[0] + (x - x_points[0]) * y_points[
        1
    ]
    denominator = x_points[1] - x_points[0]

    return np.divide(numerator, denominator)


def quadratic_interpolation(
    x: Union[float, int],
    x_points: List[Union[int, float]],
    y_points: List[Union[int, float]],
) -> float:
    """Quadratic polynomial interpolation of three points."""
    l_0 = np.divide(
        (x - x_points[1]) * (x - x_points[2]),
        (x_points[0] - x_points[1]) * (x_points[0] - x_points[2]),
    )

    l_1 = np.divide(
        (x - x_points[0]) * (x - x_points[2]),
        (x_points[1] - x_points[0]) * (x_points[1] - x_points[2]),
    )

    l_2 = np.divide(
        (x - x_points[0]) * (x - x_points[1]),
        (x_points[2] - x_points[0]) * (x_points[2] - x_points[1]),
    )

    return (y_points[0] * l_0) + (y_points[1] * l_1) + (y_points[2] * l_2)


def higher_degree_polynomial(
    x: Union[float, int],
    x_points: List[Union[int, float]],
    y_points: List[Union[int, float]],
) -> float:
    """Higher Order polynomial interpolation of n points."""
    n = len(x_points)
    total_approximation = 0

    for i in range(n):
        exclude_point = [x for x in x_points if x != x_points[i]]
        l_numerator = [(x - xn) for xn in exclude_point]
        l_denominator = [(x_points[i] - xn) for xn in exclude_point]

        total_approximation += y_points[i] * np.divide(
            math.prod(l_numerator), math.prod(l_denominator)
        )

    return total_approximation


# example of higher degree polynomial approximation
x_mock = [4.1168, 4.19236, 4.20967, 4.46908]
y_mock = [0.213631, 0.214232, 0.21441, 0.218788]

result = higher_degree_polynomial(x=10, x_points=x_mock, y_points=y_mock)

print(f"The result for f(10) is {result}")
