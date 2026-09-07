"""Galerkin-supervised autonomous neural flows."""

from .problem import NeuralSemigroupProblem
from .semigroup import NeuralSemigroup

__version__ = "0.1.0"
__all__ = ["NeuralSemigroup", "NeuralSemigroupProblem"]
