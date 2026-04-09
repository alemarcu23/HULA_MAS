"""
Engine package for MARBLE-style multi-agent coordination in MATRX.

This package implements the Engine and EnginePlanner classes for orchestrating
decentralized multi-agent planning.
"""

from engine.engine_planner import EnginePlanner
from engine.iteration_data import IterationData

__all__ = ['EnginePlanner', 'IterationData']
