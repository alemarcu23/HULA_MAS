"""
IterationData dataclass for tracking iteration state in decentralized planning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class IterationData:
    """
    Tracks one iteration of decentralized planning.

    Each iteration represents one complete cycle of:
    1. All agents plan their tasks
    2. All agents execute their tasks
    3. Results are summarized
    4. Termination is checked
    """
    iteration: int = 0
    task_assignments: Dict[str, str] = field(default_factory=dict)  # agent_id -> task description
    task_results: List[Dict[str, Any]] = field(default_factory=list)  # List of agent results
    summary: str = ""  # Aggregated summary of iteration
    continue_simulation: bool = True  # Whether to continue to next iteration
    communications: List[str] = field(default_factory=list)  # Inter-agent messages
    score: float = 0.0  # Current simulation score
