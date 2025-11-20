"""Explicit state-space exploration (Task 2)."""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Set, Tuple

# Add parent directory to path to allow imports when running as script
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from task1_parser.pnml_parser import PetriNet

MarkingDict = Dict[str, int]
MarkingTuple = Tuple[int, ...]


class ExplicitReachability:
    """Enumerate reachable markings of a 1-safe Petri net."""

    def __init__(self, net: PetriNet):
        self.net = net
        self._places: Tuple[str, ...] = tuple(sorted(net.places))
        self.reachable_markings: Set[MarkingTuple] = set()
        self.num_states: int = 0
        self.computation_time: float = 0.0

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def _marking_to_tuple(self, marking: MarkingDict) -> MarkingTuple:
        return tuple(marking.get(place, 0) for place in self._places)

    def _tuple_to_marking(self, marking: MarkingTuple) -> MarkingDict:
        return {place: marking[idx] for idx, place in enumerate(self._places)}

    # ------------------------------------------------------------------
    # Transition firing
    # ------------------------------------------------------------------

    def _is_enabled(self, marking: MarkingDict, transition: str) -> bool:
        return all(marking.get(place, 0) == 1 for place in self.net.input_arcs[transition])

    def _fire(self, marking: MarkingDict, transition: str) -> MarkingDict:
        successor = marking.copy()
        for place in self.net.input_arcs[transition]:
            successor[place] = 0
        for place in self.net.output_arcs[transition]:
            successor[place] = 1
        return successor

    # ------------------------------------------------------------------
    # Exploration
    # ------------------------------------------------------------------

    def _explore(self, frontier: Deque[MarkingDict], pop) -> Set[MarkingTuple]:
        visited: Set[MarkingTuple] = set()
        while frontier:
            current = pop()
            current_key = self._marking_to_tuple(current)
            if current_key in visited:
                continue
            visited.add(current_key)
            for transition in self.net.transitions:
                if not self._is_enabled(current, transition):
                    continue
                successor = self._fire(current, transition)
                frontier.append(successor)
        return visited

    def compute(self, strategy: str = "bfs") -> Set[MarkingTuple]:
        start = time.perf_counter()
        initial_marking = self.net.initial_marking
        frontier: Deque[MarkingDict] = deque([initial_marking])
        explored = self._explore(
            frontier,
            frontier.popleft if strategy.lower() == "bfs" else frontier.pop,
        )
        self.reachable_markings = explored
        self.num_states = len(explored)
        self.computation_time = time.perf_counter() - start
        return explored

    def compute_bfs(self) -> Set[MarkingTuple]:
        return self.compute("bfs")

    def compute_dfs(self) -> Set[MarkingTuple]:
        return self.compute("dfs")

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def print_reachable_markings(self, limit: int = 10) -> None:
        total = len(self.reachable_markings)
        sample = list(self.reachable_markings)[:limit]
        print(f"\nReachable markings (showing {len(sample)} of {total}):")
        print("-" * 60)
        print(f"Places: {self._places}")
        for idx, marking in enumerate(sample, start=1):
            print(f"{idx:>2}. {marking}")
        if total > limit:
            print(f"... and {total - limit} more")
        print()

    def compare_run(self) -> None:
        print("=" * 60)
        print("TASK 2: EXPLICIT REACHABILITY")
        print("=" * 60)
        self.compute_bfs()
        print(f"Reachable markings: {self.num_states}")
        print(f"Computation time: {self.computation_time:.4f} seconds")
        self.print_reachable_markings(limit=20)


if __name__ == "__main__":  # pragma: no cover - manual usage
    from task1_parser.pnml_parser import PNMLParser

    if len(sys.argv) != 2:
        print("Usage: python explicit_reachability.py <pnml_file>")
        raise SystemExit(1)

    model_path = sys.argv[1]
    try:
        net = PNMLParser.parse(model_path)
        net.print_summary()
        ExplicitReachability(net).compare_run()
    except Exception as exc:  # noqa: BLE001 - simple CLI helper
        print(f"Error: {exc}")
        raise SystemExit(2) from exc