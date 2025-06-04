"""Utility for performing meta analysis on .cocoon files.

This module aggregates quantum and chaos states from saved cocoon files and
visualises the resulting "dream space".  The original script was written in a
monolithic style.  It has been refactored into smaller functions for easier
maintenance and testing.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple
import asyncio

import matplotlib.pyplot as plt
import numpy as np


def simple_neural_activator(
    quantum_vec: Iterable[float], chaos_vec: Iterable[float]
) -> int:
    """Return 1 if the combined variance of inputs exceeds a threshold."""
    q_sum = sum(quantum_vec)
    c_var = np.var(list(chaos_vec))
    return int(q_sum + c_var > 1)


def codette_dream_agent(
    quantum_vec: Iterable[float], chaos_vec: Iterable[float]
) -> Tuple[List[float], List[float]]:
    """Generate dream-state vectors based on the provided states."""
    dream_q = [np.sin(q * np.pi) for q in quantum_vec]
    dream_c = [np.cos(c * np.pi) for c in chaos_vec]
    return dream_q, dream_c


def philosophical_perspective(qv: Iterable[float], cv: Iterable[float]) -> str:
    """Provide a whimsical philosophical label for the states."""
    m = max(qv) + max(cv)
    if m > 1.3:
        return "Philosophical Note: This universe is likely awake."
    return "Philosophical Note: Echoes in the void."


def load_cocoon(path: Path) -> dict:
    """Synchronously load a single cocoon file."""
    with path.open() as f:
        return json.load(f)["data"]


async def load_cocoon_async(path: Path) -> dict:
    """Asynchronously load a cocoon using a thread executor."""
    return await asyncio.to_thread(load_cocoon, path)


def analyse_cocoons(folder: Path) -> List[dict]:
    """Analyse cocoon files synchronously."""
    meta_mutations: List[dict] = []

    print("\nMeta Reflection Table:\n")
    header = (
        "Cocoon File | Quantum State | Chaos State | Neural | Dream Q/C | Philosophy"
    )
    print(header)
    print("-" * len(header))

    for path in folder.glob("*.cocoon"):
        try:
            data = load_cocoon(path)
            q = data.get("quantum_state", [0, 0])
            c = data.get("chaos_state", [0, 0, 0])
            neural = simple_neural_activator(q, c)
            dream_q, dream_c = codette_dream_agent(q, c)
            phil = philosophical_perspective(q, c)
            meta_mutations.append(
                {
                    "dreamQ": dream_q,
                    "dreamC": dream_c,
                    "neural": neural,
                    "philosophy": phil,
                }
            )
            print(f"{path.name} | {q} | {c} | {neural} | {dream_q}/{dream_c} | {phil}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: {path.name} failed ({exc})")

    return meta_mutations


async def analyse_cocoons_async(folder: Path) -> List[dict]:
    """Asynchronously analyse cocoon files in parallel."""
    meta_mutations: List[dict] = []

    print("\nMeta Reflection Table:\n")
    header = (
        "Cocoon File | Quantum State | Chaos State | Neural | Dream Q/C | Philosophy"
    )
    print(header)
    print("-" * len(header))

    tasks = [load_cocoon_async(p) for p in folder.glob("*.cocoon")]

    for coro in asyncio.as_completed(tasks):
        try:
            data = await coro
            q = data.get("quantum_state", [0, 0])
            c = data.get("chaos_state", [0, 0, 0])
            neural = simple_neural_activator(q, c)
            dream_q, dream_c = codette_dream_agent(q, c)
            phil = philosophical_perspective(q, c)
            meta_mutations.append(
                {
                    "dreamQ": dream_q,
                    "dreamC": dream_c,
                    "neural": neural,
                    "philosophy": phil,
                }
            )
            print(
                f"async | {q} | {c} | {neural} | {dream_q}/{dream_c} | {phil}"
            )
        except Exception as exc:  # pylint: disable=broad-except
            print(f"Warning: async load failed ({exc})")

    return meta_mutations


def plot_meta_dream(meta_mutations: List[dict]) -> None:
    if not meta_mutations:
        print("No valid cocoons found for meta-analysis.")
        return

    dq0 = [m["dreamQ"][0] for m in meta_mutations]
    dc0 = [m["dreamC"][0] for m in meta_mutations]
    ncls = [m["neural"] for m in meta_mutations]

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(dq0, dc0, c=ncls, cmap="spring", s=100)
    plt.xlabel("Dream Quantum[0]")
    plt.ylabel("Dream Chaos[0]")
    plt.title("Meta-Dream Codette Universes")
    plt.colorbar(sc, label="Neural Activation Class")
    plt.grid(True)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse Codette cocoon files")
    parser.add_argument(
        "folder", nargs="?", default=".", help="Folder containing .cocoon files"
    )
    parser.add_argument(
        "--async", dest="use_async", action="store_true", help="Use async loading"
    )
    args = parser.parse_args()

    if args.use_async:
        meta_mutations = asyncio.run(analyse_cocoons_async(Path(args.folder)))
    else:
        meta_mutations = analyse_cocoons(Path(args.folder))
    plot_meta_dream(meta_mutations)


if __name__ == "__main__":
    main()
