"""Hierarchical deterministic random streams.

Every entity draws from its own generator derived from (scene seed, path...),
so adding or removing one entity never changes the random values of another.
Pure Python + numpy: usable both in the planner venv and inside Blender.
"""
import hashlib

import numpy as np


def derive_seed(*parts) -> int:
    key = "/".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def rng(*parts) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(derive_seed(*parts)))
