"""F3 of ADR-0003: the independent oracle for the U-QPU.

    python tools/qpu_oracle.py

Writes experiments/u-qpu/f3_circuits.json (200 random circuits, seed 20260929,
n uniform in 1..12, depth 40, the phase-1 gate set) and f3_qiskit.json (their
final state vectors from Qiskit's quantum_info.Statevector, an implementation
independent of MTLB). tests/qpu_oracle.rs then runs every circuit on MTLB
through the ISA (assembled programs, qamp readout) and compares.

Qubit order is little-endian in both, as ADR-0003 declares.
"""

import json
import math
import random
import struct
import sys
from pathlib import Path

import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

OUT = Path(__file__).resolve().parent.parent / "experiments" / "u-qpu"
SEED, N_CIRCUITS, DEPTH, MAX_N = 20260929, 200, 40, 12
FIXED = ["h", "x", "y", "z", "s", "sdg", "t", "tdg"]
ROT = ["rx", "ry", "rz"]
TWO = ["cx", "cz"]


def circuit(rng: random.Random) -> dict:
    n = rng.randint(1, MAX_N)
    kinds = FIXED + ROT + (TWO if n >= 2 else [])
    gates = []
    for _ in range(DEPTH):
        k = rng.choice(kinds)
        if k in FIXED:
            gates.append([k, rng.randrange(n)])
        elif k in ROT:
            gates.append([k, rng.randrange(n), rng.uniform(0.0, 2.0 * math.pi)])
        else:
            a, b = rng.sample(range(n), 2)
            gates.append([k, a, b])
    return {"n": n, "gates": gates}


def run_qiskit(c: dict) -> list[list[float]]:
    qc = QuantumCircuit(c["n"])
    for g in c["gates"]:
        getattr(qc, g[0])(*g[1:]) if g[0] not in ROT else getattr(qc, g[0])(g[2], g[1])
    sv = Statevector(qc).data
    return [[float(a.real), float(a.imag)] for a in sv]


def main() -> None:
    rng = random.Random(SEED)
    circuits = [circuit(rng) for _ in range(N_CIRCUITS)]
    oracle = [run_qiskit(c) for c in circuits]
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"seed": SEED, "circuits": N_CIRCUITS, "depth": DEPTH, "max_n": MAX_N, "qiskit": qiskit.__version__,
            "python": sys.version.split()[0]}
    (OUT / "f3_circuits.json").write_text(json.dumps({"meta": meta, "circuits": circuits}), encoding="utf-8")
    (OUT / "f3_qiskit.json").write_text(json.dumps({"meta": meta, "states": oracle}), encoding="utf-8")
    # The same, dependency-free for the Rust side: every f64 as its exact bits in hex.
    bits = lambda x: "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]
    lines = []
    for c in circuits:
        lines.append(f"circuit {c['n']} {len(c['gates'])}")
        for g in c["gates"]:
            lines.append(" ".join([g[0], str(g[1])] + ([bits(g[2])] if g[0] in ROT else [str(x) for x in g[2:]])))
    (OUT / "f3_circuits.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "f3_qiskit.txt").write_text(
        "\n".join(" ".join(bits(re) + bits(im) for re, im in st) for st in oracle) + "\n", encoding="utf-8")
    ns = [c["n"] for c in circuits]
    print(f"F3 oracle: {N_CIRCUITS} circuits, n {min(ns)}..{max(ns)}, {sum(2**n for n in ns)} amplitudes, qiskit {qiskit.__version__}")


if __name__ == "__main__":
    main()
