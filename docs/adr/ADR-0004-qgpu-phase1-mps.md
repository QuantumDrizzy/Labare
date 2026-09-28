# ADR-0004 — QGPU phase 1: the same quantum instructions on a tensor backend (MPS)

**Status:** PRE-REGISTRATION. Sections 1–5 are frozen (SHA-256 in
`experiments/qgpu/PREREG_SHA256.txt`, verbatim copy beside it) before any code of this phase is
written. Results go in section 6 only.
**Date:** 2026-09-29
**Depends on:** ADR-0002 (QGPU), ADR-0003 (U-QPU phase 1: the instructions and the reference
backend), `docs/research/QPU-QGPU-2026-09-29.md`.

## 1. Decision

QGPU is the rung that holds what a state vector cannot: many qubits with little entanglement. In
phase 1 it is a **matrix product state (MPS) backend** behind the **same** quantum instructions
as the U-QPU.
- A program does not change.
- A new allocation instruction picks the backend.
- The gate between them is **QGPU = QPU**: the same instruction stream on both backends gives
  the same state wherever the MPS is not truncated.

**The state.**
- One tensor per qubit, `A[q]` of shape (χ_left, 2, χ_right), complex f64.
- Qubit order is little-endian, as in ADR-0003.

**The operations.**

| operation | how |
|---|---|
| one-qubit gate | applied to `A[q]` alone; exact |
| two-qubit gate on q, q+1 | contract, SVD (one-sided complex Jacobi, zero dependencies), keep at most χ_max singular values, split back |
| two-qubit gate on distant qubits | SWAP network to adjacency, gate, SWAPs back; every SWAP is an SVD and can truncate |
| `qmeas q` | P(1) from a full transfer-matrix contraction ⟨ψ\|P₁(q)\|ψ⟩/⟨ψ\|ψ⟩, outcome from the seeded RNG, `A[q]` projected and rescaled |
| `qamp` | product of the selected physical slices; O(n·χ²) |

**The refusal: truncation is never hidden.**
- Every SVD adds its discarded weight ε = Σ_discarded s² / Σ s² to a ledger.
- The backend reports the fidelity bound F_bound = Π(1 − ε_k).
- A new instruction `qtrunc rd` reads F_bound into rd (f64). A program can refuse its own answer.

**New instructions:**
- `qmps n, χ_max`: MPS register of n qubits, 1 ≤ n ≤ 128, 1 ≤ χ_max ≤ 256.
- `qtrunc rd`: read F_bound; 1.0 exactly for the state vector backend.

## 2. Falsifiers (each must hold; any failure is reported, never tuned)

- **G1 — exact capacity equals the oracle.**
  - ADR-0003's 200 F3 circuits (n 1..12, depth 40, same file and seed), with `qmps n, 2^⌈n/2⌉`
    replacing `qalloc n`, so no truncation is possible.
  - Assembled and run through the ISA with `qamp` readout. Amplitudes against the same Qiskit
    oracle ≤ **1e-10**. Every circuit reports F_bound ≥ 1 − 1e-12.
- **G2 — QGPU = QPU.** For the same 200 circuits, the MPS backend's amplitudes against the
  state-vector backend's, both through the ISA: ≤ 1e-10.
- **G3 — the bound does not lie.**
  - 100 random circuits (seed 20260930), n = 10, depth 60, with χ_max = 4, so truncation happens.
  - For each: the true fidelity F = \|⟨ψ_exact\|ψ_mps⟩\|² / ⟨ψ_mps\|ψ_mps⟩, with ψ_exact from the
    state vector.
  - Pass: F ≥ F_bound − 1e-9 for every circuit, and at least 80 of the 100 truncated
    (F_bound < 1), so the test is not vacuous.
- **G4 — beyond a state vector.**
  - GHZ on **100 qubits** (`qg1.h 0`, `qcx q, q+1` for q = 0..98) with χ_max = 2.
  - Amplitudes of \|0…0⟩ and \|1…1⟩ are 1/√2 to ≤ 1e-12; a random other basis index is 0.
    F_bound = 1.
  - 1,000 shots, each a fresh preparation measuring all 100 qubits: every shot is all-0 or
    all-1, and the all-0 fraction is within 4σ of 0.5.
  - A state vector of 100 qubits would need 2¹⁰⁰ × 16 B; `qalloc 100` must refuse.
- **G5 — feed-forward on the tensor backend.** `programs/qpu_teleport.uasm` with `qalloc 3`
  replaced by `qmps 3, 4`: 1,000 Haar states (ADR-0003 F5's generator and seed), worst
  \|1 − F\| ≤ 1e-12, all four branches taken.
- **G6 — nothing else breaks.** All 79 existing tests stay green, and clippy (all targets) stays
  clean.

## 3. Budgets and declared choices

- **1e-10 for G1 and G2**, looser than ADR-0003's 1e-12, because each SVD adds a Jacobi rounding
  and depth 40 chains many. Stated before the run and not widened after.
- **Jacobi stopping rule.** Off-diagonal norm ≤ 1e-15 × Frobenius norm, or 60 sweeps, whichever
  comes first. Hitting 60 sweeps is reported.
- **Singular values below 1e-15 × s_max are dropped** as numerical zero. Their weight is counted
  in ε.

## 4. What is not claimed

- **Not a GPU yet.** "QGPU" names the rung. The contractions run on the host in f64. Phase 2
  moves them to the GPU through LYTH (ADR-0028's declared split) and the ZIPPER2 path, each held
  to this backend.
- No speed.
- No noise.
- No claim that an MPS represents highly entangled states efficiently. G3 shows what truncation
  costs, and F_bound says so.

## 5. What would change the plan

- If G1 fails at exact capacity, the SVD or the SWAP network is wrong. G3 and G4 are not run
  until it passes.
- If G3 finds F < F_bound, the bound is wrong, and QGPU does not claim a fidelity until it is
  replaced.

## 6. Results

(filled after the run)
