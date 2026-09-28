# ADR-0003 — U-QPU phase 1: quantum instructions and the reference backend

**Status:** PRE-REGISTRATION. Sections 1–5 are frozen (SHA-256 in
`experiments/u-qpu/PREREG_SHA256.txt`, verbatim copy beside it) before any code of this phase is
written. Results go in section 6 only.
**Date:** 2026-09-29
**Depends on:** ADR-0001 (the programme, U-QPU milestone), ADR-0002 (QGPU),
`docs/research/QPU-QGPU-2026-09-29.md`.

## 1. Decision

The ISA gets a quantum coprocessor, simulated.

- **State.** Instructions act on a register of up to 24 qubits held by a backend.
- **The only backend in phase 1** is an exact host state vector in f64 complex, written for
  clarity, not speed. It is the reference every later backend is held to (cuStateVec, MPS/QGPU,
  a real QPU).

**Qubit order:** little-endian, as in Qiskit. Qubit q is bit q of the basis index.

| instruction | meaning |
|---|---|
| `qalloc n` | fresh register of n qubits in \|0…0⟩ (1 ≤ n ≤ 24) |
| `qg1.<g> q` | one-qubit gate g ∈ {h, x, y, z, s, sdg, t, tdg} on qubit q (q is an immediate) |
| `qrot.<a> q, rs` | rotation Rx, Ry or Rz(θ) on q, θ = f64 in lane 0 of rs |
| `qcx c, t` / `qcz a, b` | CNOT (control c, target t) and CZ |
| `qmeas rd, q` | measure q in Z. rd = 0 or 1; the state collapses and renormalises |
| `qreset q` | measure q and flip it to \|0⟩ if the outcome was 1 |
| `qseed rs` | seed the measurement RNG (deterministic replay) |
| `qamp rd, rs` | read amplitude of basis index rs into rd as complex (re, im) (for tests) |

- **Feed-forward** is ordinary ISA control flow: `qmeas` writes a register, and a branch on that
  register conditions the next quantum instruction. No new mechanism.
- **Metrics.** Quantum instructions count in a new bucket, `quantum_ops`, at a declared cost of 1
  dispatch cycle each. **This is not a timing model.** Durations and latency against coherence
  are phase 2 and get their own ADR.

## 2. Falsifiers (each must hold; any failure is reported as a FAIL, not tuned)

- **F1 — closed forms.**
  - For every fixed gate, the backend's action on both basis states equals the textbook matrix
    to ≤ 1e-15 per amplitude.
  - Rotations: Rx, Ry and Rz at 8 angles, including 0, π/2, π and 3π/2.
  - Unitarity: ‖U†U − I‖max ≤ 1e-15.
- **F2 — placement.** For n = 1..8 and every qubit q, applying a gate on q equals the dense
  Kronecker construction I⊗…⊗U⊗…⊗I, built independently, to ≤ 1e-14. For two-qubit gates, every
  ordered pair (c, t) for n ≤ 5.
- **F3 — independent oracle.**
  - 200 random circuits (seed 20260929), n uniform in 1..12, depth 40, gates drawn from the
    phase-1 set.
  - They run on MTLB **through the ISA**, meaning assembled programs with `qamp` readout.
  - Qiskit 2.2.3 `quantum_info.Statevector` runs the same circuits.
  - Pass: max |Δamplitude| ≤ 1e-12 for every circuit. The circuits and results are written to
    `experiments/u-qpu/`.
- **F4 — measurement statistics.**
  - A Bell pair is measured 100,000 times, one fresh preparation per shot, seed fixed.
    |P(00) − 0.5| ≤ 4σ (σ = √(0.25/N)), and P(01) + P(10) = 0 exactly.
  - Replaying with the same seed gives the identical outcome sequence.
- **F5 — feed-forward in the ISA.**
  - An assembled teleportation program (`programs/qpu_teleport.uasm`): prepare a state on
    qubit 0, Bell pair on 1–2, Bell measurement of 0–1 with `qmeas`, then `bne`-conditioned
    X and Z corrections on qubit 2.
  - For 1,000 random input states (Haar-random, seed fixed) the fidelity of qubit 2 with the
    input is 1 to ≤ 1e-12.
  - The fidelity is computed from `qamp` readout of the final state, reduced over qubits 0–1.
- **F6 — nothing else breaks.** All 73 existing tests stay green, and `cargo clippy` stays clean.

## 3. Budgets and declared choices

- Tolerances (1e-15, 1e-14, 1e-12) follow f64 rounding over depth 40. They are stated here and
  not widened after a run.
- The Haar-random states come from normalised complex Gaussian pairs.
- 24 qubits is the host cap: 2²⁴ amplitudes × 16 B = 256 MiB.

## 4. What is not claimed

- No speed. The phase-1 backend is a reference.
- No noise model and no timing.
- No real hardware. Real-QPU runs need the user's IBM Quantum account; that is phase 2.
- "Quantum" here means a simulated coprocessor, stated as such.

## 5. What would change the plan

- If F3 fails on conventions (qubit order, phase), the convention is fixed in the ADR and every
  circuit re-run, and that is reported.
- If F3 fails on arithmetic, that is a backend bug, and phase 2 does not start until it is fixed.

## 6. Results (2026-09-29, first run; sections 1–5 unchanged)

All six falsifiers hold.

| | result | bound |
|---|---|---|
| **F1** closed forms | 8 fixed gates and 3 axes × 8 angles match hand-typed textbook matrices on both basis states; unitary | 1e-15 |
| **F2** placement | every gate on every qubit, n = 1..8, equals the dense I⊗…⊗U⊗…⊗I construction; CX and CZ on every ordered pair, n ≤ 5, equal their definitions | 1e-14 |
| **F3** independent oracle | 200 random circuits (n 1..12, depth 40), **assembled, encoded to the object format and back, run on the CPU, read out through `qamp` + `sq`**: 140,510 amplitudes against Qiskit 2.2.3 `Statevector`, worst \|Δ\| = **1.33e-15** (per-circuit record in `experiments/u-qpu/f3_result.txt`) | 1e-12 |
| **F4** statistics | Bell pair, 100,000 fresh shots: P(00) = 0.50105 (0.66 σ), P(01) + P(10) = 0; the seed replays the same sequence exactly | 4 σ, exact |
| **F5** feed-forward | `programs/qpu_teleport.uasm`, `qmeas` and `beq`-conditioned X/Z: 1,000 Haar states, worst \|1 − F\| = **2.9e-15**; all four correction branches taken (285 / 255 / 228 / 232) | 1e-12 |
| **F6** no regressions | 79/79 tests (73 before + 6), `cargo clippy --all-targets` clean | — |

**Is F3 able to fail? Checked outside the rules.** A mutant with `qcx` control and target
swapped failed on the first circuit (\|Δ\| = 0.48). The oracle distinguishes a wrong gate.

**Declared once more:** phase 1 is a reference, not a speed claim. It has no noise and no
timing. Phase 2 (durations and feed-forward latency against coherence, checked on a real
Heron QPU) needs its own ADR and the user's IBM Quantum account.
