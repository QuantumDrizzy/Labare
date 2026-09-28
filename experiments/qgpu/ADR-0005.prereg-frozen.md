# ADR-0005 — The fidelity bound under real truncation (ADR-0004 G3, again, not vacuous)

**Status:** PRE-REGISTRATION. Sections 1–4 are frozen (SHA-256 in
`experiments/qgpu/PREREG5_SHA256.txt`, verbatim copy beside it) before this test is written or
run. Results go in section 5.
**Date:** 2026-09-29
**Depends on:** ADR-0004, whose G3 **failed its own non-vacuity condition** (10 of 100 circuits
truncated, 80 required). That FAIL stands in ADR-0004 §6 and is not re-run under new
parameters there.

## 1. Why

ADR-0004 G3 drew gates uniformly from 13 kinds, two of them entangling. At depth 60 on 10
qubits that is about 9 entangling gates, and the bond rarely exceeded χ = 4. In the 10 circuits
that did truncate, the bound held: F ≥ F_bound in all 10, and min(F − F_bound) = −2.2e-15.
Ten circuits is not a test of a bound.

The bound matters exactly where truncation is heavy. The registered bound F_bound = Π(1 − ε_k)
is the standard estimate (Zhou, Stoudenmire & Waintal, PRX 2020), and it is **not a proven lower
bound**. The proven one is the Fubini–Study bound F ≥ cos²(Σ θ_k), with θ_k = arccos √(1 − ε_k),
valid while Σ θ_k < π/2. This test asks whether QGPU's reported number can be trusted where it
is used.

## 2. Test

- **Circuits.** 100 brickwork circuits (seed 20260931) on n = 10 qubits, 12 layers. Each layer
  applies one random gate to every qubit, drawn from the 8 fixed gates and Rx/Ry/Rz with
  θ uniform in [0, 2π). Then it applies CX on every pair (i, i+1): i even on even layers, i odd
  on odd layers.
- **Execution.** As in ADR-0004: each circuit through the ISA on `qmps 10, 4` and on
  `qalloc 10`, with amplitudes and F_bound read by `qamp` and `qtrunc`.
  F = \|⟨ψ_exact\|ψ_mps⟩\|² / ⟨ψ_mps\|ψ_mps⟩.

## 3. Pass and fail

- **H1 (non-vacuity, must hold):** at least 95 of 100 circuits truncate (F_bound < 1).
- **H2 (the reported bound, the question):** F ≥ F_bound − 1e-9 in every circuit.
- **H3 (the proven bound, must hold, or the code is wrong):** F ≥ F_rigorous − 1e-9 in every
  circuit where Σθ < π/2.

**If H2 fails,** QGPU stops reporting Π(1 − ε) as a bound. `qtrunc` switches to the proven bound
in a follow-up, and the count of violations is published. **If H3 fails,** the truncation
bookkeeping is wrong, and QGPU claims nothing until it is fixed.

## 4. Not claimed

Nothing about other circuit families or larger n. One family, stated.

## 5. Results

(filled after the run)
