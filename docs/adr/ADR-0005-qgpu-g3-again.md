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

## 5. Results (2026-09-29; sections 1–4 unchanged)

| | result |
|---|---|
| **H1** non-vacuity | **PASS.** 100 / 100 truncated. F ranged from 0.055 to about 0.53 |
| **H2** the reported product bound | **FAIL.** F < Π(1 − ε) − 1e-9 in **53 of 100** circuits, by up to 0.054 (circuit records in `experiments/qgpu/adr5_result.txt`) |
| **H3** the proven bound | **PASS.** It applies in 13 circuits (Σθ < π/2), and F ≥ cos²(Σθ) in all 13, min gap +5.3e-15. The bookkeeping is right, and the proven bound is tight |

**What it means.** The standard estimate holds under light truncation: in ADR-0004's 10
truncated circuits it never over-claimed. Under heavy truncation it over-claims fidelity about
half the time. A reported fidelity that is wrong half the time where it matters is not a bound.

**What changed, as pre-registered in §3:**
- `qtrunc` reports the proven bound. When Σθ ≥ π/2 it reports 0: under that much truncation,
  nothing is guaranteed, and the engine says so.
- `qfest` reports the product as an estimate, named as one.
- The test locks H2's 53 as a regression, so a change in the numerics shows up, and asserts
  that the new `qtrunc` over-claims in 0 of 100.

Reference: Zhou, Stoudenmire & Waintal, "What limits the simulation of quantum computers?",
PRX 10, 041038 (2020), for the product estimate. The proven bound is the triangle inequality of
the Fubini–Study angle.
