# ADR-0002: QGPU, the rung in focus (LYTH + tensors)

**Status:** Proposed. The front for 2026-10-05; nothing here is built beyond what is cited.
**Date:** 2026-09-27
**Deciders:** Antonio
**Amends:** ADR-0001, which listed QGPU as a research question. QRAM stays one.

## Context

ADR-0001 laid the programme out as CPU → TPU → QPU, with QGPU and QRAM as research
questions. Two facts change the priority:

- **QPU stays simulated whatever we do.** There is no quantum hardware here, so U-QPU would
  be a state-vector simulator whichever rung it sits on. It stays planned, not in focus.
- **The QGPU work already runs.** The pieces that a "GPU whose native work is quantum states
  and tensors" needs are built and measured, in three repositories.

## Decision

### 1. What "QGPU" means here

**QGPU is classical silicon whose native workload is quantum states and tensor networks,
programmed through LYTH.** Two targets, one source:
- the RTX 5060 Ti (sm_120), through LYTH → PTX;
- MTLB, through LYTH → `lyth-uasm`, with its tensor-network unit.

It claims no quantum hardware. "Q" names the workload, not the physics of the chip.

### 2. The stack it belongs to: motor, compiler, compressor, QGPU

| Piece | Role | One line |
|---|---|---|
| **QuBLAR** | motor | Ising photonic engine: infers what cannot be seen, and returns exists / does not exist / undecided with a measured certainty |
| **LYTH** | compiler | a kernel that cannot say what it costs does not compile; movement is declared and intensity is checked |
| **Blaze** | compressor | dual-band tensor compressor (Blaze ADR-0005): compressed / declined / undecided, with a certificate |
| **MTLB (QGPU)** | the machine | executes states and tensors as native operands, compiled by LYTH |

### 3. What already exists (facts, not plans)

| Fact | Where |
|---|---|
| A Hadamard gate runs bit-exact from one `.lyth` source on PTX and on MTLB (top qubit) | LYTH `examples/hadamard.lyth`, test `a_quantum_gate_is_expressible_and_bit_exact` |
| Every single-qubit gate on every qubit, without a copy: the declared split | LYTH ADR-0028, **Proposed**, untracked since 2026-09-18 |
| MPS contraction as an instruction: `ZIPPER2`, int8 cores with per-bond scales and an f32 accumulator, fidelity 0.999962 (mean), 8 of 8 mutations caught | MTLB README, "Tensor-network unit" |
| A 256-site MPS chain measured at 69.57 flops/instr, 82 % of ceiling | MTLB `programs/mps_chain.uasm` |
| **Blaze's int8 cores are the operand format `ZIPPER2` consumes**: int8 codes with per-bond scales, error composed and measured | Blaze Phase 8; MTLB tensor-network unit |
| QuBLAR's ghost-bit posteriors compress to TT rank 1–2 in Blaze, so they are already within the χ = 2 that MTLB holds | QuBLAR RESULTS-phase6; Blaze README |

### 4. The rigour rule: no piece fails another silently

What the stack is judged by is whether it holds together under QuBLAR, so each hand-off is a
gate:

- **QuBLAR → Blaze.** QuBLAR uses Blaze's output only on a *compressed* verdict whose bound is
  inside QuBLAR's tolerance. *Declined* or *undecided* falls back to the exact path, and the
  result says so. Blaze can never make QuBLAR wrong; at worst it makes QuBLAR slower.
- **LYTH → QGPU.** A kernel runs only if its declared intensity matches the one derived from
  its body, and if it is bit-exact against the host oracle on both targets.
- **Blaze ↔ QGPU.** A TT or MPS contraction written in LYTH must reproduce Blaze's Python
  reference to a stated tolerance, on PTX and on MTLB.
- **Every limit is filed.** For example: χ = 4 does not fit a 256-bit word, which is the
  architecture's ceiling, recorded in MTLB.

### 5. What to attack on 2026-10-05 (order does not matter)

1. **LYTH ADR-0028**: decide it and commit it. It opens every single-qubit gate and is the
   first real QGPU step on the gate side.
2. **The first joint gate of the four**: QuBLAR ghost bits → Blaze int8 TT → a LYTH
   contraction on PTX and on MTLB (`ZIPPER2`) → marginals equal to Blaze's reference within
   tolerance, or a declined verdict that QuBLAR handles.
3. **Two-qubit gates**: the nested split, as its own LYTH ADR.
4. **Blaze ADR-0005**: the verdict API, so the gate in item 2 has something to check.
5. **The Hadamard gate on the ghost bits** (§8): H^⊗n on QuBLAR's ROI posterior, core by core in
   Blaze's TT, run as a LYTH kernel on both targets. Walsh correlations equal to the dense
   check, and ranks unchanged.

### 6. The pattern: each piece has one refusal

| Piece | Its refusal |
|---|---|
| QuBLAR (motor) | **What the data do not pay for does not exist.** The evidence budget: at 2²⁵ muons it declined the void, because the data paid 303 nats against the prior's 421 |
| LYTH (compiler) | **A kernel that cannot say what it costs does not compile** |
| Blaze (compressor) | **If the contract does not pass, it does not compress**: it declines, or it is undecided, and it says so |
| QGPU (machine) | **What is not bit-exact against the oracle does not execute** |

### 7. Where QGPU comes from: QuBLAR × LYTH

QGPU is not designed from hardware down. It is designed **from the workload up**:
- QGPU's operations are the kernels QuBLAR needs, written in LYTH, measured, and only then
  mapped to MTLB instructions and to PTX;
- the kernels are the annealing sweep (local field and flip), the ROI marginal contraction,
  and the Hadamard layer of §8;
- LYTH verifies QuBLAR's data path;
- Blaze is the output: it translates and compresses ghost bits into every form a consumer
  needs.

```
QuBLAR ──(ghost bits)──▶ LYTH (verifies the kernels) ──▶ QGPU (runs them: PTX | MTLB)
                                                               │
                                                               ▼
                                               Blaze: translates and compresses (the output)
```

### 8. "=" as gates: QuBLAR = LYTH = MTLB(QGPU) = Blaze

The connections between the pieces are read as gates. This is a precise statement, not a
metaphor, because the hand-off object is a tensor over bits (QuBLAR's posterior, stored as a
TT by Blaze), and two facts hold for it. Both were checked numerically on 2026-09-27 with a
6-bit posterior.

1. **A Hadamard layer gives the ghost bits' correlations.** For a distribution p over n bits,
   (H^⊗n p)_s = 2^(−n/2) · E[(−1)^(s·x)]: the Walsh spectrum, which is every parity
   correlation between groups of voxels. It answers "which hidden voxels are uncertain
   *together*".
2. **The layer is free in the compressed form.** A gate on one mode rewrites one TT core and
   leaves every rank unchanged. So H^⊗n is applied core by core, O(n·χ²), on Blaze's cores
   and on `ZIPPER2` operands, and never on the 2ⁿ vector. LYTH already expresses the gate:
   `hadamard.lyth` is bit-exact on PTX and MTLB.

**Correlation is the entanglement analogue.** The TT rank at a cut is the Schmidt rank of that
bipartition. In the check, two correlated ghost bits gave rank 2 at exactly their cut and
rank 1 everywhere else, and the Hadamard layer preserved that. On the quantum band, the
q-sample state |ψ⟩ = Σ √p(x) |x⟩, built from QuBLAR's posterior, is a real quantum state:
- its entanglement across a cut is the correlation between those voxel groups;
- Blaze's MPS → circuit synthesis (Phase 5, fidelity 1.0) turns it into a state-preparation
  circuit;
- measuring that circuit samples QuBLAR's branches.

The ranks of √p are measured, never assumed equal to those of p.

The gates, one per "=":

| "=" | Gate | Check |
|---|---|---|
| QuBLAR = LYTH | contract gate: QuBLAR's kernels in LYTH, with cost declared | intensity matches, bit-exact against the host oracle |
| LYTH = QGPU | compile gate: one source, two targets | bit-exact on PTX and MTLB |
| QGPU = Blaze | tensor gate: int8 TT cores as `ZIPPER2` operands; the Hadamard layer applied core by core | matches Blaze's Python reference to the stated tolerance |
| Blaze = output | dual-band gate: marginals, Walsh correlations, or the q-sample circuit | the verdict and its certificate (Blaze ADR-0005) |

### 9. If QRAM is ever taken up

QRAM's job is to load classical data into superposition. In this system that job already has
a place: **the Blaze = output gate**. A state-preparation circuit from an MPS loads a
structured state at a depth set by its TT rank, which is the known alternative to a
bucket-brigade memory for data with structure. QRAM would therefore enter as a fifth "=", at
the loader, with the same kind of gate. It stays parked.

## Not claimed

- No quantum speed-up, and no quantum hardware.
- No GPU throughput for MTLB: it is an emulator with a cost model.
- QRAM and U-QPU are parked, not dropped.
