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

## Not claimed

- No quantum speed-up, and no quantum hardware.
- No GPU throughput for MTLB: it is an emulator with a cost model.
- QRAM and U-QPU are parked, not dropped.
