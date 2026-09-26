# ADR-0001 — Unibit is a processor programme: CPU → TPU → QPU

**Status:** accepted (direction). Each milestone gets its own ADR before code.
**Date:** 2026-09-26
**Deciders:** Antonio

## Context

Unibit today is a 256-bit ISA with a zero-dependency Rust emulator, a two-pass assembler,
an object format, a disassembler and a cost model. It has 73 tests, clippy is clean, and
known limits are marked `[KNOWN_LIMIT]`. A register is one 256-bit word, and the
instruction decides what it means. It already has a post-quantum lattice unit and a
tensor-network unit.

The owner's correction: **an emulator was never the goal. A processor is.** The emulator is
the instrument, the way VENTUS's digital twin is the instrument for an aircraft that has to
fly. So Unibit becomes a programme with milestones that close or do not, scaling across
processor generations, simulated and bare-metal throughout.

## Decision

### Milestones (each one closes against its own falsifiers)

| Milestone | What closes it | Where the instrument is |
|---|---|---|
| **U-CPU**: the scalar and vector core | ISA, emulator and cost model, measured on three real workloads | **exists**: the current repository |
| **U-TPU**: a matrix unit | a systolic MAC array defined as ISA extensions, with a cost model for dataflow and memory movement; numerics checked against the host GPU (cuBLAS on sm_120) to a stated tolerance; LYTH as a candidate front-end compiler | the next ADR |
| **U-QPU**: a quantum coprocessor, simulated | quantum instructions dispatched from the ISA to a state-vector simulator on CUDA (about 30 qubits on 16 GB); gates checked against closed forms; a declared noise model; QUBO and annealing as a coprocessor mode, sharing QuBLAR's contract | after U-TPU |

**Research questions, not milestones:** QGPU and QRAM. QRAM is a real research topic
(bucket-brigade proposals); a QGPU is not an architecture anyone builds. They are written up
as analysis documents with literature, and they claim no hardware.

### Rules carried over from VENTUS

- **Every number is cited, measured, or marked** `[TO CITE]` or `[KNOWN_LIMIT]`.
- **A milestone is a claim that can fail.** "Runs" is not "closes". Each milestone states the
  measurement that would falsify it.
- **Simulated means simulated.** There is no RTL and no silicon. Performance is a cost
  model, calibrated where the host allows, and is never quoted as hardware speed.

### Languages

The ISA, emulator and tools are Rust with zero dependencies, as today. The simulators for
U-TPU numerics and U-QPU state vectors are CUDA on sm_120. The language follows the physics,
as it does in VENTUS.

## Consequences

- The README says where Unibit is going; the current core is U-CPU.
- U-TPU gets its own ADR before any code: dataflow, the cost model, and what the GPU
  cross-check can and cannot prove.
