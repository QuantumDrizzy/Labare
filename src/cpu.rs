// ============================================================================
// Unibit — CPU Core & Pipeline Engine
// ============================================================================
//
// 256-bit Post-Quantum AI-Native Processor Implementation.
// Features:
//   - 32 × 256-bit Architectural Registers (x0 = hardwired zero)
//   - Branch History Table (BHT) + Branch Target Buffer (BTB)
//   - Thermodynamic Landauer Bit-Erasure & Entropy Tracking
//   - Unified Syscall I/O Dispatcher
//   - Detailed Microarchitectural & Physical Metrics Report
//
// ============================================================================

use std::io::{self, Write};
use crate::alu::*;
use crate::isa::*;
use crate::memory::Memory;

// ─── Execution State ─────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CpuStatus {
    Ready,
    Running,
    Halted,
    Trapped(u64),
    Error,
}

// ─── Branch Predictor ────────────────────────────────────────────────────────

pub struct BranchPredictor {
    /// 2-bit saturating counters (0=Strongly Not Taken, 3=Strongly Taken)
    bht: [u8; 512],
    /// Branch Target Buffer mapping PC -> Predicted Target
    btb: [u64; 512],
}

impl Default for BranchPredictor {
    fn default() -> Self {
        Self::new()
    }
}

impl BranchPredictor {
    pub fn new() -> Self {
        Self {
            bht: [2; 512], // Init weakly taken (loops often jump)
            btb: [0; 512],
        }
    }

    #[inline]
    fn hash(pc: u64) -> usize {
        ((pc ^ (pc >> 4)) as usize) & 511
    }

    pub fn predict(&self, pc: u64) -> (bool, u64) {
        let idx = Self::hash(pc);
        let taken = self.bht[idx] >= 2;
        let target = self.btb[idx];
        (taken, target)
    }

    pub fn update(&mut self, pc: u64, actual_taken: bool, actual_target: u64) {
        let idx = Self::hash(pc);
        if actual_taken {
            if self.bht[idx] < 3 {
                self.bht[idx] += 1;
            }
            self.btb[idx] = actual_target;
        } else if self.bht[idx] > 0 {
            self.bht[idx] -= 1;
        }
    }
}

// ─── Microarchitectural & Physical Metrics ───────────────────────────────────

#[derive(Clone, Debug, Default)]
pub struct CpuMetrics {
    pub cycles: u64,
    pub instructions_retired: u64,
    pub branch_count: u64,
    pub branch_mispredictions: u64,
    pub pipeline_stalls: u64,
    pub data_forwards: u64,
    pub bit_erasures: u64,
    pub vector_ops: u64,
    /// Packed single-precision instructions retired. Counted apart from `vector_ops`
    /// deliberately: the cost model exists to tell eight f32 multiplies from eight integer
    /// ones, and one bucket for both is a cost model about a machine that does not exist.
    pub float_ops: u64,
    pub complex_ops: u64,
    pub lattice_ops: u64,
    pub tensor_ops: u64,
    pub info_ops: u64,
    /// Quantum coprocessor instructions (ADR-0003). One dispatch cycle each:
    /// a count, not a timing model (durations against coherence are phase 2).
    pub quantum_ops: u64,
}

impl CpuMetrics {
    pub fn ipc(&self) -> f64 {
        if self.cycles == 0 { 0.0 } else { self.instructions_retired as f64 / self.cycles as f64 }
    }

    pub fn branch_accuracy(&self) -> f64 {
        if self.branch_count == 0 {
            100.0
        } else {
            let correct = self.branch_count.saturating_sub(self.branch_mispredictions);
            (correct as f64 / self.branch_count as f64) * 100.0
        }
    }

    /// Minimum dissipated work for `bits` irreversibly destroyed, at temp_k.
    pub fn landauer_energy_joules_for(bits: u64, temp_k: f64) -> f64 {
        (bits as f64) * landauer_energy(temp_k)
    }
}

// ─── CPU Structure ───────────────────────────────────────────────────────────

pub struct Cpu {
    pub regs: [Reg256; 32],
    pub pc: u64,
    pub memory: Memory,
    pub status: CpuStatus,
    pub metrics: CpuMetrics,
    pub branch_predictor: BranchPredictor,
    pub temp_k: f64,
    pub rng_seed: u64,
    pub trace: bool,
    pub stdout_buffer: Vec<u8>,
    pub capture_output: bool,
    /// When false the machine predicts statically not-taken, which is the
    /// baseline the BHT/BTB is measured against. See `unibit bench`.
    pub predictor_enabled: bool,
    /// The quantum coprocessor's register, once `qalloc` has run (ADR-0003).
    pub qpu: Option<crate::qpu::Backend>,
    /// Seed for the next register's measurement RNG (`qseed`).
    pub qpu_seed: u64,
}

impl Cpu {
    pub fn new(mem_size: usize) -> Self {
        let mut cpu = Self {
            regs: [Reg256::ZERO; 32],
            pc: 0,
            memory: Memory::new(mem_size),
            status: CpuStatus::Ready,
            metrics: CpuMetrics::default(),
            branch_predictor: BranchPredictor::new(),
            temp_k: 300.0, // 300 Kelvin (ambient room temperature)
            rng_seed: 0x1337_CAFE_BABE_F00D,
            trace: false,
            stdout_buffer: Vec::new(),
            capture_output: false,
            predictor_enabled: true,
            qpu: None,
            qpu_seed: 0,
        };
        // Initialize Stack Pointer at high memory address
        cpu.regs[REG_SP as usize] = Reg256::from_u64((mem_size - 256) as u64);
        cpu
    }

    /// The quantum register, or an error if no `qalloc` has run.
    fn qpu_mut(&mut self) -> Result<&mut crate::qpu::Backend, String> {
        self.qpu.as_mut().ok_or_else(|| "quantum instruction before qalloc".to_string())
    }

    /// Read register value (x0 always returns ZERO)
    #[inline]
    pub fn get_reg(&self, r: u8) -> Reg256 {
        if r == 0 {
            Reg256::ZERO
        } else {
            self.regs[(r & 31) as usize]
        }
    }

    /// Write a register, accumulating the bits destroyed for the Landauer
    /// tracker.
    ///
    /// [KNOWN_LIMIT] This counts *bits overwritten with a different value*
    /// (Hamming distance old vs new). That is a cost model, not a physical
    /// measurement: it does not track logical reversibility, and it ignores
    /// the operand bits an irreversible ALU gate consumes. Memory stores are
    /// counted the same way in `Memory::overwrite`.
    #[inline]
    pub fn set_reg(&mut self, r: u8, val: Reg256) {
        if r == 0 {
            return; // x0 hardwired to zero
        }
        let idx = (r & 31) as usize;
        // Count overwritten/erased bits for physical thermodynamics
        let flipped = self.regs[idx].hamming_distance(&val);
        self.metrics.bit_erasures += flipped as u64;
        self.regs[idx] = val;
    }

    /// Reset PC and execution state
    pub fn reset(&mut self, entry_point: u64) {
        self.pc = entry_point;
        self.status = CpuStatus::Ready;
    }

    // ─── Instruction Execution ───────────────────────────────────────────────

    pub fn execute_instruction(&mut self, inst: &Instruction) -> Result<(), String> {
        self.metrics.instructions_retired += 1;
        self.metrics.cycles += 1; // Base 1 cycle per instruction in ideal pipeline

        if self.trace {
            println!("  [PC=0x{:04x}] {:<18} | a0={} ({:#x}) | t0={}",
                self.pc, format!("{:?}", inst),
                self.get_reg(REG_A0).as_i64(),
                self.get_reg(REG_A0).as_u64(),
                self.get_reg(REG_T0).as_i64()
            );
        }

        let next_pc = self.pc + 1;

        match inst {
            // ─── Scalar ALU ──────────────────────────────────────────────────
            Instruction::Add { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::add(a, b)));
                self.pc = next_pc;
            }
            Instruction::Sub { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::sub(a, b)));
                self.pc = next_pc;
            }
            Instruction::Mul { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::mul(a, b)));
                self.pc = next_pc;
            }
            Instruction::MulH { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::mulh(a, b)));
                self.pc = next_pc;
            }
            Instruction::Div { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::div(a, b)));
                self.metrics.cycles += 3; // Division latency
                self.pc = next_pc;
            }
            Instruction::Rem { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::rem(a, b)));
                self.metrics.cycles += 3;
                self.pc = next_pc;
            }
            Instruction::And { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::and(a, b)));
                self.pc = next_pc;
            }
            Instruction::Or { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::or(a, b)));
                self.pc = next_pc;
            }
            Instruction::Xor { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::xor(a, b)));
                self.pc = next_pc;
            }
            Instruction::Sll { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::sll(a, b)));
                self.pc = next_pc;
            }
            Instruction::Srl { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::srl(a, b)));
                self.pc = next_pc;
            }
            Instruction::Sra { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::sra(a, b)));
                self.pc = next_pc;
            }
            Instruction::Slt { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::slt(a, b)));
                self.pc = next_pc;
            }
            Instruction::Sltu { rd, rs1, rs2 } => {
                let a = self.get_reg(*rs1).as_u64();
                let b = self.get_reg(*rs2).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::sltu(a, b)));
                self.pc = next_pc;
            }

            // ─── Scalar Immediate ────────────────────────────────────────────
            Instruction::Addi { rd, rs1, imm } => {
                let a = self.get_reg(*rs1).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::add(a, *imm as u64)));
                self.pc = next_pc;
            }
            Instruction::Andi { rd, rs1, imm } => {
                let a = self.get_reg(*rs1).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::and(a, *imm as u64)));
                self.pc = next_pc;
            }
            Instruction::Ori { rd, rs1, imm } => {
                let a = self.get_reg(*rs1).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::or(a, *imm as u64)));
                self.pc = next_pc;
            }
            Instruction::Xori { rd, rs1, imm } => {
                let a = self.get_reg(*rs1).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::xor(a, *imm as u64)));
                self.pc = next_pc;
            }
            Instruction::Slli { rd, rs1, imm } => {
                let a = self.get_reg(*rs1).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::sll(a, *imm as u64)));
                self.pc = next_pc;
            }
            Instruction::Srli { rd, rs1, imm } => {
                let a = self.get_reg(*rs1).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::srl(a, *imm as u64)));
                self.pc = next_pc;
            }
            Instruction::Srai { rd, rs1, imm } => {
                let a = self.get_reg(*rs1).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::sra(a, *imm as u64)));
                self.pc = next_pc;
            }
            Instruction::Slti { rd, rs1, imm } => {
                let a = self.get_reg(*rs1).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::slt(a, *imm as u64)));
                self.pc = next_pc;
            }
            Instruction::Sltiu { rd, rs1, imm } => {
                let a = self.get_reg(*rs1).as_u64();
                self.set_reg(*rd, Reg256::from_u64(ScalarAlu::sltu(a, *imm as u64)));
                self.pc = next_pc;
            }
            Instruction::Lui { rd, imm } => {
                let val = (*imm as u64) << 12;
                self.set_reg(*rd, Reg256::from_u64(val));
                self.pc = next_pc;
            }

            // ─── Memory Operations ───────────────────────────────────────────
            Instruction::Ld { rd, rs1, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let val = self.memory.load_u64(addr)?;
                self.set_reg(*rd, Reg256::from_u64(val));
                self.pc = next_pc;
            }
            Instruction::Lw { rd, rs1, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let val = self.memory.load_i32(addr)? as i64 as u64;
                self.set_reg(*rd, Reg256::from_u64(val));
                self.pc = next_pc;
            }
            Instruction::Lh { rd, rs1, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let val = self.memory.load_i16(addr)? as i64 as u64;
                self.set_reg(*rd, Reg256::from_u64(val));
                self.pc = next_pc;
            }
            Instruction::Lb { rd, rs1, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let val = self.memory.load_i8(addr)? as i64 as u64;
                self.set_reg(*rd, Reg256::from_u64(val));
                self.pc = next_pc;
            }
            Instruction::Lbu { rd, rs1, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let val = self.memory.load_u8(addr)? as u64;
                self.set_reg(*rd, Reg256::from_u64(val));
                self.pc = next_pc;
            }
            Instruction::Sd { rs1, rs2, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let val = self.get_reg(*rs2).as_u64();
                self.memory.store_u64(addr, val)?;
                self.pc = next_pc;
            }
            Instruction::Sw { rs1, rs2, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let val = self.get_reg(*rs2).as_u64() as u32;
                self.memory.store_u32(addr, val)?;
                self.pc = next_pc;
            }
            Instruction::Sh { rs1, rs2, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let val = self.get_reg(*rs2).as_u64() as u16;
                self.memory.store_u16(addr, val)?;
                self.pc = next_pc;
            }
            Instruction::Sb { rs1, rs2, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let val = self.get_reg(*rs2).as_u64() as u8;
                self.memory.store_u8(addr, val)?;
                self.pc = next_pc;
            }
            Instruction::Lq { rd, rs1, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let lanes = self.memory.load_256(addr)?;
                self.set_reg(*rd, Reg256 { lanes });
                self.pc = next_pc;
            }
            Instruction::Sq { rs1, rs2, offset } => {
                let base = self.get_reg(*rs1).as_u64();
                let addr = base.wrapping_add(*offset as u64);
                let reg = self.get_reg(*rs2);
                self.memory.store_256(addr, &reg.lanes)?;
                self.pc = next_pc;
            }

            // ─── Branch Instructions with BHT Simulation ─────────────────────
            Instruction::Beq { rs1, rs2, offset } => {
                let cond = self.get_reg(*rs1) == self.get_reg(*rs2);
                self.handle_branch(cond, *offset);
            }
            Instruction::Bne { rs1, rs2, offset } => {
                let cond = self.get_reg(*rs1) != self.get_reg(*rs2);
                self.handle_branch(cond, *offset);
            }
            Instruction::Blt { rs1, rs2, offset } => {
                let cond = self.get_reg(*rs1).as_i64() < self.get_reg(*rs2).as_i64();
                self.handle_branch(cond, *offset);
            }
            Instruction::Bge { rs1, rs2, offset } => {
                let cond = self.get_reg(*rs1).as_i64() >= self.get_reg(*rs2).as_i64();
                self.handle_branch(cond, *offset);
            }
            Instruction::Bltu { rs1, rs2, offset } => {
                let cond = self.get_reg(*rs1).as_u64() < self.get_reg(*rs2).as_u64();
                self.handle_branch(cond, *offset);
            }
            Instruction::Bgeu { rs1, rs2, offset } => {
                let cond = self.get_reg(*rs1).as_u64() >= self.get_reg(*rs2).as_u64();
                self.handle_branch(cond, *offset);
            }

            // ─── Jumps ───────────────────────────────────────────────────────
            Instruction::Jal { rd, offset } => {
                self.set_reg(*rd, Reg256::from_u64(next_pc));
                self.pc = (self.pc as i64 + *offset) as u64;
            }
            Instruction::Jalr { rd, rs1, offset } => {
                let target = self.get_reg(*rs1).as_u64().wrapping_add(*offset as u64);
                self.set_reg(*rd, Reg256::from_u64(next_pc));
                self.pc = target;
            }

            // ─── Vector Instructions ─────────────────────────────────────────
            // Packed single-precision. Counted as `float_ops` rather than `vector_ops`: the
            // cost model exists to tell these apart, and folding eight f32 multiplies into
            // the same bucket as eight integer ones is how a cost model stops describing the
            // machine it is about.
            Instruction::VfAdd { rd, rs1, rs2 } => {
                let res = FloatUnit::vfadd(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.float_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VfSub { rd, rs1, rs2 } => {
                let res = FloatUnit::vfsub(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.float_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VfMul { rd, rs1, rs2 } => {
                let res = FloatUnit::vfmul(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.float_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VfMa { rd, rs1, rs2 } => {
                // `rd` is the addend as well as the destination, which is what makes this an
                // accumulate rather than a three-operand multiply-add.
                let res = FloatUnit::vfma(
                    &self.get_reg(*rs1),
                    &self.get_reg(*rs2),
                    &self.get_reg(*rd),
                );
                self.set_reg(*rd, res);
                self.metrics.float_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VfMax { rd, rs1, rs2 } => {
                let res = FloatUnit::vfmax(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.float_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VfMin { rd, rs1, rs2 } => {
                let res = FloatUnit::vfmin(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.float_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VfReduce { rd, rs1 } => {
                let res = FloatUnit::vfreduce(&self.get_reg(*rs1));
                self.set_reg(*rd, res);
                self.metrics.float_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VAdd { rd, rs1, rs2, width } => {
                let res = VectorUnit::vadd(&self.get_reg(*rs1), &self.get_reg(*rs2), *width);
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VSub { rd, rs1, rs2, width } => {
                let res = VectorUnit::vsub(&self.get_reg(*rs1), &self.get_reg(*rs2), *width);
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VMul { rd, rs1, rs2, width } => {
                let res = VectorUnit::vmul(&self.get_reg(*rs1), &self.get_reg(*rs2), *width);
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VAnd { rd, rs1, rs2 } => {
                let res = VectorUnit::vand(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VOr { rd, rs1, rs2 } => {
                let res = VectorUnit::vor(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VXor { rd, rs1, rs2 } => {
                let res = VectorUnit::vxor(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VNot { rd, rs1 } => {
                let res = VectorUnit::vnot(&self.get_reg(*rs1));
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VDot { rd, rs1, rs2, width } => {
                let res = VectorUnit::vdot(&self.get_reg(*rs1), &self.get_reg(*rs2), *width);
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VSplat { rd, rs1, width } => {
                let res = VectorUnit::vsplat(&self.get_reg(*rs1), *width);
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }
            Instruction::VReduce { rd, rs1, width } => {
                let res = VectorUnit::vreduce(&self.get_reg(*rs1), *width);
                self.set_reg(*rd, res);
                self.metrics.vector_ops += 1;
                self.pc = next_pc;
            }

            // ─── Complex Number Instructions (Quantum) ───────────────────────
            Instruction::CAdd { rd, rs1, rs2 } => {
                let res = ComplexUnit::cadd(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.complex_ops += 1;
                self.pc = next_pc;
            }
            Instruction::CSub { rd, rs1, rs2 } => {
                let res = ComplexUnit::csub(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.complex_ops += 1;
                self.pc = next_pc;
            }
            Instruction::CMul { rd, rs1, rs2 } => {
                let res = ComplexUnit::cmul(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.complex_ops += 1;
                self.pc = next_pc;
            }
            Instruction::CConj { rd, rs1 } => {
                let res = ComplexUnit::cconj(&self.get_reg(*rs1));
                self.set_reg(*rd, res);
                self.metrics.complex_ops += 1;
                self.pc = next_pc;
            }
            Instruction::CNorm { rd, rs1 } => {
                let res = ComplexUnit::cnorm(&self.get_reg(*rs1));
                self.set_reg(*rd, res);
                self.metrics.complex_ops += 1;
                self.pc = next_pc;
            }
            Instruction::CMag { rd, rs1 } => {
                let res = ComplexUnit::cmag(&self.get_reg(*rs1));
                self.set_reg(*rd, res);
                self.metrics.complex_ops += 1;
                self.pc = next_pc;
            }

            // ─── Information Theory Instructions ─────────────────────────────
            Instruction::Entropy { rd, rs1 } => {
                let res = InfoUnit::entropy(&self.get_reg(*rs1));
                self.set_reg(*rd, res);
                self.metrics.info_ops += 1;
                self.pc = next_pc;
            }
            Instruction::Hamming { rd, rs1, rs2 } => {
                let res = InfoUnit::hamming(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.info_ops += 1;
                self.pc = next_pc;
            }
            Instruction::PopCnt { rd, rs1 } => {
                let res = InfoUnit::popcount(&self.get_reg(*rs1));
                self.set_reg(*rd, res);
                self.metrics.info_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QRand { rd } => {
                let res = InfoUnit::qrand(&mut self.rng_seed);
                self.set_reg(*rd, res);
                self.metrics.info_ops += 1;
                self.pc = next_pc;
            }

            // ─── Quantum coprocessor (U-QPU phase 1, ADR-0003) ───────────
            Instruction::QAlloc { n } => {
                let mut sv = crate::qpu::StateVector::new(*n as u32)?;
                sv.rng = crate::qpu::QRng::new(self.qpu_seed);
                self.qpu = Some(crate::qpu::Backend::StateVector(sv));
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QG1 { gate, q } => {
                self.qpu_mut()?.apply1(*q as u32, &gate.matrix())?;
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QRot { axis, q, rs } => {
                let theta = f64::from_bits(self.get_reg(*rs).as_u64());
                self.qpu_mut()?.apply1(*q as u32, &axis.matrix(theta))?;
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QCx { c, t } => {
                self.qpu_mut()?.cx(*c as u32, *t as u32)?;
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QCz { a, b } => {
                self.qpu_mut()?.cz(*a as u32, *b as u32)?;
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QMeas { rd, q } => {
                let m = self.qpu_mut()?.measure(*q as u32)?;
                self.set_reg(*rd, Reg256::from_u64(m as u64));
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QReset { q } => {
                self.qpu_mut()?.reset(*q as u32)?;
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QMps { n, chi } => {
                let mut m = crate::mps::Mps::new(*n as u32, *chi as u32)?;
                m.rng = crate::qpu::QRng::new(self.qpu_seed);
                self.qpu = Some(crate::qpu::Backend::Mps(m));
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QTrunc { rd } => {
                let f = self.qpu_mut()?.fid_bound();
                self.set_reg(*rd, Reg256::from_u64(f.to_bits()));
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QFEst { rd } => {
                let f = self.qpu_mut()?.fid_estimate();
                self.set_reg(*rd, Reg256::from_u64(f.to_bits()));
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QSeed { rs } => {
                self.qpu_seed = self.get_reg(*rs).as_u64();
                if let Some(b) = self.qpu.as_mut() {
                    b.set_rng(crate::qpu::QRng::new(self.qpu_seed));
                }
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }
            Instruction::QAmp { rd, rs } => {
                let index = self.get_reg(*rs).as_u64();
                let a = self.qpu_mut()?.amp(index)?;
                // Complex layout of this ISA: lanes[0:1] = (re, im) of z0.
                self.set_reg(*rd, Reg256 { lanes: [a.re.to_bits(), a.im.to_bits(), 0, 0] });
                self.metrics.quantum_ops += 1;
                self.pc = next_pc;
            }

            // ─── Tensor Network / MPS Accelerator ───────────────────────
            Instruction::Zipper { rd, rs1, rs2 } => {
                let res = TensorNetworkUnit::zipper_step(&self.get_reg(*rd), &self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.tensor_ops += 1;
                self.pc = next_pc;
            }
            Instruction::Zipper2 { rd, rs1, rs2 } => {
                let res = TensorNetworkUnit::zipper2_step(&self.get_reg(*rd), &self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.tensor_ops += 1;
                self.pc = next_pc;
            }
            Instruction::Trunc { rd, rs1, eps_bits } => {
                let res = TensorNetworkUnit::trunc(&self.get_reg(*rs1), *eps_bits);
                self.set_reg(*rd, res);
                self.metrics.tensor_ops += 1;
                self.pc = next_pc;
            }

            // ─── Post-Quantum Lattice Cryptography Instructions ──────────────
            Instruction::Ntt { rd, rs1 } => {
                let res = LatticeUnit::ntt(&self.get_reg(*rs1), KYBER_Q).ok_or_else(|| {
                    format!("ntt: modulus {} admits no negacyclic root of unity", KYBER_Q)
                })?;
                self.set_reg(*rd, res);
                self.metrics.lattice_ops += 1;
                self.pc = next_pc;
            }
            Instruction::InvNtt { rd, rs1 } => {
                let res = LatticeUnit::inv_ntt(&self.get_reg(*rs1), KYBER_Q).ok_or_else(|| {
                    format!("invntt: modulus {} admits no negacyclic root of unity", KYBER_Q)
                })?;
                self.set_reg(*rd, res);
                self.metrics.lattice_ops += 1;
                self.pc = next_pc;
            }
            Instruction::PolyMul { rd, rs1, rs2 } => {
                let res = LatticeUnit::poly_mul(&self.get_reg(*rs1), &self.get_reg(*rs2), KYBER_Q);
                self.set_reg(*rd, res);
                self.metrics.lattice_ops += 1;
                self.pc = next_pc;
            }
            Instruction::ModRed { rd, rs1, modulus } => {
                let res = LatticeUnit::mod_red(&self.get_reg(*rs1), *modulus);
                self.set_reg(*rd, res);
                self.metrics.lattice_ops += 1;
                self.pc = next_pc;
            }
            Instruction::PolyAdd { rd, rs1, rs2 } => {
                let res = LatticeUnit::poly_add(&self.get_reg(*rs1), &self.get_reg(*rs2), KYBER_Q);
                self.set_reg(*rd, res);
                self.metrics.lattice_ops += 1;
                self.pc = next_pc;
            }

            // ─── Tensor & Neural Activations ─────────────────────────────────
            Instruction::TAct { rd, rs1, func } => {
                let res = TensorUnit::activate(&self.get_reg(*rs1), *func);
                self.set_reg(*rd, res);
                self.metrics.tensor_ops += 1;
                self.pc = next_pc;
            }
            Instruction::TSoftmax { rd, rs1 } => {
                let res = TensorUnit::softmax(&self.get_reg(*rs1));
                self.set_reg(*rd, res);
                self.metrics.tensor_ops += 1;
                self.pc = next_pc;
            }
            Instruction::TMul { rd, rs1, rs2 } => {
                let res = TensorUnit::matmul2x2(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.tensor_ops += 1;
                self.pc = next_pc;
            }
            Instruction::TDot { rd, rs1, rs2 } => {
                let res = TensorUnit::dot_f64(&self.get_reg(*rs1), &self.get_reg(*rs2));
                self.set_reg(*rd, res);
                self.metrics.tensor_ops += 1;
                self.pc = next_pc;
            }

            // ─── Pseudo-Instructions & Moves ─────────────────────────────────
            Instruction::Mv { rd, rs1 } => {
                let val = self.get_reg(*rs1);
                self.set_reg(*rd, val);
                self.pc = next_pc;
            }
            Instruction::Li { rd, imm } => {
                self.set_reg(*rd, Reg256::from_i64(*imm));
                self.pc = next_pc;
            }

            // ─── System / CSR / Traps ────────────────────────────────────────
            Instruction::Ecall => {
                self.handle_syscall()?;
                self.pc = next_pc;
            }
            Instruction::Halt => {
                self.status = CpuStatus::Halted;
            }
            Instruction::Nop => {
                self.pc = next_pc;
            }
            Instruction::Fence => {
                self.pc = next_pc;
            }
            Instruction::CsrR { rd, csr } => {
                let val = match *csr {
                    csr::CYCLE => self.metrics.cycles,
                    csr::INSTRET => self.metrics.instructions_retired,
                    csr::ENTROPY_ACC => self.metrics.bit_erasures + self.memory.bit_erasures,
                    csr::TEMP_K => self.temp_k as u64,
                    csr::LANDAUER => (landauer_energy(self.temp_k) * 1e24) as u64, // in yoctojoules
                    _ => 0,
                };
                self.set_reg(*rd, Reg256::from_u64(val));
                self.pc = next_pc;
            }
            Instruction::CsrW { csr, rs1 } => {
                let val = self.get_reg(*rs1).as_u64();
                if *csr == csr::TEMP_K {
                    self.temp_k = val as f64;
                }
                self.pc = next_pc;
            }
        }

        Ok(())
    }

    #[inline]
    fn handle_branch(&mut self, condition: bool, offset: i64) {
        self.metrics.branch_count += 1;
        // Capture the branch's own PC: self.pc is about to be redirected, and
        // both predict and update must index the BHT/BTB by the branch site.
        let branch_pc = self.pc;
        let actual_target = (branch_pc as i64 + offset) as u64;
        let (predicted_taken, _) = if self.predictor_enabled {
            self.branch_predictor.predict(branch_pc)
        } else {
            (false, 0) // static not-taken: the baseline
        };

        if condition {
            self.pc = actual_target;
        } else {
            self.pc += 1;
        }

        if condition != predicted_taken {
            self.metrics.branch_mispredictions += 1;
            self.metrics.cycles += 3; // 3-cycle pipeline flush penalty
            self.metrics.pipeline_stalls += 3;
        }

        self.branch_predictor.update(branch_pc, condition, actual_target);
    }

    // ─── Syscall Dispatcher ──────────────────────────────────────────────────

    fn handle_syscall(&mut self) -> Result<(), String> {
        let syscall_num = self.get_reg(REG_A7).as_u64();
        match syscall_num {
            syscall::PRINT_INT => {
                let val = self.get_reg(REG_A0).as_i64();
                self.emit_output(format!("{}", val));
            }
            syscall::PRINT_CHAR => {
                let val = (self.get_reg(REG_A0).as_u64() & 0xFF) as u8 as char;
                self.emit_output(format!("{}", val));
            }
            syscall::PRINT_STR => {
                let addr = self.get_reg(REG_A0).as_u64();
                let len = self.get_reg(REG_A1).as_u64() as usize;
                let text = String::from_utf8_lossy(self.memory.read_bytes(addr, len)?).into_owned();
                self.emit_output(text);
            }
            syscall::PRINT_STRZ => {
                // NUL-terminated: no hand-counted length to drift out of sync
                // with the real byte length (a UTF-8 char spans 1-4 bytes).
                let addr = self.get_reg(REG_A0).as_u64();
                let text = String::from_utf8_lossy(self.memory.read_cstr(addr)?).into_owned();
                self.emit_output(text);
            }
            syscall::PRINT_F32 => {
                let val = f32::from_bits(self.get_reg(REG_A0).as_u64() as u32);
                // `{}`, not `{:.6}`: Rust prints the shortest string that round-trips, so a
                // program's output can be parsed back to the exact bits it computed.
                self.emit_output(format!("{}", val));
            }
            syscall::PRINT_HEX => {
                let val = self.get_reg(REG_A0).as_u64();
                self.emit_output(format!("0x{:016x}", val));
            }
            syscall::PRINT_F64 => {
                let val = f64::from_bits(self.get_reg(REG_A0).as_u64());
                self.emit_output(format!("{:.6}", val));
            }
            syscall::PRINT_REG256 => {
                let r_idx = (self.get_reg(REG_A0).as_u64() & 31) as u8;
                let reg = self.get_reg(r_idx);
                self.emit_output(format!("[x{:<2} = {:?}]", r_idx, reg));
            }
            syscall::EXIT => {
                let code = self.get_reg(REG_A0).as_u64();
                self.status = CpuStatus::Trapped(code);
            }
            _ => {
                return Err(format!("unknown syscall number: {}", syscall_num));
            }
        }
        Ok(())
    }

    fn emit_output(&mut self, text: String) {
        if self.capture_output {
            self.stdout_buffer.extend_from_slice(text.as_bytes());
        } else {
            print!("{}", text);
            let _ = io::stdout().flush();
        }
    }

    // ─── Main Program Run Loop ───────────────────────────────────────────────

    pub fn run_program(&mut self, instructions: &[Instruction], max_cycles: u64) -> Result<(), String> {
        self.status = CpuStatus::Running;
        while self.status == CpuStatus::Running {
            if self.pc as usize >= instructions.len() {
                self.status = CpuStatus::Halted;
                break;
            }
            if self.metrics.cycles >= max_cycles {
                return Err(format!("execution timed out after {} cycles", max_cycles));
            }
            let inst = &instructions[self.pc as usize];
            self.execute_instruction(inst)?;
        }
        Ok(())
    }

    // ─── Formatted Microarchitectural & Physical Report ──────────────────────

    pub fn print_report(&self) {
        let m = &self.metrics;
        // Whole-machine erasure count: register writes plus memory stores.
        let total_erasures = m.bit_erasures + self.memory.bit_erasures;
        let landauer_ej = CpuMetrics::landauer_energy_joules_for(total_erasures, self.temp_k);
        let landauer_floor = landauer_energy(self.temp_k);

        println!("\n╔══════════════════════════════════════════════════════════════════════════════════╗");
        println!("║                      Unibit PROCESSOR EXECUTION REPORT                        ║");
        println!("╠══════════════════════════════════════════════════════════════════════════════════╣");
        println!("║  ┌─ Microarchitecture & Pipeline ─────────────────────────────────────────────┐  ║");
        println!("║  │ Total Instructions Retired:  {:<14} Cycles: {:<20}│  ║", m.instructions_retired, m.cycles);
        println!("║  │ IPC (Instructions / Cycle):  {:<14.4} Stalls: {:<20}│  ║", m.ipc(), m.pipeline_stalls);
        println!("║  │ Branch Predictor (BHT/BTB):  {:<6} branches ({:<5.2}% accuracy, {:<3} miss)    │  ║",
            m.branch_count, m.branch_accuracy(), m.branch_mispredictions);
        println!("║  └────────────────────────────────────────────────────────────────────────────┘  ║");
        println!("║                                                                                  ║");
        println!("║  ┌─ Post-Quantum & Domain-Specific Units ─────────────────────────────────────┐  ║");
        println!("║  │ Vector SIMD (256-bit):       {:<14} Complex (Quantum): {:<14}│  ║", m.vector_ops, m.complex_ops);
        println!("║  │ Packed f32 (8 lanes):        {:<14} Flops (f32 lanes): {:<14}│  ║", m.float_ops, m.float_ops * 8);
        println!("║  │ Lattice PQC (NTT/Poly):      {:<14} Neural Tensor:     {:<14}│  ║", m.lattice_ops, m.tensor_ops);
        println!("║  │ Information Theory (Entropy):{:<14} Memory Reads/Writes:{:<5}/{:<6}│  ║", m.info_ops, self.memory.reads, self.memory.writes);
        println!("║  └────────────────────────────────────────────────────────────────────────────┘  ║");
        println!("║                                                                                  ║");
        println!("║  ┌─ Landauer Thermodynamic Dissipation (Physics Engine) ──────────────────────┐  ║");
        println!("║  │ Operating Temperature:       {:<7.2} K (Ambient / Dilution Ref configurable) │  ║", self.temp_k);
        println!("║  │ Bits Destroyed (reg + mem):  {:<14} {:>10} reg / {:<10} mem  │  ║",
            total_erasures, m.bit_erasures, self.memory.bit_erasures);
        println!("║  │ Landauer Energy Floor (1b):  {:<14.5e} Joules (k_B·T·ln2)                  │  ║", landauer_floor);
        println!("║  │ Min Thermodynamic Work:      {:<14.5e} Joules ({:<7.2} zeptoJoules)     │  ║",
            landauer_ej, landauer_ej * 1e21);
        println!("║  └────────────────────────────────────────────────────────────────────────────┘  ║");
        println!("╚══════════════════════════════════════════════════════════════════════════════════╝\n");
    }
}
