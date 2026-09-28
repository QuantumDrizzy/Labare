// ============================================================================
// Unibit — Instruction Set Architecture
// ============================================================================
//
// A novel 256-bit post-quantum AI-native processor ISA.
//
// What makes this ISA different from every other RISC ISA:
//
//   1. 256-bit registers with SEMANTIC access modes (scalar, vector, complex,
//      polynomial) — the hardware understands data TYPES, not just widths
//   2. Information-theoretic instructions (ENTROPY, HAMMING, POPCNT)
//   3. First-class complex number arithmetic (quantum-native)
//   4. Post-quantum lattice crypto primitives (NTT, POLYMUL)
//   5. Reversible execution mode (Landauer-aware)
//
// ============================================================================

use std::fmt;

use crate::qpu::{QAxis, QGate};

// ─── Physical Constants (for energy modeling) ────────────────────────────────

/// Boltzmann constant (J/K)
pub const K_B: f64 = 1.380_649e-23;
/// ln(2)
pub const LN2: f64 = std::f64::consts::LN_2;

/// Landauer limit at temperature T: minimum energy to erase 1 bit
#[inline]
pub fn landauer_energy(temp_k: f64) -> f64 {
    K_B * temp_k * LN2
}

// ─── 256-bit Register ────────────────────────────────────────────────────────

/// A 256-bit register composed of four 64-bit lanes.
///
/// This is the core data type of Unibit. Each register can be accessed as:
///   - Scalar:     1 × 64-bit   (lane 0, default for scalar ops)
///   - DWord:      4 × 64-bit   (vector mode)
///   - Word:       8 × 32-bit   (AI inference, fp32)
///   - Half:      16 × 16-bit   (AI inference, fp16/bf16)
///   - Byte:      32 × 8-bit    (quantized AI, int8)
///   - Complex:    2 × 128-bit  (quantum state vectors)
///   - Poly:       1 × 256-bit  (post-quantum polynomial coefficients)
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub struct Reg256 {
    pub lanes: [u64; 4],
}

impl Reg256 {
    pub const ZERO: Self = Self { lanes: [0; 4] };

    /// Create from a scalar 64-bit value (lane 0)
    #[inline]
    pub fn from_u64(val: u64) -> Self {
        Self { lanes: [val, 0, 0, 0] }
    }

    /// Create from a signed 64-bit value (lane 0)
    #[inline]
    pub fn from_i64(val: i64) -> Self {
        Self::from_u64(val as u64)
    }

    /// Read lane 0 as u64 (default scalar access)
    #[inline]
    pub fn as_u64(&self) -> u64 {
        self.lanes[0]
    }

    /// Read lane 0 as i64 (signed scalar access)
    #[inline]
    pub fn as_i64(&self) -> i64 {
        self.lanes[0] as i64
    }

    /// Read a specific 64-bit lane
    #[inline]
    pub fn d(&self, lane: usize) -> u64 {
        self.lanes[lane & 3]
    }

    /// Write a specific 64-bit lane
    #[inline]
    pub fn set_d(&mut self, lane: usize, val: u64) {
        self.lanes[lane & 3] = val;
    }

    /// Read a 32-bit word (8 words in a 256-bit register)
    #[inline]
    pub fn w(&self, idx: usize) -> u32 {
        let lane = (idx >> 1) & 3;
        let half = idx & 1;
        ((self.lanes[lane] >> (half * 32)) & 0xFFFF_FFFF) as u32
    }

    /// Write a 32-bit word
    #[inline]
    pub fn set_w(&mut self, idx: usize, val: u32) {
        let lane = (idx >> 1) & 3;
        let half = idx & 1;
        let mask = !(0xFFFF_FFFF_u64 << (half * 32));
        self.lanes[lane] = (self.lanes[lane] & mask) | ((val as u64) << (half * 32));
    }

    /// Read a byte (32 bytes in a 256-bit register)
    #[inline]
    pub fn b(&self, idx: usize) -> u8 {
        let lane = (idx >> 3) & 3;
        let byte_in_lane = idx & 7;
        ((self.lanes[lane] >> (byte_in_lane * 8)) & 0xFF) as u8
    }

    /// Write a byte
    #[inline]
    pub fn set_b(&mut self, idx: usize, val: u8) {
        let lane = (idx >> 3) & 3;
        let byte_in_lane = idx & 7;
        let mask = !(0xFF_u64 << (byte_in_lane * 8));
        self.lanes[lane] = (self.lanes[lane] & mask) | ((val as u64) << (byte_in_lane * 8));
    }

    /// Compute population count (total set bits across all 256 bits)
    pub fn popcount(&self) -> u32 {
        self.lanes.iter().map(|l| l.count_ones()).sum()
    }

    /// Compute byte-level Shannon entropy of the register contents.
    /// Returns entropy in bits (0.0 = all same, 5.0 = random-ish for 32 bytes).
    ///
    /// This is the ENTROPY instruction — no other ISA has this.
    pub fn shannon_entropy(&self) -> f64 {
        let mut freq = [0u32; 256];
        for i in 0..32 {
            freq[self.b(i) as usize] += 1;
        }
        let n = 32.0_f64;
        let mut entropy = 0.0_f64;
        for &count in &freq {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.log2();
            }
        }
        entropy
    }

    /// Hamming distance between two registers (count of differing bits)
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.lanes.iter()
            .zip(other.lanes.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Read as complex number pair (lane0 = real0, lane1 = imag0, lane2 = real1, lane3 = imag1)
    #[inline]
    pub fn complex0(&self) -> (f64, f64) {
        (f64::from_bits(self.lanes[0]), f64::from_bits(self.lanes[1]))
    }

    /// Read second complex number
    #[inline]
    pub fn complex1(&self) -> (f64, f64) {
        (f64::from_bits(self.lanes[2]), f64::from_bits(self.lanes[3]))
    }

    /// Write complex number 0
    #[inline]
    pub fn set_complex0(&mut self, re: f64, im: f64) {
        self.lanes[0] = re.to_bits();
        self.lanes[1] = im.to_bits();
    }

    /// Write complex number 1
    #[inline]
    pub fn set_complex1(&mut self, re: f64, im: f64) {
        self.lanes[2] = re.to_bits();
        self.lanes[3] = im.to_bits();
    }

    /// Read one of the eight f32 halves. Eight f32 is exactly 256 bits, which is
    /// what makes a 2x2 complex transfer matrix fit in one register (see
    /// `TensorNetworkUnit::zipper2_step`).
    #[inline]
    pub fn f32_at(&self, idx: usize) -> f32 {
        f32::from_bits(self.w(idx))
    }

    /// Write one of the eight f32 halves.
    #[inline]
    pub fn set_f32_at(&mut self, idx: usize, val: f32) {
        self.set_w(idx, val.to_bits());
    }

    /// Read one of the 32 bytes as a signed quantization code.
    #[inline]
    pub fn i8_at(&self, idx: usize) -> i8 {
        self.b(idx) as i8
    }

    /// Write one of the 32 bytes as a signed quantization code.
    #[inline]
    pub fn set_i8_at(&mut self, idx: usize, val: i8) {
        self.set_b(idx, val as u8);
    }
}

impl fmt::Debug for Reg256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:#018x} | {:#018x} | {:#018x} | {:#018x}]",
            self.lanes[3], self.lanes[2], self.lanes[1], self.lanes[0])
    }
}

impl fmt::Display for Reg256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Default: show lane 0 as decimal
        write!(f, "{}", self.lanes[0])
    }
}

// ─── Register Aliases ────────────────────────────────────────────────────────

pub const REG_ZERO: u8 = 0;   // Hardwired zero
pub const REG_RA: u8   = 1;   // Return address
pub const REG_SP: u8   = 2;   // Stack pointer
pub const REG_GP: u8   = 3;   // Global pointer
pub const REG_TP: u8   = 4;   // Thread pointer
pub const REG_T0: u8   = 5;   // Temporaries
pub const REG_T1: u8   = 6;
pub const REG_T2: u8   = 7;
pub const REG_FP: u8   = 8;   // Frame pointer / s0
pub const REG_S1: u8   = 9;
pub const REG_A0: u8   = 10;  // Arguments / return values
pub const REG_A1: u8   = 11;
pub const REG_A2: u8   = 12;
pub const REG_A3: u8   = 13;
pub const REG_A4: u8   = 14;
pub const REG_A5: u8   = 15;
pub const REG_A6: u8   = 16;
pub const REG_A7: u8   = 17;  // Syscall number

/// Map register name to index
pub fn parse_register(name: &str) -> Option<u8> {
    let name = name.trim().to_lowercase();
    match name.as_str() {
        "zero" | "x0"  => Some(0),
        "ra"   | "x1"  => Some(1),
        "sp"   | "x2"  => Some(2),
        "gp"   | "x3"  => Some(3),
        "tp"   | "x4"  => Some(4),
        "t0"   | "x5"  => Some(5),
        "t1"   | "x6"  => Some(6),
        "t2"   | "x7"  => Some(7),
        "fp"   | "s0"  | "x8"  => Some(8),
        "s1"   | "x9"  => Some(9),
        "a0"   | "x10" => Some(10),
        "a1"   | "x11" => Some(11),
        "a2"   | "x12" => Some(12),
        "a3"   | "x13" => Some(13),
        "a4"   | "x14" => Some(14),
        "a5"   | "x15" => Some(15),
        "a6"   | "x16" => Some(16),
        "a7"   | "x17" => Some(17),
        "s2"   | "x18" => Some(18),
        "s3"   | "x19" => Some(19),
        "s4"   | "x20" => Some(20),
        "s5"   | "x21" => Some(21),
        "s6"   | "x22" => Some(22),
        "s7"   | "x23" => Some(23),
        "s8"   | "x24" => Some(24),
        "s9"   | "x25" => Some(25),
        "s10"  | "x26" => Some(26),
        "s11"  | "x27" => Some(27),
        "t3"   | "x28" => Some(28),
        "t4"   | "x29" => Some(29),
        "t5"   | "x30" => Some(30),
        "t6"   | "x31" => Some(31),
        _ => None,
    }
}

/// Get register alias name
pub fn reg_name(idx: u8) -> &'static str {
    match idx {
        0  => "zero", 1  => "ra",  2  => "sp",  3  => "gp",
        4  => "tp",   5  => "t0",  6  => "t1",  7  => "t2",
        8  => "fp",   9  => "s1",  10 => "a0",  11 => "a1",
        12 => "a2",   13 => "a3",  14 => "a4",  15 => "a5",
        16 => "a6",   17 => "a7",  18 => "s2",  19 => "s3",
        20 => "s4",   21 => "s5",  22 => "s6",  23 => "s7",
        24 => "s8",   25 => "s9",  26 => "s10", 27 => "s11",
        28 => "t3",   29 => "t4",  30 => "t5",  31 => "t6",
        _  => "??",
    }
}

// ─── Lane Width ──────────────────────────────────────────────────────────────

/// Specifies the lane width for vector/tensor operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Width {
    B8,    // 32 lanes × 8-bit  (quantized AI, int8)
    B16,   // 16 lanes × 16-bit (AI inference, fp16/bf16)
    B32,   //  8 lanes × 32-bit (AI training, fp32)
    B64,   //  4 lanes × 64-bit (general vector, f64)
}

impl Width {
    pub fn lanes(&self) -> usize {
        match self {
            Width::B8  => 32,
            Width::B16 => 16,
            Width::B32 => 8,
            Width::B64 => 4,
        }
    }

    pub fn bits(&self) -> usize {
        match self {
            Width::B8  => 8,
            Width::B16 => 16,
            Width::B32 => 32,
            Width::B64 => 64,
        }
    }
}

impl fmt::Display for Width {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Width::B8  => write!(f, ".b"),
            Width::B16 => write!(f, ".h"),
            Width::B32 => write!(f, ".w"),
            Width::B64 => write!(f, ".d"),
        }
    }
}

// ─── Activation Functions (for tensor unit) ──────────────────────────────────

/// Neural network activation functions — native to the ISA.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationFn {
    ReLU,
    Sigmoid,
    Tanh,
    GeLU,
    SiLU,    // Swish
}

// ─── Instruction Set ─────────────────────────────────────────────────────────

/// The complete Unibit instruction set.
///
/// Categories:
///   SCALAR    — traditional RISC ALU on 64-bit (lane 0)
///   MEMORY    — load/store at various widths
///   BRANCH    — conditional/unconditional flow control
///   VECTOR    — SIMD on all lanes at specified width     [NOVEL]
///   TENSOR    — matrix/neural-network operations          [NOVEL]
///   LATTICE   — post-quantum cryptographic primitives     [NOVEL]
///   INFO      — information-theoretic operations          [NOVEL]
///   COMPLEX   — first-class complex number arithmetic     [NOVEL]
///   SYSTEM    — traps, CSRs, fences, halt
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instruction {
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SCALAR ALU — operate on lane 0 (64-bit), like a normal RISC processor
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Add    { rd: u8, rs1: u8, rs2: u8 },
    Sub    { rd: u8, rs1: u8, rs2: u8 },
    Mul    { rd: u8, rs1: u8, rs2: u8 },
    MulH   { rd: u8, rs1: u8, rs2: u8 },  // upper 64 bits of 128-bit product
    Div    { rd: u8, rs1: u8, rs2: u8 },
    Rem    { rd: u8, rs1: u8, rs2: u8 },
    And    { rd: u8, rs1: u8, rs2: u8 },
    Or     { rd: u8, rs1: u8, rs2: u8 },
    Xor    { rd: u8, rs1: u8, rs2: u8 },
    Sll    { rd: u8, rs1: u8, rs2: u8 },   // shift left logical
    Srl    { rd: u8, rs1: u8, rs2: u8 },   // shift right logical
    Sra    { rd: u8, rs1: u8, rs2: u8 },   // shift right arithmetic
    Slt    { rd: u8, rs1: u8, rs2: u8 },   // set if less than (signed)
    Sltu   { rd: u8, rs1: u8, rs2: u8 },   // set if less than (unsigned)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SCALAR IMMEDIATE — ALU with 12-bit/20-bit immediate
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Addi   { rd: u8, rs1: u8, imm: i64 },
    Andi   { rd: u8, rs1: u8, imm: i64 },
    Ori    { rd: u8, rs1: u8, imm: i64 },
    Xori   { rd: u8, rs1: u8, imm: i64 },
    Slli   { rd: u8, rs1: u8, imm: i64 },
    Srli   { rd: u8, rs1: u8, imm: i64 },
    Srai   { rd: u8, rs1: u8, imm: i64 },
    Slti   { rd: u8, rs1: u8, imm: i64 },
    Sltiu  { rd: u8, rs1: u8, imm: i64 },
    Lui    { rd: u8, imm: i64 },           // load upper immediate

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // MEMORY — load/store from data memory
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Ld     { rd: u8, rs1: u8, offset: i64 },   // load 64-bit doubleword
    Lw     { rd: u8, rs1: u8, offset: i64 },   // load 32-bit word (sign-extended)
    Lh     { rd: u8, rs1: u8, offset: i64 },   // load 16-bit half (sign-extended)
    Lb     { rd: u8, rs1: u8, offset: i64 },   // load 8-bit byte (sign-extended)
    Lbu    { rd: u8, rs1: u8, offset: i64 },   // load 8-bit byte (unsigned)
    Sd     { rs1: u8, rs2: u8, offset: i64 },  // store 64-bit
    Sw     { rs1: u8, rs2: u8, offset: i64 },  // store 32-bit
    Sh     { rs1: u8, rs2: u8, offset: i64 },  // store 16-bit
    Sb     { rs1: u8, rs2: u8, offset: i64 },  // store 8-bit

    // Full 256-bit load/store (for vector/tensor/crypto operations)
    Lq     { rd: u8, rs1: u8, offset: i64 },   // load 256-bit quad
    Sq     { rs1: u8, rs2: u8, offset: i64 },  // store 256-bit quad

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // BRANCH — conditional, target = signed instruction offset
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Beq    { rs1: u8, rs2: u8, offset: i64 },
    Bne    { rs1: u8, rs2: u8, offset: i64 },
    Blt    { rs1: u8, rs2: u8, offset: i64 },  // signed
    Bge    { rs1: u8, rs2: u8, offset: i64 },  // signed
    Bltu   { rs1: u8, rs2: u8, offset: i64 },  // unsigned
    Bgeu   { rs1: u8, rs2: u8, offset: i64 },  // unsigned

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // JUMP — unconditional, with link
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Jal    { rd: u8, offset: i64 },             // rd = PC+1, PC += offset
    Jalr   { rd: u8, rs1: u8, offset: i64 },    // rd = PC+1, PC = rs1 + offset

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // VECTOR — SIMD on all lanes at specified width              ◆ NOVEL ◆
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    VAdd   { rd: u8, rs1: u8, rs2: u8, width: Width },

    // ─── Packed single-precision ─────────────────────────────────────────────
    //
    // Eight f32 lanes, which is what `Width::B32` has always been documented as. These carry
    // no `Width`: single precision is 32 bits by definition, so a width suffix would be a
    // field with one legal value, and a field with one legal value is a field that will
    // eventually be set to the other one.
    VfAdd  { rd: u8, rs1: u8, rs2: u8 },
    VfSub  { rd: u8, rs1: u8, rs2: u8 },
    VfMul  { rd: u8, rs1: u8, rs2: u8 },
    /// `rd = rs1 * rs2 + rd`, fused: one rounding, not two.
    VfMa   { rd: u8, rs1: u8, rs2: u8 },
    VfMax  { rd: u8, rs1: u8, rs2: u8 },
    VfMin  { rd: u8, rs1: u8, rs2: u8 },
    /// Horizontal sum of the eight f32 lanes into lane 0, as a **tree**: pairs at stride 4,
    /// then 2, then 1.
    ///
    /// The order is part of the instruction, not an implementation detail. `VREDUCE` beside it
    /// sums integers with `wrapping_add`, which is associative, so its order was never
    /// observable and was therefore never a decision. Float addition is not associative: a
    /// sequential chain and a tree give different bit patterns over the same lanes, so this
    /// instruction has to say which one it is. A tree is also what the hardware would be --
    /// three layers of adders, not a chain of eight.
    VfReduce { rd: u8, rs1: u8 },
    VSub   { rd: u8, rs1: u8, rs2: u8, width: Width },
    VMul   { rd: u8, rs1: u8, rs2: u8, width: Width },
    VAnd   { rd: u8, rs1: u8, rs2: u8 },  // bitwise across full 256-bit
    VOr    { rd: u8, rs1: u8, rs2: u8 },
    VXor   { rd: u8, rs1: u8, rs2: u8 },
    VNot   { rd: u8, rs1: u8 },
    VDot   { rd: u8, rs1: u8, rs2: u8, width: Width }, // dot product → scalar in lane 0
    VSplat { rd: u8, rs1: u8, width: Width },           // broadcast lane 0 to all lanes
    VReduce{ rd: u8, rs1: u8, width: Width },           // sum all lanes → lane 0

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TENSOR NETWORK / MPS ACCELERATOR                                ◆ NOVEL ◆
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // ZIPPER and ZIPPER2 are ACCUMULATOR instructions: unlike every other
    // three-operand instruction in this ISA, rd is *read* as the incoming
    // transfer matrix E before being written with the outgoing one.
    //     zipper  rd, rs1, rs2   ->   rd = rd * sum_d rs1[d] * conj(rs2[d])
    Zipper { rd: u8, rs1: u8, rs2: u8 },     // MPS zipper contraction step at bond dimension 1
    Zipper2{ rd: u8, rs1: u8, rs2: u8 },     // MPS zipper step at bond dimension 2 (f32 E, int8 cores)
    Trunc  { rd: u8, rs1: u8, eps_bits: u64 }, // Schmidt coefficient truncation below eps * max(S)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // TENSOR — Neural network / matrix operations                ◆ NOVEL ◆
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    TMul   { rd: u8, rs1: u8, rs2: u8 },                 // 2×2 f64 matrix multiply (row-major)
    TDot   { rd: u8, rs1: u8, rs2: u8 },                 // f64 dot product over 4 lanes → lane 0
    TAct   { rd: u8, rs1: u8, func: ActivationFn },      // activation function
    TSoftmax { rd: u8, rs1: u8 },                         // softmax over lanes

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // LATTICE — Post-quantum cryptographic primitives            ◆ NOVEL ◆
    // Negacyclic transform over R_q = Z_q[X]/(X^4+1); see alu::LatticeUnit.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Ntt    { rd: u8, rs1: u8 },                    // Number Theoretic Transform
    InvNtt { rd: u8, rs1: u8 },                    // Inverse NTT
    PolyMul{ rd: u8, rs1: u8, rs2: u8 },           // Polynomial multiply in ring
    ModRed { rd: u8, rs1: u8, modulus: u64 },      // Modular reduction
    PolyAdd{ rd: u8, rs1: u8, rs2: u8 },           // Polynomial addition

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // INFORMATION THEORY — no other ISA has these                ◆ NOVEL ◆
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Entropy  { rd: u8, rs1: u8 },     // rd = Shannon entropy of rs1 (as f64 bits)
    Hamming  { rd: u8, rs1: u8, rs2: u8 }, // rd = Hamming distance (differing bits)
    PopCnt   { rd: u8, rs1: u8 },     // rd = population count of rs1 (total set bits)
    QRand    { rd: u8 },              // fill rd with PRNG output (quantum-inspired)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // COMPLEX — first-class complex number arithmetic            ◆ NOVEL ◆
    // Operates on 128-bit complex pairs: lanes[0:1] = z0, lanes[2:3] = z1
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    CAdd  { rd: u8, rs1: u8, rs2: u8 },  // complex add (both pairs)
    CSub  { rd: u8, rs1: u8, rs2: u8 },  // complex subtract
    CMul  { rd: u8, rs1: u8, rs2: u8 },  // complex multiply: (a+bi)(c+di)
    CConj { rd: u8, rs1: u8 },           // complex conjugate (negate imag parts)
    CNorm { rd: u8, rs1: u8 },           // |z|² → real part of each pair
    CMag  { rd: u8, rs1: u8 },           // |z| → real part (sqrt of norm)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // QUANTUM COPROCESSOR (U-QPU, simulated; ADR-0003)
    // Qubit indices are immediates. A register of up to 24 qubits lives in the
    // backend (phase 1: an exact host state vector). Feed-forward is ordinary
    // control flow: QMEAS writes a register, a branch conditions what follows.
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    QAlloc { n: u8 },                        // fresh n-qubit register in |0..0>
    QG1    { gate: QGate, q: u8 },           // fixed one-qubit gate on q
    QRot   { axis: QAxis, q: u8, rs: u8 },   // R_axis(theta) on q, theta = f64 in lane 0 of rs
    QCx    { c: u8, t: u8 },                 // CNOT
    QCz    { a: u8, b: u8 },                 // CZ
    QMeas  { rd: u8, q: u8 },                // Z measurement of q into rd (0/1); collapses
    QReset { q: u8 },                        // measure q and bring it to |0>
    QSeed  { rs: u8 },                       // seed the measurement RNG from lane 0 of rs
    QAmp   { rd: u8, rs: u8 },               // amplitude of basis index rs -> rd as complex (re, im)

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // SYSTEM
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Ecall,                               // system call (a7 = syscall number)
    Halt,                                // stop execution
    Nop,                                 // no operation
    Fence,                               // memory fence
    CsrR   { rd: u8, csr: u16 },        // read CSR to rd
    CsrW   { csr: u16, rs1: u8 },       // write rs1 to CSR

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // MOVE / CONVERSION
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Mv     { rd: u8, rs1: u8 },          // rd = rs1 (alias for ADDI rd, rs1, 0)
    Li     { rd: u8, imm: i64 },         // load immediate (pseudo-instruction)
    // `la rd, label` is not a variant: the assembler resolves the label and
    // emits Li, so no unresolved symbol can ever reach the execution stage.
}

impl Instruction {
}

// ─── CSR (Control and Status Registers) ──────────────────────────────────────

/// CSR addresses for Unibit
pub mod csr {
    pub const CYCLE: u16       = 0xC00;  // Cycle counter (read-only)
    pub const INSTRET: u16     = 0xC02;  // Instructions retired (read-only)
    pub const ENTROPY_ACC: u16 = 0x800;  // Accumulated entropy (custom, read/write)
    pub const ENERGY_EST: u16  = 0x801;  // Estimated energy consumption (custom)
    pub const TEMP_K: u16      = 0x802;  // Operating temperature in Kelvin (custom)
    pub const LANDAUER: u16    = 0x803;  // Current Landauer limit at TEMP_K (read-only)
}

// ─── Syscall numbers ─────────────────────────────────────────────────────────

pub mod syscall {
    pub const PRINT_INT: u64    = 1;   // a0 = integer to print
    pub const PRINT_CHAR: u64   = 2;   // a0 = char to print
    pub const PRINT_STR: u64    = 3;   // a0 = address, a1 = length
    pub const PRINT_HEX: u64    = 4;   // a0 = integer to print as hex
    pub const PRINT_F64: u64    = 5;   // a0 = f64 bits to print
    pub const PRINT_STRZ: u64   = 6;   // a0 = address of NUL-terminated string
    /// a0 = f32 bits in the low 32. Printed **shortest round-trip**, not to fixed precision.
    ///
    /// The machine has had eight f32 lanes and packed arithmetic over them since the float
    /// unit landed, and no way to print one: PRINT_F64 takes f64 bits and there is no
    /// conversion instruction to make them from an f32. Same shape of gap as the arithmetic
    /// itself had.
    ///
    /// The formatting is part of the syscall. PRINT_F64 beside it prints {:.6}, which cannot
    /// be read back -- two different f64 print the same eight characters. A program whose
    /// output is checked against a host oracle needs its floats to survive the trip, so this
    /// one emits the shortest string that parses back to the same bits.
    pub const PRINT_F32: u64    = 7;
    pub const READ_INT: u64     = 10;  // a0 = read integer from stdin
    pub const PRINT_REG256: u64 = 20;  // a0 = register index, prints full 256-bit
    pub const EXIT: u64         = 93;  // a0 = exit code
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reg256_scalar() {
        let r = Reg256::from_u64(42);
        assert_eq!(r.as_u64(), 42);
        assert_eq!(r.as_i64(), 42);
        assert_eq!(r.d(0), 42);
        assert_eq!(r.d(1), 0);
    }

    #[test]
    fn test_reg256_negative() {
        let r = Reg256::from_i64(-1);
        assert_eq!(r.as_i64(), -1);
        assert_eq!(r.as_u64(), u64::MAX);
    }

    #[test]
    fn test_reg256_bytes() {
        let mut r = Reg256::ZERO;
        r.set_b(0, 0xAB);
        r.set_b(7, 0xCD);
        assert_eq!(r.b(0), 0xAB);
        assert_eq!(r.b(7), 0xCD);
        assert_eq!(r.b(1), 0x00);
    }

    #[test]
    fn test_reg256_lane_accessors() {
        // d/set_d are the 64-bit half of the symmetric lane API.
        let mut r = Reg256::ZERO;
        for lane in 0..4 {
            r.set_d(lane, 0x1122_3344_5566_7700 | lane as u64);
        }
        for lane in 0..4 {
            assert_eq!(r.d(lane), 0x1122_3344_5566_7700 | lane as u64);
        }
        // The lane index wraps at 4, matching the other accessors.
        assert_eq!(r.d(4), r.d(0));
        r.set_d(5, 0xDEAD);
        assert_eq!(r.d(1), 0xDEAD);
    }

    #[test]
    fn test_reg256_words() {
        let mut r = Reg256::ZERO;
        r.set_w(0, 0xDEAD_BEEF);
        assert_eq!(r.w(0), 0xDEAD_BEEF);
        assert_eq!(r.w(1), 0);
    }

    #[test]
    fn test_popcount() {
        let r = Reg256::from_u64(0xFF); // 8 bits set
        assert_eq!(r.popcount(), 8);
    }

    #[test]
    fn test_entropy_zero() {
        let r = Reg256::ZERO; // all zeros → entropy = 0
        assert_eq!(r.shannon_entropy(), 0.0);
    }

    #[test]
    fn test_hamming() {
        let a = Reg256::from_u64(0b1111);
        let b = Reg256::from_u64(0b0000);
        assert_eq!(a.hamming_distance(&b), 4);
    }

    #[test]
    fn test_complex() {
        let mut r = Reg256::ZERO;
        r.set_complex0(3.0, 4.0);
        let (re, im) = r.complex0();
        assert_eq!(re, 3.0);
        assert_eq!(im, 4.0);
    }

    #[test]
    fn test_parse_register() {
        assert_eq!(parse_register("zero"), Some(0));
        assert_eq!(parse_register("a0"), Some(10));
        assert_eq!(parse_register("x31"), Some(31));
        assert_eq!(parse_register("invalid"), None);
    }
}
