// ============================================================================
// Unibit — Object Format & Instruction Encoding
// ============================================================================
//
// The on-disk representation of an assembled program, and the encoder/decoder
// that turns instructions into raw bytes and back.
//
// This module is what makes `unibit disasm` an actual disassembler: it decodes
// a byte stream with no access to the assembler's AST or symbol table.
//
// ─── Instruction record: 16 bytes, little-endian ────────────────────────────
//
//   [0]      opcode
//   [1]      rd
//   [2]      rs1
//   [3]      rs2
//   [4..8]   aux    (vector width / activation function / CSR number)
//   [8..16]  imm    (immediate, branch offset, or 64-bit unsigned operand)
//
// A fixed-width record keeps decoding trivially seekable: instruction i always
// starts at code_offset + i * 16, so a disassembler can start anywhere.
//
// ─── Object file layout ─────────────────────────────────────────────────────
//
//   magic        [u8; 4] = b"UBIT"
//   version      u16
//   reserved     u16
//   entry_point  u64     (instruction index, not a byte offset)
//   code_len     u64     (instruction count)
//   data_len     u64     (data segment count)
//   code         code_len * 16 bytes
//   data         data_len * (addr u64, len u64, len bytes)
//
// ============================================================================
//
// GENERATED SECTIONS: the opcode table, `encode_instruction` and
// `decode_instruction` are emitted from a single source table, so the two
// directions cannot drift apart. `test_encode_decode_covers_every_opcode`
// checks the round-trip over every opcode in the ISA.

use crate::isa::{ActivationFn, Instruction, Width};

/// Bytes per encoded instruction.
pub const INSTRUCTION_BYTES: usize = 16;

/// Object file magic number.
pub const MAGIC: [u8; 4] = *b"UBIT";

/// Object format version.
pub const VERSION: u16 = 1;

const HEADER_BYTES: usize = 4 + 2 + 2 + 8 + 8 + 8;

/// Opcode assignments. Values are part of the on-disk format: append new
/// opcodes at the end, never renumber existing ones.
pub mod op {
    pub const ADD:       u8 = 1;
    pub const SUB:       u8 = 2;
    pub const MUL:       u8 = 3;
    pub const MULH:      u8 = 4;
    pub const DIV:       u8 = 5;
    pub const REM:       u8 = 6;
    pub const AND:       u8 = 7;
    pub const OR:        u8 = 8;
    pub const XOR:       u8 = 9;
    pub const SLL:       u8 = 10;
    pub const SRL:       u8 = 11;
    pub const SRA:       u8 = 12;
    pub const SLT:       u8 = 13;
    pub const SLTU:      u8 = 14;
    pub const ADDI:      u8 = 15;
    pub const ANDI:      u8 = 16;
    pub const ORI:       u8 = 17;
    pub const XORI:      u8 = 18;
    pub const SLLI:      u8 = 19;
    pub const SRLI:      u8 = 20;
    pub const SRAI:      u8 = 21;
    pub const SLTI:      u8 = 22;
    pub const SLTIU:     u8 = 23;
    pub const LUI:       u8 = 24;
    pub const LD:        u8 = 25;
    pub const LW:        u8 = 26;
    pub const LH:        u8 = 27;
    pub const LB:        u8 = 28;
    pub const LBU:       u8 = 29;
    pub const LQ:        u8 = 30;
    pub const SD:        u8 = 31;
    pub const SW:        u8 = 32;
    pub const SH:        u8 = 33;
    pub const SB:        u8 = 34;
    pub const SQ:        u8 = 35;
    pub const BEQ:       u8 = 36;
    pub const BNE:       u8 = 37;
    pub const BLT:       u8 = 38;
    pub const BGE:       u8 = 39;
    pub const BLTU:      u8 = 40;
    pub const BGEU:      u8 = 41;
    pub const JAL:       u8 = 42;
    pub const JALR:      u8 = 43;
    pub const VADD:      u8 = 44;
    pub const VSUB:      u8 = 45;
    pub const VMUL:      u8 = 46;
    pub const VDOT:      u8 = 47;
    pub const VAND:      u8 = 48;
    pub const VOR:       u8 = 49;
    pub const VXOR:      u8 = 50;
    pub const VNOT:      u8 = 51;
    pub const VSPLAT:    u8 = 52;
    pub const VREDUCE:   u8 = 53;
    pub const ZIPPER:    u8 = 54;
    pub const TRUNC:     u8 = 55;
    pub const TMUL:      u8 = 56;
    pub const TDOT:      u8 = 57;
    pub const TACT:      u8 = 58;
    pub const TSOFTMAX:  u8 = 59;
    pub const NTT:       u8 = 60;
    pub const INVNTT:    u8 = 61;
    pub const POLYMUL:   u8 = 62;
    pub const MODRED:    u8 = 63;
    pub const POLYADD:   u8 = 64;
    pub const ENTROPY:   u8 = 65;
    pub const HAMMING:   u8 = 66;
    pub const POPCNT:    u8 = 67;
    pub const QRAND:     u8 = 68;
    pub const CADD:      u8 = 69;
    pub const CSUB:      u8 = 70;
    pub const CMUL:      u8 = 71;
    pub const CCONJ:     u8 = 72;
    pub const CNORM:     u8 = 73;
    pub const CMAG:      u8 = 74;

    // Packed single-precision. Appended rather than slotted in beside VADD, because an
    // object file built before these existed must still decode: renumbering an opcode
    // silently changes what an old `.ubo` means.
    pub const VFADD:     u8 = 84;
    pub const VFSUB:     u8 = 85;
    pub const VFMUL:     u8 = 86;
    pub const VFMA:      u8 = 87;
    pub const VFMAX:     u8 = 88;
    pub const VFMIN:     u8 = 89;
    pub const VFREDUCE:  u8 = 90;
    // U-QPU phase 1 (ADR-0003)
    pub const QALLOC:    u8 = 91;
    pub const QG1:       u8 = 92;
    pub const QROT:      u8 = 93;
    pub const QCX:       u8 = 94;
    pub const QCZ:       u8 = 95;
    pub const QMEAS:     u8 = 96;
    pub const QRESET:    u8 = 97;
    pub const QSEED:     u8 = 98;
    pub const QAMP:      u8 = 99;
    pub const ECALL:     u8 = 75;
    pub const HALT:      u8 = 76;
    pub const NOP:       u8 = 77;
    pub const FENCE:     u8 = 78;
    pub const CSRR:      u8 = 79;
    pub const CSRW:      u8 = 80;
    pub const MV:        u8 = 81;
    pub const LI:        u8 = 82;
    pub const ZIPPER2:   u8 = 83;
}

/// An assembled program in loadable form.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Object {
    pub entry_point: u64,
    pub code: Vec<Instruction>,
    pub data: Vec<(u64, Vec<u8>)>,
}

// ─── Auxiliary field codecs ─────────────────────────────────────────────────

fn width_code(w: Width) -> u32 {
    match w {
        Width::B8 => 0,
        Width::B16 => 1,
        Width::B32 => 2,
        Width::B64 => 3,
    }
}

fn width_from_code(c: u32) -> Result<Width, String> {
    match c {
        0 => Ok(Width::B8),
        1 => Ok(Width::B16),
        2 => Ok(Width::B32),
        3 => Ok(Width::B64),
        _ => Err(format!("invalid vector width code {}", c)),
    }
}

fn act_code(f: ActivationFn) -> u32 {
    match f {
        ActivationFn::ReLU => 0,
        ActivationFn::Sigmoid => 1,
        ActivationFn::Tanh => 2,
        ActivationFn::GeLU => 3,
        ActivationFn::SiLU => 4,
    }
}

fn act_from_code(c: u32) -> Result<ActivationFn, String> {
    match c {
        0 => Ok(ActivationFn::ReLU),
        1 => Ok(ActivationFn::Sigmoid),
        2 => Ok(ActivationFn::Tanh),
        3 => Ok(ActivationFn::GeLU),
        4 => Ok(ActivationFn::SiLU),
        _ => Err(format!("invalid activation function code {}", c)),
    }
}

// ─── Instruction encoding ───────────────────────────────────────────────────

/// The decoded shape of one instruction record, before it is matched back to
/// an `Instruction` variant.
struct Fields {
    op: u8,
    rd: u8,
    rs1: u8,
    rs2: u8,
    aux: u32,
    imm: i64,
}

/// Encode one instruction into its 16-byte record.
pub fn encode_instruction(inst: &Instruction) -> [u8; INSTRUCTION_BYTES] {
    let f = match inst {
        Instruction::Add { rd, rs1, rs2 } => Fields { op: op::ADD, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Sub { rd, rs1, rs2 } => Fields { op: op::SUB, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Mul { rd, rs1, rs2 } => Fields { op: op::MUL, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::MulH { rd, rs1, rs2 } => Fields { op: op::MULH, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Div { rd, rs1, rs2 } => Fields { op: op::DIV, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Rem { rd, rs1, rs2 } => Fields { op: op::REM, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::And { rd, rs1, rs2 } => Fields { op: op::AND, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Or { rd, rs1, rs2 } => Fields { op: op::OR, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Xor { rd, rs1, rs2 } => Fields { op: op::XOR, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Sll { rd, rs1, rs2 } => Fields { op: op::SLL, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Srl { rd, rs1, rs2 } => Fields { op: op::SRL, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Sra { rd, rs1, rs2 } => Fields { op: op::SRA, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Slt { rd, rs1, rs2 } => Fields { op: op::SLT, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Sltu { rd, rs1, rs2 } => Fields { op: op::SLTU, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Addi { rd, rs1, imm } => Fields { op: op::ADDI, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *imm },
        Instruction::Andi { rd, rs1, imm } => Fields { op: op::ANDI, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *imm },
        Instruction::Ori { rd, rs1, imm } => Fields { op: op::ORI, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *imm },
        Instruction::Xori { rd, rs1, imm } => Fields { op: op::XORI, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *imm },
        Instruction::Slli { rd, rs1, imm } => Fields { op: op::SLLI, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *imm },
        Instruction::Srli { rd, rs1, imm } => Fields { op: op::SRLI, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *imm },
        Instruction::Srai { rd, rs1, imm } => Fields { op: op::SRAI, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *imm },
        Instruction::Slti { rd, rs1, imm } => Fields { op: op::SLTI, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *imm },
        Instruction::Sltiu { rd, rs1, imm } => Fields { op: op::SLTIU, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *imm },
        Instruction::Lui { rd, imm } => Fields { op: op::LUI, rd: *rd, rs1: 0, rs2: 0, aux: 0, imm: *imm },
        Instruction::Ld { rd, rs1, offset } => Fields { op: op::LD, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *offset },
        Instruction::Lw { rd, rs1, offset } => Fields { op: op::LW, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *offset },
        Instruction::Lh { rd, rs1, offset } => Fields { op: op::LH, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *offset },
        Instruction::Lb { rd, rs1, offset } => Fields { op: op::LB, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *offset },
        Instruction::Lbu { rd, rs1, offset } => Fields { op: op::LBU, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *offset },
        Instruction::Lq { rd, rs1, offset } => Fields { op: op::LQ, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *offset },
        Instruction::Sd { rs1, rs2, offset } => Fields { op: op::SD, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Sw { rs1, rs2, offset } => Fields { op: op::SW, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Sh { rs1, rs2, offset } => Fields { op: op::SH, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Sb { rs1, rs2, offset } => Fields { op: op::SB, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Sq { rs1, rs2, offset } => Fields { op: op::SQ, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Beq { rs1, rs2, offset } => Fields { op: op::BEQ, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Bne { rs1, rs2, offset } => Fields { op: op::BNE, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Blt { rs1, rs2, offset } => Fields { op: op::BLT, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Bge { rs1, rs2, offset } => Fields { op: op::BGE, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Bltu { rs1, rs2, offset } => Fields { op: op::BLTU, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Bgeu { rs1, rs2, offset } => Fields { op: op::BGEU, rd: 0, rs1: *rs1, rs2: *rs2, aux: 0, imm: *offset },
        Instruction::Jal { rd, offset } => Fields { op: op::JAL, rd: *rd, rs1: 0, rs2: 0, aux: 0, imm: *offset },
        Instruction::Jalr { rd, rs1, offset } => Fields { op: op::JALR, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *offset },
        Instruction::VAdd { rd, rs1, rs2, width } => Fields { op: op::VADD, rd: *rd, rs1: *rs1, rs2: *rs2, aux: width_code(*width), imm: 0 },
        Instruction::VfAdd { rd, rs1, rs2 } => Fields { op: op::VFADD, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::VfSub { rd, rs1, rs2 } => Fields { op: op::VFSUB, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::VfMul { rd, rs1, rs2 } => Fields { op: op::VFMUL, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::VfMa  { rd, rs1, rs2 } => Fields { op: op::VFMA,  rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::VfMax { rd, rs1, rs2 } => Fields { op: op::VFMAX, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::VfMin { rd, rs1, rs2 } => Fields { op: op::VFMIN, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::VfReduce { rd, rs1 } => Fields { op: op::VFREDUCE, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::VSub { rd, rs1, rs2, width } => Fields { op: op::VSUB, rd: *rd, rs1: *rs1, rs2: *rs2, aux: width_code(*width), imm: 0 },
        Instruction::VMul { rd, rs1, rs2, width } => Fields { op: op::VMUL, rd: *rd, rs1: *rs1, rs2: *rs2, aux: width_code(*width), imm: 0 },
        Instruction::VDot { rd, rs1, rs2, width } => Fields { op: op::VDOT, rd: *rd, rs1: *rs1, rs2: *rs2, aux: width_code(*width), imm: 0 },
        Instruction::VAnd { rd, rs1, rs2 } => Fields { op: op::VAND, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::VOr { rd, rs1, rs2 } => Fields { op: op::VOR, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::VXor { rd, rs1, rs2 } => Fields { op: op::VXOR, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::VNot { rd, rs1 } => Fields { op: op::VNOT, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::VSplat { rd, rs1, width } => Fields { op: op::VSPLAT, rd: *rd, rs1: *rs1, rs2: 0, aux: width_code(*width), imm: 0 },
        Instruction::VReduce { rd, rs1, width } => Fields { op: op::VREDUCE, rd: *rd, rs1: *rs1, rs2: 0, aux: width_code(*width), imm: 0 },
        Instruction::Zipper { rd, rs1, rs2 } => Fields { op: op::ZIPPER, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Zipper2 { rd, rs1, rs2 } => Fields { op: op::ZIPPER2, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Trunc { rd, rs1, eps_bits } => Fields { op: op::TRUNC, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *eps_bits as i64 },
        Instruction::TMul { rd, rs1, rs2 } => Fields { op: op::TMUL, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::TDot { rd, rs1, rs2 } => Fields { op: op::TDOT, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::TAct { rd, rs1, func } => Fields { op: op::TACT, rd: *rd, rs1: *rs1, rs2: 0, aux: act_code(*func), imm: 0 },
        Instruction::TSoftmax { rd, rs1 } => Fields { op: op::TSOFTMAX, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::Ntt { rd, rs1 } => Fields { op: op::NTT, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::InvNtt { rd, rs1 } => Fields { op: op::INVNTT, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::PolyMul { rd, rs1, rs2 } => Fields { op: op::POLYMUL, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::ModRed { rd, rs1, modulus } => Fields { op: op::MODRED, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: *modulus as i64 },
        Instruction::PolyAdd { rd, rs1, rs2 } => Fields { op: op::POLYADD, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::Entropy { rd, rs1 } => Fields { op: op::ENTROPY, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::Hamming { rd, rs1, rs2 } => Fields { op: op::HAMMING, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::PopCnt { rd, rs1 } => Fields { op: op::POPCNT, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::QRand { rd } => Fields { op: op::QRAND, rd: *rd, rs1: 0, rs2: 0, aux: 0, imm: 0 },
        Instruction::QAlloc { n } => Fields { op: op::QALLOC, rd: 0, rs1: 0, rs2: 0, aux: *n as u32, imm: 0 },
        Instruction::QG1 { gate, q } => Fields { op: op::QG1, rd: 0, rs1: *q, rs2: 0, aux: gate.code() as u32, imm: 0 },
        Instruction::QRot { axis, q, rs } => Fields { op: op::QROT, rd: 0, rs1: *q, rs2: *rs, aux: axis.code() as u32, imm: 0 },
        Instruction::QCx { c, t } => Fields { op: op::QCX, rd: 0, rs1: *c, rs2: *t, aux: 0, imm: 0 },
        Instruction::QCz { a, b } => Fields { op: op::QCZ, rd: 0, rs1: *a, rs2: *b, aux: 0, imm: 0 },
        Instruction::QMeas { rd, q } => Fields { op: op::QMEAS, rd: *rd, rs1: *q, rs2: 0, aux: 0, imm: 0 },
        Instruction::QReset { q } => Fields { op: op::QRESET, rd: 0, rs1: *q, rs2: 0, aux: 0, imm: 0 },
        Instruction::QSeed { rs } => Fields { op: op::QSEED, rd: 0, rs1: *rs, rs2: 0, aux: 0, imm: 0 },
        Instruction::QAmp { rd, rs } => Fields { op: op::QAMP, rd: *rd, rs1: *rs, rs2: 0, aux: 0, imm: 0 },
        Instruction::CAdd { rd, rs1, rs2 } => Fields { op: op::CADD, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::CSub { rd, rs1, rs2 } => Fields { op: op::CSUB, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::CMul { rd, rs1, rs2 } => Fields { op: op::CMUL, rd: *rd, rs1: *rs1, rs2: *rs2, aux: 0, imm: 0 },
        Instruction::CConj { rd, rs1 } => Fields { op: op::CCONJ, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::CNorm { rd, rs1 } => Fields { op: op::CNORM, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::CMag { rd, rs1 } => Fields { op: op::CMAG, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::Ecall => Fields { op: op::ECALL, rd: 0, rs1: 0, rs2: 0, aux: 0, imm: 0 },
        Instruction::Halt => Fields { op: op::HALT, rd: 0, rs1: 0, rs2: 0, aux: 0, imm: 0 },
        Instruction::Nop => Fields { op: op::NOP, rd: 0, rs1: 0, rs2: 0, aux: 0, imm: 0 },
        Instruction::Fence => Fields { op: op::FENCE, rd: 0, rs1: 0, rs2: 0, aux: 0, imm: 0 },
        Instruction::CsrR { rd, csr } => Fields { op: op::CSRR, rd: *rd, rs1: 0, rs2: 0, aux: *csr as u32, imm: 0 },
        Instruction::CsrW { csr, rs1 } => Fields { op: op::CSRW, rd: 0, rs1: *rs1, rs2: 0, aux: *csr as u32, imm: 0 },
        Instruction::Mv { rd, rs1 } => Fields { op: op::MV, rd: *rd, rs1: *rs1, rs2: 0, aux: 0, imm: 0 },
        Instruction::Li { rd, imm } => Fields { op: op::LI, rd: *rd, rs1: 0, rs2: 0, aux: 0, imm: *imm },
    };

    let mut out = [0u8; INSTRUCTION_BYTES];
    out[0] = f.op;
    out[1] = f.rd;
    out[2] = f.rs1;
    out[3] = f.rs2;
    out[4..8].copy_from_slice(&f.aux.to_le_bytes());
    out[8..16].copy_from_slice(&f.imm.to_le_bytes());
    out
}

/// Decode one 16-byte instruction record.
pub fn decode_instruction(bytes: &[u8]) -> Result<Instruction, String> {
    if bytes.len() < INSTRUCTION_BYTES {
        return Err(format!(
            "instruction record truncated: got {} bytes, need {}",
            bytes.len(),
            INSTRUCTION_BYTES
        ));
    }

    let f = Fields {
        op: bytes[0],
        rd: bytes[1],
        rs1: bytes[2],
        rs2: bytes[3],
        aux: u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
        imm: i64::from_le_bytes(bytes[8..16].try_into().unwrap()),
    };

    Ok(match f.op {
        op::ADD => Instruction::Add { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::SUB => Instruction::Sub { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::MUL => Instruction::Mul { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::MULH => Instruction::MulH { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::DIV => Instruction::Div { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::REM => Instruction::Rem { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::AND => Instruction::And { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::OR => Instruction::Or { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::XOR => Instruction::Xor { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::SLL => Instruction::Sll { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::SRL => Instruction::Srl { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::SRA => Instruction::Sra { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::SLT => Instruction::Slt { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::SLTU => Instruction::Sltu { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::ADDI => Instruction::Addi { rd: f.rd, rs1: f.rs1, imm: f.imm },
        op::ANDI => Instruction::Andi { rd: f.rd, rs1: f.rs1, imm: f.imm },
        op::ORI => Instruction::Ori { rd: f.rd, rs1: f.rs1, imm: f.imm },
        op::XORI => Instruction::Xori { rd: f.rd, rs1: f.rs1, imm: f.imm },
        op::SLLI => Instruction::Slli { rd: f.rd, rs1: f.rs1, imm: f.imm },
        op::SRLI => Instruction::Srli { rd: f.rd, rs1: f.rs1, imm: f.imm },
        op::SRAI => Instruction::Srai { rd: f.rd, rs1: f.rs1, imm: f.imm },
        op::SLTI => Instruction::Slti { rd: f.rd, rs1: f.rs1, imm: f.imm },
        op::SLTIU => Instruction::Sltiu { rd: f.rd, rs1: f.rs1, imm: f.imm },
        op::LUI => Instruction::Lui { rd: f.rd, imm: f.imm },
        op::LD => Instruction::Ld { rd: f.rd, rs1: f.rs1, offset: f.imm },
        op::LW => Instruction::Lw { rd: f.rd, rs1: f.rs1, offset: f.imm },
        op::LH => Instruction::Lh { rd: f.rd, rs1: f.rs1, offset: f.imm },
        op::LB => Instruction::Lb { rd: f.rd, rs1: f.rs1, offset: f.imm },
        op::LBU => Instruction::Lbu { rd: f.rd, rs1: f.rs1, offset: f.imm },
        op::LQ => Instruction::Lq { rd: f.rd, rs1: f.rs1, offset: f.imm },
        op::SD => Instruction::Sd { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::SW => Instruction::Sw { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::SH => Instruction::Sh { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::SB => Instruction::Sb { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::SQ => Instruction::Sq { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::BEQ => Instruction::Beq { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::BNE => Instruction::Bne { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::BLT => Instruction::Blt { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::BGE => Instruction::Bge { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::BLTU => Instruction::Bltu { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::BGEU => Instruction::Bgeu { rs1: f.rs1, rs2: f.rs2, offset: f.imm },
        op::JAL => Instruction::Jal { rd: f.rd, offset: f.imm },
        op::JALR => Instruction::Jalr { rd: f.rd, rs1: f.rs1, offset: f.imm },
        op::VADD => Instruction::VAdd { rd: f.rd, rs1: f.rs1, rs2: f.rs2, width: width_from_code(f.aux)? },
        op::VFADD => Instruction::VfAdd { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::VFSUB => Instruction::VfSub { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::VFMUL => Instruction::VfMul { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::VFMA  => Instruction::VfMa  { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::VFMAX => Instruction::VfMax { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::VFMIN => Instruction::VfMin { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::VFREDUCE => Instruction::VfReduce { rd: f.rd, rs1: f.rs1 },
        op::VSUB => Instruction::VSub { rd: f.rd, rs1: f.rs1, rs2: f.rs2, width: width_from_code(f.aux)? },
        op::VMUL => Instruction::VMul { rd: f.rd, rs1: f.rs1, rs2: f.rs2, width: width_from_code(f.aux)? },
        op::VDOT => Instruction::VDot { rd: f.rd, rs1: f.rs1, rs2: f.rs2, width: width_from_code(f.aux)? },
        op::VAND => Instruction::VAnd { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::VOR => Instruction::VOr { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::VXOR => Instruction::VXor { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::VNOT => Instruction::VNot { rd: f.rd, rs1: f.rs1 },
        op::VSPLAT => Instruction::VSplat { rd: f.rd, rs1: f.rs1, width: width_from_code(f.aux)? },
        op::VREDUCE => Instruction::VReduce { rd: f.rd, rs1: f.rs1, width: width_from_code(f.aux)? },
        op::ZIPPER => Instruction::Zipper { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::ZIPPER2 => Instruction::Zipper2 { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::TRUNC => Instruction::Trunc { rd: f.rd, rs1: f.rs1, eps_bits: f.imm as u64 },
        op::TMUL => Instruction::TMul { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::TDOT => Instruction::TDot { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::TACT => Instruction::TAct { rd: f.rd, rs1: f.rs1, func: act_from_code(f.aux)? },
        op::TSOFTMAX => Instruction::TSoftmax { rd: f.rd, rs1: f.rs1 },
        op::NTT => Instruction::Ntt { rd: f.rd, rs1: f.rs1 },
        op::INVNTT => Instruction::InvNtt { rd: f.rd, rs1: f.rs1 },
        op::POLYMUL => Instruction::PolyMul { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::MODRED => Instruction::ModRed { rd: f.rd, rs1: f.rs1, modulus: f.imm as u64 },
        op::POLYADD => Instruction::PolyAdd { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::ENTROPY => Instruction::Entropy { rd: f.rd, rs1: f.rs1 },
        op::HAMMING => Instruction::Hamming { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::POPCNT => Instruction::PopCnt { rd: f.rd, rs1: f.rs1 },
        op::QRAND => Instruction::QRand { rd: f.rd },
        op::CADD => Instruction::CAdd { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::CSUB => Instruction::CSub { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::CMUL => Instruction::CMul { rd: f.rd, rs1: f.rs1, rs2: f.rs2 },
        op::CCONJ => Instruction::CConj { rd: f.rd, rs1: f.rs1 },
        op::CNORM => Instruction::CNorm { rd: f.rd, rs1: f.rs1 },
        op::CMAG => Instruction::CMag { rd: f.rd, rs1: f.rs1 },
        op::ECALL => Instruction::Ecall,
        op::HALT => Instruction::Halt,
        op::NOP => Instruction::Nop,
        op::FENCE => Instruction::Fence,
        op::CSRR => Instruction::CsrR { rd: f.rd, csr: f.aux as u16 },
        op::CSRW => Instruction::CsrW { csr: f.aux as u16, rs1: f.rs1 },
        op::MV => Instruction::Mv { rd: f.rd, rs1: f.rs1 },
        op::LI => Instruction::Li { rd: f.rd, imm: f.imm },
        op::QALLOC => Instruction::QAlloc { n: f.aux as u8 },
        op::QG1 => Instruction::QG1 {
            gate: crate::qpu::QGate::from_code(f.aux as u8).ok_or_else(|| format!("invalid gate code {}", f.aux))?,
            q: f.rs1,
        },
        op::QROT => Instruction::QRot {
            axis: crate::qpu::QAxis::from_code(f.aux as u8).ok_or_else(|| format!("invalid axis code {}", f.aux))?,
            q: f.rs1,
            rs: f.rs2,
        },
        op::QCX => Instruction::QCx { c: f.rs1, t: f.rs2 },
        op::QCZ => Instruction::QCz { a: f.rs1, b: f.rs2 },
        op::QMEAS => Instruction::QMeas { rd: f.rd, q: f.rs1 },
        op::QRESET => Instruction::QReset { q: f.rs1 },
        op::QSEED => Instruction::QSeed { rs: f.rs1 },
        op::QAMP => Instruction::QAmp { rd: f.rd, rs: f.rs1 },
        other => return Err(format!("unknown opcode 0x{:02x}", other)),
    })
}

// ─── Object file serialisation ──────────────────────────────────────────────

/// Serialise a program into the `UBIT` object format.
pub fn write_object(obj: &Object) -> Vec<u8> {
    let mut out = Vec::with_capacity(HEADER_BYTES + obj.code.len() * INSTRUCTION_BYTES);
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // reserved
    out.extend_from_slice(&obj.entry_point.to_le_bytes());
    out.extend_from_slice(&(obj.code.len() as u64).to_le_bytes());
    out.extend_from_slice(&(obj.data.len() as u64).to_le_bytes());

    for inst in &obj.code {
        out.extend_from_slice(&encode_instruction(inst));
    }
    for (addr, bytes) in &obj.data {
        out.extend_from_slice(&addr.to_le_bytes());
        out.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        out.extend_from_slice(bytes);
    }
    out
}

/// Parse an `UBIT` object file. Every length is validated against the actual
/// buffer before it is used, so a truncated or hostile file yields an error
/// rather than a panic.
pub fn read_object(buf: &[u8]) -> Result<Object, String> {
    if buf.len() < HEADER_BYTES {
        return Err(format!(
            "not a Unibit object: {} bytes is shorter than the {}-byte header",
            buf.len(),
            HEADER_BYTES
        ));
    }
    if buf[0..4] != MAGIC {
        return Err(format!(
            "bad magic: expected {:?}, found {:?}",
            MAGIC,
            &buf[0..4]
        ));
    }
    let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
    if version != VERSION {
        return Err(format!(
            "unsupported object version {} (this build reads version {})",
            version, VERSION
        ));
    }

    let entry_point = u64::from_le_bytes(buf[8..16].try_into().unwrap());
    let code_len = u64::from_le_bytes(buf[16..24].try_into().unwrap()) as usize;
    let data_len = u64::from_le_bytes(buf[24..32].try_into().unwrap()) as usize;

    let code_bytes = code_len
        .checked_mul(INSTRUCTION_BYTES)
        .ok_or_else(|| format!("code length {} overflows", code_len))?;
    if buf.len() < HEADER_BYTES + code_bytes {
        return Err(format!(
            "truncated code section: header declares {} instructions ({} bytes), {} bytes available",
            code_len,
            code_bytes,
            buf.len() - HEADER_BYTES
        ));
    }

    let mut code = Vec::with_capacity(code_len);
    for i in 0..code_len {
        let at = HEADER_BYTES + i * INSTRUCTION_BYTES;
        code.push(
            decode_instruction(&buf[at..at + INSTRUCTION_BYTES])
                .map_err(|e| format!("instruction {}: {}", i, e))?,
        );
    }

    let mut cursor = HEADER_BYTES + code_bytes;
    let mut data = Vec::with_capacity(data_len);
    for i in 0..data_len {
        if cursor + 16 > buf.len() {
            return Err(format!("truncated data segment header {}", i));
        }
        let addr = u64::from_le_bytes(buf[cursor..cursor + 8].try_into().unwrap());
        let len = u64::from_le_bytes(buf[cursor + 8..cursor + 16].try_into().unwrap()) as usize;
        cursor += 16;
        if cursor + len > buf.len() {
            return Err(format!(
                "truncated data segment {}: declares {} bytes, {} available",
                i,
                len,
                buf.len() - cursor
            ));
        }
        data.push((addr, buf[cursor..cursor + len].to_vec()));
        cursor += len;
    }

    Ok(Object { entry_point, code, data })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// One canonical instance of every instruction in the ISA. Distinct field
    /// values so a swapped rd/rs1/rs2 cannot round-trip by accident.
    fn every_instruction() -> Vec<Instruction> {
        vec![
            Instruction::Add { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Sub { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Mul { rd: 1, rs1: 2, rs2: 3 },
            Instruction::MulH { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Div { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Rem { rd: 1, rs1: 2, rs2: 3 },
            Instruction::And { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Or { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Xor { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Sll { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Srl { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Sra { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Slt { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Sltu { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Addi { rd: 4, rs1: 5, imm: -1234 },
            Instruction::Andi { rd: 4, rs1: 5, imm: -1234 },
            Instruction::Ori { rd: 4, rs1: 5, imm: -1234 },
            Instruction::Xori { rd: 4, rs1: 5, imm: -1234 },
            Instruction::Slli { rd: 4, rs1: 5, imm: -1234 },
            Instruction::Srli { rd: 4, rs1: 5, imm: -1234 },
            Instruction::Srai { rd: 4, rs1: 5, imm: -1234 },
            Instruction::Slti { rd: 4, rs1: 5, imm: -1234 },
            Instruction::Sltiu { rd: 4, rs1: 5, imm: -1234 },
            Instruction::Lui { rd: 8, imm: -77 },
            Instruction::Ld { rd: 4, rs1: 5, offset: -1234 },
            Instruction::Lw { rd: 4, rs1: 5, offset: -1234 },
            Instruction::Lh { rd: 4, rs1: 5, offset: -1234 },
            Instruction::Lb { rd: 4, rs1: 5, offset: -1234 },
            Instruction::Lbu { rd: 4, rs1: 5, offset: -1234 },
            Instruction::Lq { rd: 4, rs1: 5, offset: -1234 },
            Instruction::Sd { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Sw { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Sh { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Sb { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Sq { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Beq { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Bne { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Blt { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Bge { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Bltu { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Bgeu { rs1: 6, rs2: 7, offset: 4321 },
            Instruction::Jal { rd: 8, offset: -77 },
            Instruction::Jalr { rd: 4, rs1: 5, offset: -1234 },
            Instruction::VAdd { rd: 12, rs1: 13, rs2: 14, width: Width::B16 },
            Instruction::VSub { rd: 12, rs1: 13, rs2: 14, width: Width::B16 },
            Instruction::VMul { rd: 12, rs1: 13, rs2: 14, width: Width::B16 },
            Instruction::VDot { rd: 12, rs1: 13, rs2: 14, width: Width::B16 },
            Instruction::VAnd { rd: 1, rs1: 2, rs2: 3 },
            Instruction::VOr { rd: 1, rs1: 2, rs2: 3 },
            Instruction::VXor { rd: 1, rs1: 2, rs2: 3 },
            Instruction::VNot { rd: 9, rs1: 10 },
            Instruction::VSplat { rd: 15, rs1: 16, width: Width::B32 },
            Instruction::VReduce { rd: 15, rs1: 16, width: Width::B32 },
            // The packed-float set. Step 0 of LYTH's ADR-0025 added six of these and none of
            // them reached this list, so their encoding was never round-tripped -- an opcode
            // typo in `decode` would have been found by a LYTH program failing to verify
            // rather than by the table that exists to find it.
            Instruction::VfAdd { rd: 5, rs1: 6, rs2: 7 },
            Instruction::VfSub { rd: 5, rs1: 6, rs2: 7 },
            Instruction::VfMul { rd: 5, rs1: 6, rs2: 7 },
            Instruction::VfMa { rd: 5, rs1: 6, rs2: 7 },
            Instruction::VfMax { rd: 5, rs1: 6, rs2: 7 },
            Instruction::VfMin { rd: 5, rs1: 6, rs2: 7 },
            Instruction::VfReduce { rd: 15, rs1: 16 },
            Instruction::Zipper { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Zipper2 { rd: 4, rs1: 5, rs2: 6 },
            Instruction::Trunc { rd: 19, rs1: 20, eps_bits: 0xDEAD_BEEF_CAFE_F00D },
            Instruction::TMul { rd: 1, rs1: 2, rs2: 3 },
            Instruction::TDot { rd: 1, rs1: 2, rs2: 3 },
            Instruction::TAct { rd: 17, rs1: 18, func: ActivationFn::SiLU },
            Instruction::TSoftmax { rd: 9, rs1: 10 },
            Instruction::Ntt { rd: 9, rs1: 10 },
            Instruction::InvNtt { rd: 9, rs1: 10 },
            Instruction::PolyMul { rd: 1, rs1: 2, rs2: 3 },
            Instruction::ModRed { rd: 19, rs1: 20, modulus: 0xDEAD_BEEF_CAFE_F00D },
            Instruction::PolyAdd { rd: 1, rs1: 2, rs2: 3 },
            Instruction::Entropy { rd: 9, rs1: 10 },
            Instruction::Hamming { rd: 1, rs1: 2, rs2: 3 },
            Instruction::PopCnt { rd: 9, rs1: 10 },
            Instruction::QRand { rd: 11 },
            Instruction::CAdd { rd: 1, rs1: 2, rs2: 3 },
            Instruction::CSub { rd: 1, rs1: 2, rs2: 3 },
            Instruction::CMul { rd: 1, rs1: 2, rs2: 3 },
            Instruction::CConj { rd: 9, rs1: 10 },
            Instruction::CNorm { rd: 9, rs1: 10 },
            Instruction::CMag { rd: 9, rs1: 10 },
            Instruction::Ecall,
            Instruction::Halt,
            Instruction::Nop,
            Instruction::Fence,
            Instruction::CsrR { rd: 21, csr: 0x803 },
            Instruction::CsrW { csr: 0x800, rs1: 22 },
            Instruction::Mv { rd: 9, rs1: 10 },
            Instruction::Li { rd: 8, imm: -77 },
        ]
    }

    #[test]
    fn test_encode_decode_covers_every_opcode() {
        for inst in every_instruction() {
            let bytes = encode_instruction(&inst);
            let back = decode_instruction(&bytes)
                .unwrap_or_else(|e| panic!("decoding {:?} failed: {}", inst, e));
            assert_eq!(back, inst, "round-trip changed {:?}", inst);
        }
    }

    #[test]
    fn test_every_opcode_is_distinct() {
        let mut seen = std::collections::HashMap::new();
        for inst in every_instruction() {
            let opcode = encode_instruction(&inst)[0];
            if let Some(prev) = seen.insert(opcode, format!("{:?}", inst)) {
                panic!("opcode {} shared by {:?} and {}", opcode, inst, prev);
            }
        }
        assert_eq!(seen.len(), every_instruction().len());
    }

    #[test]
    fn test_object_roundtrip() {
        let obj = Object {
            entry_point: 3,
            code: every_instruction(),
            data: vec![(0x1000, b"hello".to_vec()), (0x2000, vec![0xAA; 64])],
        };
        let bytes = write_object(&obj);
        assert_eq!(&bytes[0..4], &MAGIC);
        assert_eq!(read_object(&bytes).expect("read back"), obj);
    }

    #[test]
    fn test_rejects_bad_magic() {
        let mut bytes = write_object(&Object { entry_point: 0, code: vec![Instruction::Halt], data: vec![] });
        bytes[0] = b'X';
        assert!(read_object(&bytes).unwrap_err().contains("magic"));
    }

    #[test]
    fn test_rejects_truncated_file() {
        let bytes = write_object(&Object {
            entry_point: 0,
            code: vec![Instruction::Halt, Instruction::Nop],
            data: vec![],
        });
        for cut in [0, 8, HEADER_BYTES, HEADER_BYTES + 4, bytes.len() - 1] {
            assert!(read_object(&bytes[..cut]).is_err(), "accepted a file cut to {} bytes", cut);
        }
    }

    #[test]
    fn test_rejects_unknown_opcode() {
        let mut record = encode_instruction(&Instruction::Halt);
        record[0] = 0xFE;
        assert!(decode_instruction(&record).unwrap_err().contains("unknown opcode"));
    }

    #[test]
    fn test_rejects_invalid_aux_codes() {
        let mut record = encode_instruction(&Instruction::VAdd { rd: 1, rs1: 2, rs2: 3, width: Width::B8 });
        record[4] = 9;
        assert!(decode_instruction(&record).unwrap_err().contains("width"));

        let mut record = encode_instruction(&Instruction::TAct { rd: 1, rs1: 2, func: ActivationFn::GeLU });
        record[4] = 9;
        assert!(decode_instruction(&record).unwrap_err().contains("activation"));
    }
}
