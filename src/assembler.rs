// ============================================================================
// Unibit — Two-Pass Assembler & Symbol Linker
// ============================================================================
//
// Converts Unibit assembly (.uasm) source code into executable instruction
// streams and memory data payloads.
//
// Features:
//   - Label resolution for branches, jumps, and data references
//   - Rich pseudo-instructions (li, la, mv, j, call, ret, nop, halt)
//   - Suffix parsing (.b, .h, .w, .d) for SIMD vector instructions
//   - Full .data section parsing (.ascii, .asciiz, .byte, .word, .dword, .quad, .zero)
//   - Helpful error messages with line numbers
//
// ============================================================================

use std::collections::HashMap;
use crate::isa::*;

pub struct AssembledProgram {
    pub instructions: Vec<Instruction>,
    pub data_segment: Vec<(u64, Vec<u8>)>,
    pub entry_point: u64,
    pub symbols: HashMap<String, u64>,
}

pub struct Assembler {
    symbols: HashMap<String, u64>,
    data_bytes: Vec<(u64, Vec<u8>)>,
    current_data_addr: u64,
}

impl Default for Assembler {
    fn default() -> Self {
        Self::new()
    }
}

impl Assembler {
    pub fn new() -> Self {
        Self {
            symbols: HashMap::new(),
            data_bytes: Vec::new(),
            current_data_addr: 0x10000, // Data section starts at 64 KiB
        }
    }

    /// Assemble full source text into executable program
    pub fn assemble(&mut self, source: &str) -> Result<AssembledProgram, String> {
        let lines: Vec<&str> = source.lines().collect();

        // ─── PASS 1: Symbol Collection & Layout ──────────────────────────────
        let mut in_text = true;
        let mut code_addr = 0u64;

        for (line_no, raw_line) in lines.iter().enumerate() {
            let line = sanitize_line(raw_line);
            if line.is_empty() {
                continue;
            }

            if line.starts_with('.') {
                let parts: Vec<&str> = line.split_whitespace().collect();
                match parts[0] {
                    ".text" => in_text = true,
                    ".data" => in_text = false,
                    ".global" | ".globl" => {}
                    ".ascii" | ".asciiz" | ".byte" | ".half" | ".word" | ".dword" | ".quad" | ".zero" | ".align" => {
                        if in_text {
                            return Err(format!("line {}: data directive '{}' found inside .text section", line_no + 1, parts[0]));
                        }
                    }
                    _ => {}
                }
                continue;
            }

            // Check for labels (e.g. `_start:`, `loop:`, `msg:`)
            let mut remaining = line.as_str();
            if let Some(colon_pos) = remaining.find(':') {
                let label_name = remaining[..colon_pos].trim().to_string();
                if in_text {
                    self.symbols.insert(label_name, code_addr);
                } else {
                    self.symbols.insert(label_name, self.current_data_addr);
                }
                remaining = remaining[colon_pos + 1..].trim();
            }

            if remaining.is_empty() {
                continue;
            }

            if in_text {
                // Every instruction advances instruction counter by 1
                code_addr += 1;
            } else {
                // Parse data directives
                self.process_data_directive(line_no + 1, remaining)?;
            }
        }

        // ─── PASS 2: Code Generation & Reference Linking ─────────────────────
        let mut instructions = Vec::new();
        in_text = true;
        code_addr = 0;

        for (line_no, raw_line) in lines.iter().enumerate() {
            let line = sanitize_line(raw_line);
            if line.is_empty() {
                continue;
            }

            if line.starts_with('.') {
                let parts: Vec<&str> = line.split_whitespace().collect();
                match parts[0] {
                    ".text" => in_text = true,
                    ".data" => in_text = false,
                    _ => {}
                }
                continue;
            }

            let mut remaining = line.as_str();
            if let Some(colon_pos) = remaining.find(':') {
                remaining = remaining[colon_pos + 1..].trim();
            }

            if remaining.is_empty() || !in_text {
                continue;
            }

            let inst = self.parse_instruction(line_no + 1, code_addr, remaining)?;
            instructions.push(inst);
            code_addr += 1;
        }

        let entry_point = *self.symbols.get("_start").or_else(|| self.symbols.get("main")).unwrap_or(&0);

        Ok(AssembledProgram {
            instructions,
            data_segment: self.data_bytes.clone(),
            entry_point,
            symbols: self.symbols.clone(),
        })
    }

    // ─── Parse Data Directives ───────────────────────────────────────────────

    fn process_data_directive(&mut self, line_no: usize, text: &str) -> Result<(), String> {
        let parts: Vec<&str> = text.splitn(2, |c: char| c.is_whitespace()).collect();
        let directive = parts[0];
        let args = if parts.len() > 1 { parts[1].trim() } else { "" };

        match directive {
            ".ascii" | ".asciiz" => {
                let is_z = directive == ".asciiz";
                let str_content = extract_quoted_string(args)
                    .ok_or_else(|| format!("line {}: invalid string literal in {}", line_no, directive))?;
                let mut bytes = unescape_string(&str_content);
                if is_z {
                    bytes.push(0);
                }
                let addr = self.current_data_addr;
                self.current_data_addr += bytes.len() as u64;
                self.data_bytes.push((addr, bytes));
            }
            ".byte" => {
                let mut bytes = Vec::new();
                for num_str in args.split(',') {
                    let val = parse_immediate(num_str.trim(), &self.symbols, 0)? as u8;
                    bytes.push(val);
                }
                let addr = self.current_data_addr;
                self.current_data_addr += bytes.len() as u64;
                self.data_bytes.push((addr, bytes));
            }
            ".word" => {
                let mut bytes = Vec::new();
                for num_str in args.split(',') {
                    let val = parse_immediate(num_str.trim(), &self.symbols, 0)? as u32;
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
                let addr = self.current_data_addr;
                self.current_data_addr += bytes.len() as u64;
                self.data_bytes.push((addr, bytes));
            }
            ".dword" => {
                let mut bytes = Vec::new();
                for num_str in args.split(',') {
                    let val = parse_immediate(num_str.trim(), &self.symbols, 0)? as u64;
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
                let addr = self.current_data_addr;
                self.current_data_addr += bytes.len() as u64;
                self.data_bytes.push((addr, bytes));
            }
            ".zero" => {
                let count = parse_immediate(args, &self.symbols, 0)? as usize;
                let bytes = vec![0u8; count];
                let addr = self.current_data_addr;
                self.current_data_addr += count as u64;
                self.data_bytes.push((addr, bytes));
            }
            ".align" => {
                let align = parse_immediate(args, &self.symbols, 0)? as u64;
                if align > 0 {
                    let rem = self.current_data_addr % align;
                    if rem != 0 {
                        self.current_data_addr += align - rem;
                    }
                }
            }
            _ => return Err(format!("line {}: unknown data directive '{}'", line_no, directive)),
        }
        Ok(())
    }

    // ─── Parse Instruction ───────────────────────────────────────────────────

    fn parse_instruction(&self, line_no: usize, current_pc: u64, text: &str) -> Result<Instruction, String> {
        let (mnemonic, args_str) = match text.split_once(|c: char| c.is_whitespace()) {
            Some((m, a)) => (m.trim(), a.trim()),
            None => (text.trim(), ""),
        };

        let mnemonic_lower = mnemonic.to_lowercase();
        let args: Vec<&str> = if args_str.is_empty() {
            Vec::new()
        } else {
            args_str.split(',').map(|s| s.trim()).collect()
        };

        let parse_reg = |idx: usize, name: &str| -> Result<u8, String> {
            args.get(idx)
                .and_then(|&s| parse_register(s))
                .ok_or_else(|| format!("line {}: invalid or missing register argument {} for {}", line_no, idx + 1, name))
        };

        let parse_imm = |idx: usize, name: &str| -> Result<i64, String> {
            args.get(idx)
                .ok_or_else(|| format!("line {}: missing immediate argument {} for {}", line_no, idx + 1, name))
                .and_then(|&s| parse_immediate(s, &self.symbols, current_pc))
        };

        // A qubit index is an immediate in 0..MAX_QUBITS (ADR-0003).
        let qubit = |v: i64| -> Result<u8, String> {
            if (0..crate::qpu::MAX_QUBITS as i64).contains(&v) { Ok(v as u8) }
            else { Err(format!("line {}: qubit {} is outside 0..{}", line_no, v, crate::qpu::MAX_QUBITS)) }
        };

        // Helper for memory operands like `8(sp)` or `0(a0)`
        let parse_mem_op = |idx: usize| -> Result<(u8, i64), String> {
            let s = args.get(idx).ok_or_else(|| format!("line {}: missing memory operand", line_no))?;
            parse_memory_address(s, &self.symbols, current_pc)
        };

        match mnemonic_lower.as_str() {
            // ─── Scalar ALU ──────────────────────────────────────────────────
            "add"  => Ok(Instruction::Add  { rd: parse_reg(0, "add")?, rs1: parse_reg(1, "add")?, rs2: parse_reg(2, "add")? }),
            "sub"  => Ok(Instruction::Sub  { rd: parse_reg(0, "sub")?, rs1: parse_reg(1, "sub")?, rs2: parse_reg(2, "sub")? }),
            "mul"  => Ok(Instruction::Mul  { rd: parse_reg(0, "mul")?, rs1: parse_reg(1, "mul")?, rs2: parse_reg(2, "mul")? }),
            "mulh" => Ok(Instruction::MulH { rd: parse_reg(0, "mulh")?, rs1: parse_reg(1, "mulh")?, rs2: parse_reg(2, "mulh")? }),
            "div"  => Ok(Instruction::Div  { rd: parse_reg(0, "div")?, rs1: parse_reg(1, "div")?, rs2: parse_reg(2, "div")? }),
            "rem"  => Ok(Instruction::Rem  { rd: parse_reg(0, "rem")?, rs1: parse_reg(1, "rem")?, rs2: parse_reg(2, "rem")? }),
            "and"  => Ok(Instruction::And  { rd: parse_reg(0, "and")?, rs1: parse_reg(1, "and")?, rs2: parse_reg(2, "and")? }),
            "or"   => Ok(Instruction::Or   { rd: parse_reg(0, "or")?,  rs1: parse_reg(1, "or")?,  rs2: parse_reg(2, "or")?  }),
            "xor"  => Ok(Instruction::Xor  { rd: parse_reg(0, "xor")?, rs1: parse_reg(1, "xor")?, rs2: parse_reg(2, "xor")? }),
            "sll"  => Ok(Instruction::Sll  { rd: parse_reg(0, "sll")?, rs1: parse_reg(1, "sll")?, rs2: parse_reg(2, "sll")? }),
            "srl"  => Ok(Instruction::Srl  { rd: parse_reg(0, "srl")?, rs1: parse_reg(1, "srl")?, rs2: parse_reg(2, "srl")? }),
            "sra"  => Ok(Instruction::Sra  { rd: parse_reg(0, "sra")?, rs1: parse_reg(1, "sra")?, rs2: parse_reg(2, "sra")? }),
            "slt"  => Ok(Instruction::Slt  { rd: parse_reg(0, "slt")?, rs1: parse_reg(1, "slt")?, rs2: parse_reg(2, "slt")? }),
            "sltu" => Ok(Instruction::Sltu { rd: parse_reg(0, "sltu")?, rs1: parse_reg(1, "sltu")?, rs2: parse_reg(2, "sltu")? }),

            // ─── Scalar Immediate ────────────────────────────────────────────
            "addi"  => Ok(Instruction::Addi  { rd: parse_reg(0, "addi")?, rs1: parse_reg(1, "addi")?, imm: parse_imm(2, "addi")? }),
            "andi"  => Ok(Instruction::Andi  { rd: parse_reg(0, "andi")?, rs1: parse_reg(1, "andi")?, imm: parse_imm(2, "andi")? }),
            "ori"   => Ok(Instruction::Ori   { rd: parse_reg(0, "ori")?,  rs1: parse_reg(1, "ori")?,  imm: parse_imm(2, "ori")?  }),
            "xori"  => Ok(Instruction::Xori  { rd: parse_reg(0, "xori")?, rs1: parse_reg(1, "xori")?, imm: parse_imm(2, "xori")? }),
            "slli"  => Ok(Instruction::Slli  { rd: parse_reg(0, "slli")?, rs1: parse_reg(1, "slli")?, imm: parse_imm(2, "slli")? }),
            "srli"  => Ok(Instruction::Srli  { rd: parse_reg(0, "srli")?, rs1: parse_reg(1, "srli")?, imm: parse_imm(2, "srli")? }),
            "srai"  => Ok(Instruction::Srai  { rd: parse_reg(0, "srai")?, rs1: parse_reg(1, "srai")?, imm: parse_imm(2, "srai")? }),
            "slti"  => Ok(Instruction::Slti  { rd: parse_reg(0, "slti")?, rs1: parse_reg(1, "slti")?, imm: parse_imm(2, "slti")? }),
            "sltiu" => Ok(Instruction::Sltiu { rd: parse_reg(0, "sltiu")?, rs1: parse_reg(1, "sltiu")?, imm: parse_imm(2, "sltiu")? }),
            "lui"   => Ok(Instruction::Lui   { rd: parse_reg(0, "lui")?, imm: parse_imm(1, "lui")? }),

            // ─── Memory ──────────────────────────────────────────────────────
            "ld"  => {
                let rd = parse_reg(0, "ld")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Ld { rd, rs1, offset })
            }
            "lw"  => {
                let rd = parse_reg(0, "lw")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Lw { rd, rs1, offset })
            }
            "lh"  => {
                let rd = parse_reg(0, "lh")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Lh { rd, rs1, offset })
            }
            "lb"  => {
                let rd = parse_reg(0, "lb")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Lb { rd, rs1, offset })
            }
            "lbu" => {
                let rd = parse_reg(0, "lbu")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Lbu { rd, rs1, offset })
            }
            "sd"  => {
                let rs2 = parse_reg(0, "sd")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Sd { rs1, rs2, offset })
            }
            "sw"  => {
                let rs2 = parse_reg(0, "sw")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Sw { rs1, rs2, offset })
            }
            "sh"  => {
                let rs2 = parse_reg(0, "sh")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Sh { rs1, rs2, offset })
            }
            "sb"  => {
                let rs2 = parse_reg(0, "sb")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Sb { rs1, rs2, offset })
            }
            "lq"  => {
                let rd = parse_reg(0, "lq")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Lq { rd, rs1, offset })
            }
            "sq"  => {
                let rs2 = parse_reg(0, "sq")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Sq { rs1, rs2, offset })
            }

            // ─── Branches ────────────────────────────────────────────────────
            "beq"  => Ok(Instruction::Beq  { rs1: parse_reg(0, "beq")?,  rs2: parse_reg(1, "beq")?,  offset: parse_branch_target(args.get(2), &self.symbols, current_pc, line_no)? }),
            "bne"  => Ok(Instruction::Bne  { rs1: parse_reg(0, "bne")?,  rs2: parse_reg(1, "bne")?,  offset: parse_branch_target(args.get(2), &self.symbols, current_pc, line_no)? }),
            "blt"  => Ok(Instruction::Blt  { rs1: parse_reg(0, "blt")?,  rs2: parse_reg(1, "blt")?,  offset: parse_branch_target(args.get(2), &self.symbols, current_pc, line_no)? }),
            "bge"  => Ok(Instruction::Bge  { rs1: parse_reg(0, "bge")?,  rs2: parse_reg(1, "bge")?,  offset: parse_branch_target(args.get(2), &self.symbols, current_pc, line_no)? }),
            "bltu" => Ok(Instruction::Bltu { rs1: parse_reg(0, "bltu")?, rs2: parse_reg(1, "bltu")?, offset: parse_branch_target(args.get(2), &self.symbols, current_pc, line_no)? }),
            "bgeu" => Ok(Instruction::Bgeu { rs1: parse_reg(0, "bgeu")?, rs2: parse_reg(1, "bgeu")?, offset: parse_branch_target(args.get(2), &self.symbols, current_pc, line_no)? }),

            // ─── Jumps ───────────────────────────────────────────────────────
            "jal" => {
                if args.len() == 1 {
                    // `jal target` -> `jal ra, target`
                    let offset = parse_branch_target(args.first(), &self.symbols, current_pc, line_no)?;
                    Ok(Instruction::Jal { rd: REG_RA, offset })
                } else {
                    let rd = parse_reg(0, "jal")?;
                    let offset = parse_branch_target(args.get(1), &self.symbols, current_pc, line_no)?;
                    Ok(Instruction::Jal { rd, offset })
                }
            }
            "jalr" => {
                let rd = parse_reg(0, "jalr")?;
                let (rs1, offset) = parse_mem_op(1)?;
                Ok(Instruction::Jalr { rd, rs1, offset })
            }

            // ─── Packed single-precision ─────────────────────────────────────
            //
            // No width suffix: f32 is 32 bits, so `vfadd.w` would be the only spelling and
            // `vfadd.b` would have to be rejected at parse time. A mnemonic that admits one
            // suffix is a mnemonic with no suffix.
            "vfadd" => Ok(Instruction::VfAdd { rd: parse_reg(0, "vfadd")?, rs1: parse_reg(1, "vfadd")?, rs2: parse_reg(2, "vfadd")? }),
            "vfsub" => Ok(Instruction::VfSub { rd: parse_reg(0, "vfsub")?, rs1: parse_reg(1, "vfsub")?, rs2: parse_reg(2, "vfsub")? }),
            "vfmul" => Ok(Instruction::VfMul { rd: parse_reg(0, "vfmul")?, rs1: parse_reg(1, "vfmul")?, rs2: parse_reg(2, "vfmul")? }),
            "vfma"  => Ok(Instruction::VfMa  { rd: parse_reg(0, "vfma")?,  rs1: parse_reg(1, "vfma")?,  rs2: parse_reg(2, "vfma")?  }),
            "vfmax" => Ok(Instruction::VfMax { rd: parse_reg(0, "vfmax")?, rs1: parse_reg(1, "vfmax")?, rs2: parse_reg(2, "vfmax")? }),
            "vfmin" => Ok(Instruction::VfMin { rd: parse_reg(0, "vfmin")?, rs1: parse_reg(1, "vfmin")?, rs2: parse_reg(2, "vfmin")? }),
            "vfreduce" => Ok(Instruction::VfReduce { rd: parse_reg(0, "vfreduce")?, rs1: parse_reg(1, "vfreduce")? }),

            // ─── Vector SIMD Instructions ────────────────────────────────────
            "vadd" | "vadd.b" | "vadd.h" | "vadd.w" | "vadd.d" => {
                let width = parse_width_suffix(&mnemonic_lower);
                Ok(Instruction::VAdd { rd: parse_reg(0, "vadd")?, rs1: parse_reg(1, "vadd")?, rs2: parse_reg(2, "vadd")?, width })
            }
            "vsub" | "vsub.b" | "vsub.h" | "vsub.w" | "vsub.d" => {
                let width = parse_width_suffix(&mnemonic_lower);
                Ok(Instruction::VSub { rd: parse_reg(0, "vsub")?, rs1: parse_reg(1, "vsub")?, rs2: parse_reg(2, "vsub")?, width })
            }
            "vmul" | "vmul.b" | "vmul.h" | "vmul.w" | "vmul.d" => {
                let width = parse_width_suffix(&mnemonic_lower);
                Ok(Instruction::VMul { rd: parse_reg(0, "vmul")?, rs1: parse_reg(1, "vmul")?, rs2: parse_reg(2, "vmul")?, width })
            }
            "vand" => Ok(Instruction::VAnd { rd: parse_reg(0, "vand")?, rs1: parse_reg(1, "vand")?, rs2: parse_reg(2, "vand")? }),
            "vor"  => Ok(Instruction::VOr  { rd: parse_reg(0, "vor")?,  rs1: parse_reg(1, "vor")?,  rs2: parse_reg(2, "vor")?  }),
            "vxor" => Ok(Instruction::VXor { rd: parse_reg(0, "vxor")?, rs1: parse_reg(1, "vxor")?, rs2: parse_reg(2, "vxor")? }),
            "vnot" => Ok(Instruction::VNot { rd: parse_reg(0, "vnot")?, rs1: parse_reg(1, "vnot")? }),
            "vdot" | "vdot.b" | "vdot.h" | "vdot.w" | "vdot.d" => {
                let width = parse_width_suffix(&mnemonic_lower);
                Ok(Instruction::VDot { rd: parse_reg(0, "vdot")?, rs1: parse_reg(1, "vdot")?, rs2: parse_reg(2, "vdot")?, width })
            }
            "vsplat" | "vsplat.b" | "vsplat.h" | "vsplat.w" | "vsplat.d" => {
                let width = parse_width_suffix(&mnemonic_lower);
                Ok(Instruction::VSplat { rd: parse_reg(0, "vsplat")?, rs1: parse_reg(1, "vsplat")?, width })
            }
            "vreduce" | "vreduce.b" | "vreduce.h" | "vreduce.w" | "vreduce.d" => {
                let width = parse_width_suffix(&mnemonic_lower);
                Ok(Instruction::VReduce { rd: parse_reg(0, "vreduce")?, rs1: parse_reg(1, "vreduce")?, width })
            }

            // ─── Tensor Network & MPS Instructions ──────────────────────
            "zipper" => Ok(Instruction::Zipper { rd: parse_reg(0, "zipper")?, rs1: parse_reg(1, "zipper")?, rs2: parse_reg(2, "zipper")? }),
            "zipper2" => Ok(Instruction::Zipper2 { rd: parse_reg(0, "zipper2")?, rs1: parse_reg(1, "zipper2")?, rs2: parse_reg(2, "zipper2")? }),
            "trunc"  => {
                let rd = parse_reg(0, "trunc")?;
                let rs1 = parse_reg(1, "trunc")?;
                let eps_bits = parse_imm(2, "trunc")? as u64;
                Ok(Instruction::Trunc { rd, rs1, eps_bits })
            }

            // ─── Complex Arithmetic (Quantum) ────────────────────────────────
            "cadd"  => Ok(Instruction::CAdd  { rd: parse_reg(0, "cadd")?, rs1: parse_reg(1, "cadd")?, rs2: parse_reg(2, "cadd")? }),
            "csub"  => Ok(Instruction::CSub  { rd: parse_reg(0, "csub")?, rs1: parse_reg(1, "csub")?, rs2: parse_reg(2, "csub")? }),
            "cmul"  => Ok(Instruction::CMul  { rd: parse_reg(0, "cmul")?, rs1: parse_reg(1, "cmul")?, rs2: parse_reg(2, "cmul")? }),
            "cconj" => Ok(Instruction::CConj { rd: parse_reg(0, "cconj")?, rs1: parse_reg(1, "cconj")? }),
            "cnorm" => Ok(Instruction::CNorm { rd: parse_reg(0, "cnorm")?, rs1: parse_reg(1, "cnorm")? }),
            "cmag"  => Ok(Instruction::CMag  { rd: parse_reg(0, "cmag")?,  rs1: parse_reg(1, "cmag")?  }),

            // ─── Information Theory ──────────────────────────────────────────
            "entropy" => Ok(Instruction::Entropy { rd: parse_reg(0, "entropy")?, rs1: parse_reg(1, "entropy")? }),
            "hamming" => Ok(Instruction::Hamming { rd: parse_reg(0, "hamming")?, rs1: parse_reg(1, "hamming")?, rs2: parse_reg(2, "hamming")? }),
            "popcnt"  => Ok(Instruction::PopCnt  { rd: parse_reg(0, "popcnt")?,  rs1: parse_reg(1, "popcnt")? }),
            "qrand"   => Ok(Instruction::QRand   { rd: parse_reg(0, "qrand")? }),

            // ─── Quantum coprocessor (U-QPU, ADR-0003). Qubits are immediates. ───
            "qalloc" => {
                let n = parse_imm(0, "qalloc")?;
                if !(1..=crate::qpu::MAX_QUBITS as i64).contains(&n) {
                    return Err(format!("line {}: qalloc: {} qubits is outside 1..={}", line_no, n, crate::qpu::MAX_QUBITS));
                }
                Ok(Instruction::QAlloc { n: n as u8 })
            }
            m if m.starts_with("qg1.") => {
                let gate = crate::qpu::QGate::from_name(&m[4..])
                    .ok_or_else(|| format!("line {}: unknown one-qubit gate `{}`", line_no, &m[4..]))?;
                Ok(Instruction::QG1 { gate, q: qubit(parse_imm(0, "qg1")?)? })
            }
            m if m.starts_with("qrot.") => {
                let axis = crate::qpu::QAxis::from_name(&m[5..])
                    .ok_or_else(|| format!("line {}: unknown rotation axis `{}`", line_no, &m[5..]))?;
                Ok(Instruction::QRot { axis, q: qubit(parse_imm(0, "qrot")?)?, rs: parse_reg(1, "qrot")? })
            }
            "qcx"    => Ok(Instruction::QCx    { c: qubit(parse_imm(0, "qcx")?)?, t: qubit(parse_imm(1, "qcx")?)? }),
            "qcz"    => Ok(Instruction::QCz    { a: qubit(parse_imm(0, "qcz")?)?, b: qubit(parse_imm(1, "qcz")?)? }),
            "qmeas"  => Ok(Instruction::QMeas  { rd: parse_reg(0, "qmeas")?, q: qubit(parse_imm(1, "qmeas")?)? }),
            "qreset" => Ok(Instruction::QReset { q: qubit(parse_imm(0, "qreset")?)? }),
            "qseed"  => Ok(Instruction::QSeed  { rs: parse_reg(0, "qseed")? }),
            "qamp"   => Ok(Instruction::QAmp   { rd: parse_reg(0, "qamp")?, rs: parse_reg(1, "qamp")? }),

            // ─── Post-Quantum Lattice Cryptography ───────────────────────────
            "ntt"     => Ok(Instruction::Ntt     { rd: parse_reg(0, "ntt")?, rs1: parse_reg(1, "ntt")? }),
            "invntt"  => Ok(Instruction::InvNtt  { rd: parse_reg(0, "invntt")?, rs1: parse_reg(1, "invntt")? }),
            "polymul" => Ok(Instruction::PolyMul { rd: parse_reg(0, "polymul")?, rs1: parse_reg(1, "polymul")?, rs2: parse_reg(2, "polymul")? }),
            "polyadd" => Ok(Instruction::PolyAdd { rd: parse_reg(0, "polyadd")?, rs1: parse_reg(1, "polyadd")?, rs2: parse_reg(2, "polyadd")? }),
            "modred"  => {
                let rd = parse_reg(0, "modred")?;
                let rs1 = parse_reg(1, "modred")?;
                let modulus = parse_imm(2, "modred")? as u64;
                Ok(Instruction::ModRed { rd, rs1, modulus })
            }

            // ─── Tensor & Neural Activations ─────────────────────────────────
            "tact" => {
                let rd = parse_reg(0, "tact")?;
                let rs1 = parse_reg(1, "tact")?;
                let func_str = args.get(2).ok_or_else(|| format!("line {}: missing activation function", line_no))?;
                let func = match func_str.to_lowercase().as_str() {
                    "relu" => ActivationFn::ReLU,
                    "sigmoid" => ActivationFn::Sigmoid,
                    "tanh" => ActivationFn::Tanh,
                    "gelu" => ActivationFn::GeLU,
                    "silu" => ActivationFn::SiLU,
                    _ => return Err(format!("line {}: unknown activation function '{}'", line_no, func_str)),
                };
                Ok(Instruction::TAct { rd, rs1, func })
            }
            "tsoftmax" => Ok(Instruction::TSoftmax { rd: parse_reg(0, "tsoftmax")?, rs1: parse_reg(1, "tsoftmax")? }),
            "tmul"     => Ok(Instruction::TMul     { rd: parse_reg(0, "tmul")?, rs1: parse_reg(1, "tmul")?, rs2: parse_reg(2, "tmul")? }),
            "tdot"     => Ok(Instruction::TDot     { rd: parse_reg(0, "tdot")?, rs1: parse_reg(1, "tdot")?, rs2: parse_reg(2, "tdot")? }),

            // ─── Pseudo-Instructions ─────────────────────────────────────────
            "li" => {
                let rd = parse_reg(0, "li")?;
                let imm = parse_imm(1, "li")?;
                Ok(Instruction::Li { rd, imm })
            }
            "la" => {
                let rd = parse_reg(0, "la")?;
                let label = args.get(1).ok_or_else(|| format!("line {}: missing label for la", line_no))?.trim();
                let addr = self.symbols.get(label)
                    .copied()
                    .ok_or_else(|| format!("line {}: unknown label '{}' in la", line_no, label))?;
                Ok(Instruction::Li { rd, imm: addr as i64 })
            }
            "mv" => {
                let rd = parse_reg(0, "mv")?;
                let rs1 = parse_reg(1, "mv")?;
                Ok(Instruction::Mv { rd, rs1 })
            }
            "j" => {
                let offset = parse_branch_target(args.first(), &self.symbols, current_pc, line_no)?;
                Ok(Instruction::Jal { rd: REG_ZERO, offset })
            }
            "call" => {
                let offset = parse_branch_target(args.first(), &self.symbols, current_pc, line_no)?;
                Ok(Instruction::Jal { rd: REG_RA, offset })
            }
            "ret" => {
                Ok(Instruction::Jalr { rd: REG_ZERO, rs1: REG_RA, offset: 0 })
            }
            "nop" => Ok(Instruction::Nop),
            "halt" => Ok(Instruction::Halt),
            "ecall" => Ok(Instruction::Ecall),
            "fence" => Ok(Instruction::Fence),

            // ─── CSR Operations ──────────────────────────────────────────────
            "csrr" => {
                let rd = parse_reg(0, "csrr")?;
                let csr = parse_csr_name(args.get(1).unwrap_or(&""))?;
                Ok(Instruction::CsrR { rd, csr })
            }
            "csrw" => {
                let csr = parse_csr_name(args.first().unwrap_or(&""))?;
                let rs1 = parse_reg(1, "csrw")?;
                Ok(Instruction::CsrW { csr, rs1 })
            }

            _ => Err(format!("line {}: unknown instruction '{}'", line_no, mnemonic)),
        }
    }
}

// ─── Parsing Helpers ─────────────────────────────────────────────────────────

fn sanitize_line(line: &str) -> String {
    let mut s = line.trim();
    if let Some(pos) = s.find(';') {
        s = s[..pos].trim();
    }
    if let Some(pos) = s.find("//") {
        s = s[..pos].trim();
    }
    s.to_string()
}

fn parse_width_suffix(mnemonic: &str) -> Width {
    if mnemonic.ends_with(".b") {
        Width::B8
    } else if mnemonic.ends_with(".h") {
        Width::B16
    } else if mnemonic.ends_with(".w") {
        Width::B32
    } else {
        Width::B64
    }
}

fn parse_immediate(text: &str, symbols: &HashMap<String, u64>, _pc: u64) -> Result<i64, String> {
    let t = text.trim();
    if let Some(val) = symbols.get(t) {
        return Ok(*val as i64);
    }
    if t.starts_with("0x") || t.starts_with("0X") {
        // Parse as u64 and reinterpret: 0xBFF0000000000000 (the f64 bit pattern
        // of -1.0) exceeds i64::MAX and would otherwise be rejected outright.
        return u64::from_str_radix(&t[2..], 16)
            .map(|v| v as i64)
            .map_err(|e| format!("invalid hex number '{}': {}", t, e));
    }
    if t.starts_with("0b") || t.starts_with("0B") {
        return i64::from_str_radix(&t[2..], 2)
            .map_err(|e| format!("invalid binary number '{}': {}", t, e));
    }
    if t.starts_with('\'') && t.ends_with('\'') && t.len() >= 3 {
        let ch = t.chars().nth(1).unwrap();
        return Ok(ch as i64);
    }
    t.parse::<i64>().map_err(|e| format!("cannot parse immediate '{}': {}", t, e))
}

fn parse_branch_target(arg: Option<&&str>, symbols: &HashMap<String, u64>, pc: u64, line_no: usize) -> Result<i64, String> {
    let s = arg.ok_or_else(|| format!("line {}: missing branch target", line_no))?.trim();
    if let Some(&target_pc) = symbols.get(s) {
        // Offset relative to current PC
        Ok(target_pc as i64 - pc as i64)
    } else {
        parse_immediate(s, symbols, pc)
    }
}

fn parse_memory_address(s: &str, symbols: &HashMap<String, u64>, pc: u64) -> Result<(u8, i64), String> {
    let s = s.trim();
    if let Some(open) = s.find('(') {
        if let Some(close) = s.find(')') {
            let offset_str = &s[..open].trim();
            let reg_str = &s[open + 1..close].trim();
            let reg = parse_register(reg_str).ok_or_else(|| format!("invalid register in address: '{}'", reg_str))?;
            let offset = if offset_str.is_empty() {
                0
            } else {
                parse_immediate(offset_str, symbols, pc)?
            };
            return Ok((reg, offset));
        }
    }
    // Direct register `a0` with 0 offset
    if let Some(reg) = parse_register(s) {
        return Ok((reg, 0));
    }
    Err(format!("invalid memory operand format: '{}' (expected 'offset(reg)')", s))
}

fn parse_csr_name(s: &str) -> Result<u16, String> {
    match s.trim().to_lowercase().as_str() {
        "cycle" => Ok(csr::CYCLE),
        "instret" => Ok(csr::INSTRET),
        "entropy" | "entropy_acc" => Ok(csr::ENTROPY_ACC),
        "temp" | "temp_k" => Ok(csr::TEMP_K),
        "landauer" => Ok(csr::LANDAUER),
        _ => {
            if let Some(hex) = s.strip_prefix("0x") {
                u16::from_str_radix(hex, 16).map_err(|e| format!("invalid CSR hex: {}", e))
            } else {
                s.parse::<u16>().map_err(|e| format!("unknown CSR name '{}': {}", s, e))
            }
        }
    }
}

fn extract_quoted_string(s: &str) -> Option<String> {
    let first = s.find('"')?;
    let last = s.rfind('"')?;
    if first < last {
        Some(s[first + 1..last].to_string())
    } else {
        None
    }
}

/// Appends a char as its UTF-8 encoding. Using `c as u8` here would truncate
/// every non-ASCII char to its low byte (box-drawing U+2554 -> 0x54 = 'T').
fn push_utf8(out: &mut Vec<u8>, c: char) {
    let mut buf = [0u8; 4];
    out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
}

fn unescape_string(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push(b'\n'),
                Some('r') => out.push(b'\r'),
                Some('t') => out.push(b'\t'),
                Some('\\') => out.push(b'\\'),
                Some('0') => out.push(0),
                Some('"') => out.push(b'"'),
                Some(other) => push_utf8(&mut out, other),
                None => out.push(b'\\'),
            }
        } else {
            push_utf8(&mut out, c);
        }
    }
    out
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assemble_simple() {
        let asm_code = r#"
            .text
            .global _start
        _start:
            addi a0, zero, 42
            addi a7, zero, 1
            ecall
            halt
        "#;
        let mut assembler = Assembler::new();
        let program = assembler.assemble(asm_code).unwrap();
        assert_eq!(program.instructions.len(), 4);
    }

    #[test]
    fn test_assemble_loop() {
        let asm_code = r#"
            .text
        _start:
            addi t0, zero, 10
        loop:
            beq  t0, zero, done
            addi t0, t0, -1
            j    loop
        done:
            halt
        "#;
        let mut assembler = Assembler::new();
        let program = assembler.assemble(asm_code).unwrap();
        assert_eq!(program.instructions.len(), 5);
    }
}
