//! F3 of ADR-0003: MTLB's quantum instructions against an independent oracle.
//!
//! tools/qpu_oracle.py writes 200 random circuits (seed 20260929, n 1..12,
//! depth 40) and their final states from Qiskit's quantum_info.Statevector.
//! Here every circuit becomes an MTLB assembly program: assembled, encoded to
//! the object format and decoded back, run on the CPU, and the whole state
//! read out through the ISA (`qamp` + `sq`). Pass: every amplitude within 1e-12.
//! The per-circuit result is written to experiments/u-qpu/f3_result.txt.

use std::fmt::Write as _;

use unibit::assembler::Assembler;
use unibit::binary::{self, Object};
use unibit::cpu::Cpu;

const OUT: u64 = 0x4_0000;
const TOL: f64 = 1e-12;

fn dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments").join("u-qpu")
}

fn hex_f64(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("hex f64"))
}

/// One circuit as an MTLB program.
fn program(n: u32, gates: &[Vec<String>]) -> String {
    let mut s = String::from("        .text\n        .global _start\n_start:\n");
    let _ = writeln!(s, "        qalloc  {n}");
    for g in gates {
        match g[0].as_str() {
            "rx" | "ry" | "rz" => {
                let bits = u64::from_str_radix(&g[2], 16).unwrap() as i64;
                let _ = writeln!(s, "        li      t2, {bits}");
                let _ = writeln!(s, "        qrot.{}  {}, t2", &g[0][1..], g[1]);
            }
            "cx" => {
                let _ = writeln!(s, "        qcx     {}, {}", g[1], g[2]);
            }
            "cz" => {
                let _ = writeln!(s, "        qcz     {}, {}", g[1], g[2]);
            }
            fixed => {
                let _ = writeln!(s, "        qg1.{fixed}  {}", g[1]);
            }
        }
    }
    let _ = write!(
        s,
        "        li      a3, {OUT}\n        li      t0, 0\n        li      t1, {}\n\
         readout:\n        qamp    t2, t0\n        sq      t2, 0(a3)\n        addi    a3, a3, 32\n\
                 addi    t0, t0, 1\n        blt     t0, t1, readout\n        halt\n",
        1u64 << n
    );
    s
}

#[test]
fn f3_every_random_circuit_matches_qiskit_through_the_isa() {
    let circuits = std::fs::read_to_string(dir().join("f3_circuits.txt"))
        .expect("run `python tools/qpu_oracle.py` first (F3 needs its independent oracle)");
    let oracle = std::fs::read_to_string(dir().join("f3_qiskit.txt")).expect("f3_qiskit.txt");
    let mut states = oracle.lines();

    let mut lines = circuits.lines();
    let mut report = String::from("# circuit n gates max_abs_diff (ADR-0003 F3, tolerance 1e-12)\n");
    let (mut worst, mut count, mut amps_checked) = (0.0f64, 0usize, 0usize);
    while let Some(head) = lines.next() {
        let h: Vec<&str> = head.split_whitespace().collect();
        assert_eq!(h[0], "circuit");
        let n: u32 = h[1].parse().unwrap();
        let depth: usize = h[2].parse().unwrap();
        let gates: Vec<Vec<String>> =
            (0..depth).map(|_| lines.next().unwrap().split_whitespace().map(str::to_string).collect()).collect();

        let prog = Assembler::new().assemble(&program(n, &gates)).expect("assembles");
        let obj = binary::read_object(&binary::write_object(&Object {
            entry_point: prog.entry_point,
            code: prog.instructions,
            data: prog.data_segment,
        }))
        .expect("round-trips");
        let mut cpu = Cpu::new(1024 * 1024);
        cpu.reset(obj.entry_point);
        cpu.run_program(&obj.code, 1_000_000).expect("runs");

        let want: Vec<(f64, f64)> = states
            .next()
            .expect("an oracle state per circuit")
            .split_whitespace()
            .map(|p| (hex_f64(&p[..16]), hex_f64(&p[16..])))
            .collect();
        assert_eq!(want.len(), 1usize << n);
        let mut d: f64 = 0.0;
        for (i, (re, im)) in want.iter().enumerate() {
            let l = cpu.memory.load_256(OUT + 32 * i as u64).unwrap();
            let (gr, gi) = (f64::from_bits(l[0]), f64::from_bits(l[1]));
            d = d.max((gr - re).abs()).max((gi - im).abs());
        }
        let _ = writeln!(report, "{count} {n} {depth} {d:.3e}");
        worst = worst.max(d);
        amps_checked += want.len();
        count += 1;
        assert!(d <= TOL, "circuit {count} (n = {n}): max |d amplitude| = {d:e}");
    }
    let _ = writeln!(report, "# {count} circuits, {amps_checked} amplitudes, worst {worst:.3e}");
    std::fs::write(dir().join("f3_result.txt"), &report).unwrap();
    println!("F3: {count} circuits, {amps_checked} amplitudes through the ISA, worst |d| = {worst:.3e} (tolerance {TOL:e})");
    assert_eq!(count, 200);
}
