//! ADR-0005: QGPU's fidelity bound under real truncation (ADR-0004 G3, again,
//! not vacuous). Brickwork circuits, n = 10, 12 layers, chi = 4, seed 20260931,
//! each run through the ISA on the MPS and on the state vector.
//!   H1 >= 95 of 100 truncate; H2 F >= prod(1 - eps) - 1e-9 (the reported bound);
//!   H3 F >= cos^2(sum theta) - 1e-9 where sum theta < pi/2 (the proven bound).
//! Measured 2026-09-29: H1 100/100, H2 FAILED (53/100 over-claimed), H3 held.
//! As pre-registered, `qtrunc` now reports the proven bound and the product is
//! `qfest`, an estimate. H2's failure is kept below as a locked regression.

use std::fmt::Write as _;

use unibit::assembler::Assembler;
use unibit::binary::{self, Object};
use unibit::cpu::Cpu;
use unibit::qpu::{Backend, QRng, C64};

const OUT: u64 = 0x4_0000;

fn run(src: &str) -> Cpu {
    let prog = Assembler::new().assemble(src).expect("assembles");
    let obj = binary::read_object(&binary::write_object(&Object {
        entry_point: prog.entry_point,
        code: prog.instructions,
        data: prog.data_segment,
    }))
    .expect("round-trips");
    let mut cpu = Cpu::new(1024 * 1024);
    cpu.reset(obj.entry_point);
    cpu.run_program(&obj.code, 5_000_000).expect("runs");
    cpu
}

/// One brickwork circuit as MTLB assembly (without allocation and readout).
fn brickwork(rng: &mut QRng, n: u32, layers: u32) -> String {
    const ONE: [&str; 11] = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz"];
    let mut s = String::new();
    for layer in 0..layers {
        for q in 0..n {
            let g = ONE[(rng.next_u64() % 11) as usize];
            if let Some(axis) = g.strip_prefix('r') {
                let t = rng.uniform() * 2.0 * std::f64::consts::PI;
                let _ = writeln!(s, "        li      t2, {}\n        qrot.{}  {q}, t2", t.to_bits() as i64, axis);
            } else {
                let _ = writeln!(s, "        qg1.{g}  {q}");
            }
        }
        let mut i = layer % 2;
        while i + 1 < n {
            let _ = writeln!(s, "        qcx     {i}, {}", i + 1);
            i += 2;
        }
    }
    s
}

fn program(alloc: &str, n: u32, body: &str) -> String {
    format!(
        "        .text\n        .global _start\n_start:\n        {alloc}\n{body}\
         \x20       li      a3, {OUT}\n        li      t0, 0\n        li      t1, {}\n\
         readout:\n        qamp    t2, t0\n        sq      t2, 0(a3)\n        addi    a3, a3, 32\n\
         \x20       addi    t0, t0, 1\n        blt     t0, t1, readout\n        qfest   t2\n        sq      t2, 0(a3)\n        qtrunc  t2\n        sq      t2, 32(a3)\n        halt\n",
        1u64 << n
    )
}

/// Amplitudes, the product estimate (qfest) and the proven bound (qtrunc).
fn readout(cpu: &mut Cpu, n: u32) -> (Vec<C64>, f64, f64) {
    let d = 1usize << n;
    let amps = (0..d)
        .map(|i| {
            let l = cpu.memory.load_256(OUT + 32 * i as u64).unwrap();
            C64::new(f64::from_bits(l[0]), f64::from_bits(l[1]))
        })
        .collect();
    let est = f64::from_bits(cpu.memory.load_256(OUT + 32 * d as u64).unwrap()[0]);
    let proven = f64::from_bits(cpu.memory.load_256(OUT + 32 * (d as u64 + 1)).unwrap()[0]);
    (amps, est, proven)
}

#[test]
fn h1_h2_h3_the_bound_under_heavy_truncation() {
    let (n, layers, chi) = (10u32, 12u32, 4u32);
    let mut rng = QRng::new(20260931);
    let mut report = String::from("# circuit F_true F_bound(prod) F_rigorous truncations max_eps (ADR-0005)\n");
    let mut qtrunc_over = 0usize;
    let (mut truncated, mut h2_fail, mut h3_fail, mut h3_applies) = (0usize, 0usize, 0usize, 0usize);
    let (mut min_gap_prod, mut min_gap_rig) = (f64::INFINITY, f64::INFINITY);
    let (mut min_f, mut min_fb) = (1.0f64, 1.0f64);
    for k in 0..100 {
        let body = brickwork(&mut rng, n, layers);
        let mut m = run(&program(&format!("qmps    {n}, {chi}"), n, &body));
        let (psi_m, fb, proven) = readout(&mut m, n);
        let mut e = run(&program(&format!("qalloc  {n}"), n, &body));
        let (psi_e, _, _) = readout(&mut e, n);
        let ov = psi_e.iter().zip(&psi_m).fold(C64::ZERO, |s, (x, y)| s + x.conj() * *y);
        let f = ov.norm_sqr() / psi_m.iter().map(|a| a.norm_sqr()).sum::<f64>();
        let eps = match m.qpu.as_ref().unwrap() {
            Backend::Mps(mm) => mm.eps_log.clone(),
            Backend::StateVector(_) => unreachable!(),
        };
        let theta: f64 = eps.iter().map(|x| (1.0 - x).sqrt().min(1.0).acos()).sum();
        let applies = theta < std::f64::consts::FRAC_PI_2;
        let f_rig = if applies { theta.cos().powi(2) } else { 0.0 };
        let max_eps = eps.iter().cloned().fold(0.0, f64::max);
        let _ = writeln!(report, "{k} {f:.12} {fb:.12} {f_rig:.12} {} {max_eps:.3e}", eps.len());
        truncated += usize::from(fb < 1.0);
        qtrunc_over += usize::from(f < proven - 1e-9);
        h2_fail += usize::from(f < fb - 1e-9);
        h3_applies += usize::from(applies);
        h3_fail += usize::from(applies && f < f_rig - 1e-9);
        min_gap_prod = min_gap_prod.min(f - fb);
        if applies {
            min_gap_rig = min_gap_rig.min(f - f_rig);
        }
        min_f = min_f.min(f);
        min_fb = min_fb.min(fb);
    }
    let _ = writeln!(
        report,
        "# H1 truncated {truncated}/100; H2 F < F_bound in {h2_fail}; H3 applies to {h3_applies}, F < F_rigorous in {h3_fail}; \
         min(F - F_bound) {min_gap_prod:.3e}; min(F - F_rigorous) {min_gap_rig:.3e}; min F {min_f:.4}; min F_bound {min_fb:.4}"
    );
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments/qgpu/adr5_result.txt");
    std::fs::write(&path, &report).unwrap();
    println!(
        "ADR-0005: H1 truncated {truncated}/100 | H2 F<F_bound in {h2_fail} (min F-F_bound {min_gap_prod:.3e}) | \
         H3 applies {h3_applies}, F<F_rig in {h3_fail} (min F-F_rig {min_gap_rig:.3e}) | min F {min_f:.4}, min F_bound {min_fb:.4}"
    );
    assert!(truncated >= 95, "H1: only {truncated} truncated");
    assert_eq!(h3_fail, 0, "H3: the proven bound failed -- the bookkeeping is wrong");
    // H2 FAILED as registered (2026-09-29): locked, so a change in the numerics shows up.
    assert_eq!(h2_fail, 53, "H2's recorded failure count moved: the numerics changed");
    // The consequence: qtrunc reports the proven bound, and it never over-claims.
    println!("qtrunc (proven) over-claimed in {qtrunc_over} of 100");
    assert_eq!(qtrunc_over, 0, "qtrunc over-claimed fidelity");
}
