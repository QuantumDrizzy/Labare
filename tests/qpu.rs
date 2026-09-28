//! U-QPU phase 1, the frozen falsifiers F1, F2, F4, F5 (docs/adr/ADR-0003-u-qpu-phase1.md).
//! F3 (the independent Qiskit oracle) is in tests/qpu_oracle.rs.
//!
//! The expected matrices here are written out by hand, independently of
//! src/qpu.rs, so the backend is never checked against itself.

// Matrix code reads as mathematics with explicit row/column indices.
#![allow(clippy::needless_range_loop)]

use unibit::assembler::Assembler;
use unibit::binary::{self, Object};
use unibit::cpu::Cpu;
use unibit::qpu::{QAxis, QGate, QRng, StateVector, C64};

use std::f64::consts::{FRAC_1_SQRT_2 as R, PI};

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn close(a: C64, b: C64, tol: f64) -> bool {
    (a.re - b.re).abs() <= tol && (a.im - b.im).abs() <= tol
}

/// Textbook matrices, typed here, not taken from src/qpu.rs.
fn textbook(g: QGate) -> [[C64; 2]; 2] {
    let (o, l) = (c(0.0, 0.0), c(1.0, 0.0));
    match g {
        QGate::H => [[c(R, 0.0), c(R, 0.0)], [c(R, 0.0), c(-R, 0.0)]],
        QGate::X => [[o, l], [l, o]],
        QGate::Y => [[o, c(0.0, -1.0)], [c(0.0, 1.0), o]],
        QGate::Z => [[l, o], [o, c(-1.0, 0.0)]],
        QGate::S => [[l, o], [o, c(0.0, 1.0)]],
        QGate::Sdg => [[l, o], [o, c(0.0, -1.0)]],
        QGate::T => [[l, o], [o, c((PI / 4.0).cos(), (PI / 4.0).sin())]],
        QGate::Tdg => [[l, o], [o, c((PI / 4.0).cos(), -(PI / 4.0).sin())]],
    }
}

/// Rotation exp(-i θ σ/2), typed from the definition.
fn textbook_rot(a: QAxis, t: f64) -> [[C64; 2]; 2] {
    let (co, si) = ((t / 2.0).cos(), (t / 2.0).sin());
    match a {
        QAxis::X => [[c(co, 0.0), c(0.0, -si)], [c(0.0, -si), c(co, 0.0)]],
        QAxis::Y => [[c(co, 0.0), c(-si, 0.0)], [c(si, 0.0), c(co, 0.0)]],
        QAxis::Z => [[c(co, -si), c(0.0, 0.0)], [c(0.0, 0.0), c(co, si)]],
    }
}

fn unitarity_err(m: &[[C64; 2]; 2]) -> f64 {
    let mut worst: f64 = 0.0;
    for i in 0..2 {
        for j in 0..2 {
            let mut s = C64::ZERO;
            for k in 0..2 {
                s = s + m[k][i].conj() * m[k][j];
            }
            let want = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((s.re - want).abs()).max(s.im.abs());
        }
    }
    worst
}

// ---------------------------------------------------------------- F1

#[test]
fn f1_every_gate_matches_its_closed_form_on_both_basis_states() {
    let check = |m_backend: &[[C64; 2]; 2], m_book: &[[C64; 2]; 2], label: &str| {
        for col in 0..2 {
            let mut s = StateVector::new(1).unwrap();
            if col == 1 {
                s.apply1(0, &textbook(QGate::X)).unwrap();
            }
            s.apply1(0, m_backend).unwrap();
            for row in 0..2 {
                assert!(close(s.amps[row], m_book[row][col], 1e-15), "{label}: column {col}, row {row}");
            }
        }
        assert!(unitarity_err(m_backend) <= 1e-15, "{label}: not unitary to 1e-15");
    };
    for g in QGate::ALL {
        check(&g.matrix(), &textbook(g), g.name());
    }
    let angles = [0.0, PI / 2.0, PI, 3.0 * PI / 2.0, PI / 4.0, 1.0, 2.5, -0.7];
    for a in [QAxis::X, QAxis::Y, QAxis::Z] {
        for t in angles {
            check(&a.matrix(t), &textbook_rot(a, t), &format!("r{}({t})", a.name()));
        }
    }
}

// ---------------------------------------------------------------- F2

fn random_state(n: u32, rng: &mut QRng) -> Vec<C64> {
    let mut v: Vec<C64> = (0..1usize << n).map(|_| c(rng.uniform() - 0.5, rng.uniform() - 0.5)).collect();
    let norm = v.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt();
    for a in &mut v {
        *a = c(a.re / norm, a.im / norm);
    }
    v
}

/// Dense I ⊗ … ⊗ U ⊗ … ⊗ I with U on qubit q (little-endian), built from the definition.
fn dense_one(n: u32, q: u32, u: &[[C64; 2]; 2]) -> Vec<Vec<C64>> {
    let d = 1usize << n;
    let mut m = vec![vec![C64::ZERO; d]; d];
    for (row, mrow) in m.iter_mut().enumerate() {
        for (col, entry) in mrow.iter_mut().enumerate() {
            // Other qubits must agree; the q-th bits index U.
            if (row ^ col) & !(1usize << q) != 0 {
                continue;
            }
            *entry = u[(row >> q) & 1][(col >> q) & 1];
        }
    }
    m
}

fn matvec(m: &[Vec<C64>], v: &[C64]) -> Vec<C64> {
    m.iter().map(|row| row.iter().zip(v).fold(C64::ZERO, |s, (a, b)| s + *a * *b)).collect()
}

#[test]
fn f2_placement_matches_the_dense_kronecker_construction() {
    let mut rng = QRng::new(20260929);
    for n in 1..=8u32 {
        for q in 0..n {
            for g in QGate::ALL {
                let v = random_state(n, &mut rng);
                let mut s = StateVector::new(n).unwrap();
                s.amps = v.clone();
                s.apply1(q, &g.matrix()).unwrap();
                let want = matvec(&dense_one(n, q, &textbook(g)), &v);
                for (a, b) in s.amps.iter().zip(&want) {
                    assert!(close(*a, *b, 1e-14), "n={n} q={q} {}", g.name());
                }
            }
        }
    }
    // Two-qubit gates, every ordered pair for n <= 5, from their definitions.
    for n in 2..=5u32 {
        for a in 0..n {
            for b in 0..n {
                if a == b {
                    continue;
                }
                let d = 1usize << n;
                let v = random_state(n, &mut rng);
                let (ab, bb) = (1usize << a, 1usize << b);
                // CX(control a, target b): |x> -> |x xor b-bit if a-bit set>
                let mut cx_want = vec![C64::ZERO; d];
                for (x, amp) in v.iter().enumerate() {
                    let y = if x & ab != 0 { x ^ bb } else { x };
                    cx_want[y] = *amp;
                }
                let mut s = StateVector::new(n).unwrap();
                s.amps = v.clone();
                s.cx(a, b).unwrap();
                for (x, (got, want)) in s.amps.iter().zip(&cx_want).enumerate() {
                    assert!(close(*got, *want, 1e-14), "cx n={n} c={a} t={b} x={x}");
                }
                // CZ: phase -1 where both bits are set
                let cz_want: Vec<C64> =
                    v.iter().enumerate().map(|(x, amp)| if x & ab != 0 && x & bb != 0 { c(-amp.re, -amp.im) } else { *amp }).collect();
                let mut s = StateVector::new(n).unwrap();
                s.amps = v.clone();
                s.cz(a, b).unwrap();
                for (got, want) in s.amps.iter().zip(&cz_want) {
                    assert!(close(*got, *want, 1e-14), "cz n={n} a={a} b={b}");
                }
            }
        }
    }
}

// ---------------------------------------------------------------- F4

fn bell_shots(seed: u64, shots: usize) -> Vec<u8> {
    let mut rng = QRng::new(seed);
    let mut out = Vec::with_capacity(shots);
    for _ in 0..shots {
        let mut s = StateVector::new(2).unwrap();
        s.rng = rng.clone();
        s.apply1(0, &QGate::H.matrix()).unwrap();
        s.cx(0, 1).unwrap();
        let m0 = s.measure(0).unwrap();
        let m1 = s.measure(1).unwrap();
        out.push(m0 | (m1 << 1));
        rng = s.rng.clone();
    }
    out
}

#[test]
fn f4_bell_statistics_and_exact_replay() {
    const N: usize = 100_000;
    let shots = bell_shots(20260929, N);
    let n00 = shots.iter().filter(|&&s| s == 0).count();
    let n01_10 = shots.iter().filter(|&&s| s == 1 || s == 2).count();
    let p00 = n00 as f64 / N as f64;
    let sigma = (0.25 / N as f64).sqrt();
    println!("F4: P(00) = {p00:.5}, |P - 0.5| = {:.2} sigma, P(01)+P(10) = {n01_10}", (p00 - 0.5).abs() / sigma);
    assert!((p00 - 0.5).abs() <= 4.0 * sigma, "P(00) = {p00}");
    assert_eq!(n01_10, 0, "a Bell pair never reads 01 or 10");
    assert_eq!(shots, bell_shots(20260929, N), "the same seed replays the same outcomes");
}

// ---------------------------------------------------------------- F5

/// A complex Gaussian pair, normalised: a Haar-random qubit state (a, b).
fn haar(rng: &mut QRng) -> (C64, C64) {
    let mut g = || {
        let (u1, u2) = (rng.uniform().max(1e-300), rng.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    };
    let (a, b) = (c(g(), g()), c(g(), g()));
    let n = (a.norm_sqr() + b.norm_sqr()).sqrt();
    (c(a.re / n, a.im / n), c(b.re / n, b.im / n))
}

#[test]
fn f5_teleportation_with_feed_forward_in_the_isa() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/programs/qpu_teleport.uasm")).unwrap();
    let prog = Assembler::new().assemble(&src).expect("assembles");
    let obj = binary::read_object(&binary::write_object(&Object {
        entry_point: prog.entry_point,
        code: prog.instructions,
        data: prog.data_segment,
    }))
    .expect("round-trips");

    let mut rng = QRng::new(20260929);
    let mut worst: f64 = 0.0;
    let mut outcomes = [0usize; 4];
    const OUT: u64 = 0x4_0000;
    for trial in 0..1000u64 {
        let (a, b) = haar(&mut rng);
        // Ry(theta) then Rz(phi) prepares (a, b) up to a global phase.
        let theta = 2.0 * a.re.hypot(a.im).min(1.0).acos();
        let phi = b.im.atan2(b.re) - a.im.atan2(a.re);

        let mut cpu = Cpu::new(1024 * 1024);
        cpu.reset(obj.entry_point);
        cpu.regs[10] = unibit::Reg256::from_u64(theta.to_bits());
        cpu.regs[11] = unibit::Reg256::from_u64(phi.to_bits());
        cpu.regs[12] = unibit::Reg256::from_u64(0xC0FFEE ^ trial);
        cpu.regs[13] = unibit::Reg256::from_u64(OUT);
        cpu.run_program(&obj.code, 10_000).expect("runs");

        // The final state as the ISA wrote it: 8 amplitudes, 32 B each.
        let amps: Vec<C64> = (0..8u64)
            .map(|i| {
                let l = cpu.memory.load_256(OUT + 32 * i).unwrap();
                c(f64::from_bits(l[0]), f64::from_bits(l[1]))
            })
            .collect();
        let m = (0..4).max_by(|&x, &y| {
            let p = |k: usize| amps[k].norm_sqr() + amps[k + 4].norm_sqr();
            p(x).partial_cmp(&p(y)).unwrap()
        });
        let m = m.unwrap();
        outcomes[m] += 1;
        let (lo, hi) = (amps[m], amps[m + 4]);
        // Fidelity |<psi|phi>|^2, global phase free.
        let ov = a.conj() * lo + b.conj() * hi;
        let f = ov.norm_sqr();
        worst = worst.max((1.0 - f).abs());
    }
    println!("F5: 1000 Haar states, worst |1 - F| = {worst:.3e}, outcome counts (m0 + 2 m1) = {outcomes:?}");
    assert!(worst <= 1e-12, "teleportation lost fidelity: {worst}");
    assert!(outcomes.iter().all(|&k| k > 150), "all four correction branches were exercised");
}
