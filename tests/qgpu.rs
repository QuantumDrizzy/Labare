//! QGPU phase 1, the frozen falsifiers G1–G5 (docs/adr/ADR-0004-qgpu-phase1-mps.md).
//! Every program is MTLB assembly: assembled, encoded to the object format and
//! decoded back, run on the CPU, read out through the ISA (`qamp`, `qtrunc`, `sq`).

use std::fmt::Write as _;

use unibit::assembler::Assembler;
use unibit::binary::{self, Object};
use unibit::cpu::Cpu;
use unibit::qpu::{Backend, QRng, C64};

const OUT: u64 = 0x4_0000;

fn dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("experiments")
}

fn hex_f64(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).expect("hex f64"))
}

/// Assemble, round-trip through the object format, run; return the CPU.
fn run(src: &str, regs: &[(usize, u64)]) -> Cpu {
    let prog = Assembler::new().assemble(src).expect("assembles");
    let obj = binary::read_object(&binary::write_object(&Object {
        entry_point: prog.entry_point,
        code: prog.instructions,
        data: prog.data_segment,
    }))
    .expect("round-trips");
    let mut cpu = Cpu::new(1024 * 1024);
    cpu.reset(obj.entry_point);
    for &(r, v) in regs {
        cpu.regs[r] = unibit::Reg256::from_u64(v);
    }
    cpu.run_program(&obj.code, 5_000_000).expect("runs");
    cpu
}

/// A circuit as a program: `alloc` line, gates, full readout of 2^n amplitudes
/// and then the fidelity bound, all written through the ISA.
fn program(alloc: &str, n: u32, gates: &[Vec<String>]) -> String {
    let mut s = String::from("        .text\n        .global _start\n_start:\n");
    let _ = writeln!(s, "        {alloc}");
    for g in gates {
        match g[0].as_str() {
            "rx" | "ry" | "rz" => {
                let bits = u64::from_str_radix(&g[2], 16).unwrap() as i64;
                let _ = writeln!(s, "        li      t2, {bits}\n        qrot.{}  {}, t2", &g[0][1..], g[1]);
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
                 addi    t0, t0, 1\n        blt     t0, t1, readout\n        qtrunc  t2\n        sq      t2, 0(a3)\n        halt\n",
        1u64 << n
    );
    s
}

/// Amplitudes and the fidelity bound, as the program wrote them.
fn readout(cpu: &mut Cpu, n: u32) -> (Vec<C64>, f64) {
    let d = 1usize << n;
    let amps = (0..d)
        .map(|i| {
            let l = cpu.memory.load_256(OUT + 32 * i as u64).unwrap();
            C64::new(f64::from_bits(l[0]), f64::from_bits(l[1]))
        })
        .collect();
    let fb = f64::from_bits(cpu.memory.load_256(OUT + 32 * d as u64).unwrap()[0]);
    (amps, fb)
}

fn max_diff(a: &[C64], b: &[C64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x.re - y.re).abs().max((x.im - y.im).abs())).fold(0.0, f64::max)
}

type Circuit = (u32, Vec<Vec<String>>);

fn f3_circuits() -> (Vec<Circuit>, Vec<Vec<C64>>) {
    let c = std::fs::read_to_string(dir().join("u-qpu").join("f3_circuits.txt"))
        .expect("run `python tools/qpu_oracle.py` first");
    let o = std::fs::read_to_string(dir().join("u-qpu").join("f3_qiskit.txt")).expect("f3_qiskit.txt");
    let mut lines = c.lines();
    let mut circuits = Vec::new();
    while let Some(h) = lines.next() {
        let h: Vec<&str> = h.split_whitespace().collect();
        let (n, depth): (u32, usize) = (h[1].parse().unwrap(), h[2].parse().unwrap());
        circuits.push((n, (0..depth).map(|_| lines.next().unwrap().split_whitespace().map(str::to_string).collect()).collect()));
    }
    let states = o
        .lines()
        .map(|l| l.split_whitespace().map(|p| C64::new(hex_f64(&p[..16]), hex_f64(&p[16..]))).collect())
        .collect();
    (circuits, states)
}

// ---------------------------------------------------------------- G1 + G2

#[test]
fn g1_g2_exact_capacity_matches_qiskit_and_the_state_vector() {
    let (circuits, oracle) = f3_circuits();
    let mut report = String::from("# circuit n chi max|mps-qiskit| max|mps-sv| F_bound max_bond (ADR-0004 G1/G2)\n");
    let (mut w1, mut w2, mut min_fb) = (0.0f64, 0.0f64, 1.0f64);
    for (k, ((n, gates), want)) in circuits.iter().zip(&oracle).enumerate() {
        let chi = 1u32 << n.div_ceil(2);
        let mut mps_cpu = run(&program(&format!("qmps    {n}, {chi}"), *n, gates), &[]);
        let (mps, fb) = readout(&mut mps_cpu, *n);
        let mut sv_cpu = run(&program(&format!("qalloc  {n}"), *n, gates), &[]);
        let (sv, _) = readout(&mut sv_cpu, *n);
        let (d1, d2) = (max_diff(&mps, want), max_diff(&mps, &sv));
        let bond = match mps_cpu.qpu.as_ref().unwrap() {
            Backend::Mps(m) => m.max_bond(),
            Backend::StateVector(_) => 0,
        };
        let _ = writeln!(report, "{k} {n} {chi} {d1:.3e} {d2:.3e} {fb:.15} {bond}");
        w1 = w1.max(d1);
        w2 = w2.max(d2);
        min_fb = min_fb.min(fb);
        assert!(d1 <= 1e-10, "G1 circuit {k} (n = {n}): |mps - qiskit| = {d1:e}");
        assert!(d2 <= 1e-10, "G2 circuit {k} (n = {n}): |mps - sv| = {d2:e}");
        assert!(fb >= 1.0 - 1e-12, "G1 circuit {k}: F_bound {fb} at exact capacity");
    }
    let _ = writeln!(report, "# worst G1 {w1:.3e}, worst G2 {w2:.3e}, min F_bound {min_fb:.15}");
    std::fs::create_dir_all(dir().join("qgpu")).unwrap();
    std::fs::write(dir().join("qgpu").join("g1_g2_result.txt"), &report).unwrap();
    println!("G1/G2: 200 circuits at exact capacity: worst |mps-qiskit| {w1:.3e}, worst |mps-sv| {w2:.3e}, min F_bound {min_fb:.15}");
}

// ---------------------------------------------------------------- G3

fn random_circuit(rng: &mut QRng, n: u32, depth: usize) -> Vec<Vec<String>> {
    const KINDS: [&str; 13] = ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "rx", "ry", "rz", "cx", "cz"];
    let pick = |rng: &mut QRng, m: u64| (rng.next_u64() % m) as u32;
    (0..depth)
        .map(|_| {
            let k = KINDS[pick(rng, 13) as usize];
            match k {
                "rx" | "ry" | "rz" => {
                    let t = rng.uniform() * 2.0 * std::f64::consts::PI;
                    vec![k.into(), pick(rng, n as u64).to_string(), format!("{:016x}", t.to_bits())]
                }
                "cx" | "cz" => {
                    let a = pick(rng, n as u64);
                    let mut b = pick(rng, n as u64 - 1);
                    if b >= a {
                        b += 1;
                    }
                    vec![k.into(), a.to_string(), b.to_string()]
                }
                _ => vec![k.into(), pick(rng, n as u64).to_string()],
            }
        })
        .collect()
}

#[test]
#[ignore = "ADR-0004 G3 FAILED its own non-vacuity condition (10 of 100 truncated, 80 required); kept as the record; superseded by ADR-0005 (tests/qgpu_bound.rs)"]
fn g3_the_fidelity_bound_does_not_lie() {
    let mut rng = QRng::new(20260930);
    let (n, depth, chi) = (10u32, 60usize, 4u32);
    let mut report = String::from("# circuit F_true F_bound(prod) F_rigorous(cos^2 sum theta) truncations (ADR-0004 G3)\n");
    let (mut truncated, mut violations, mut rigorous_violations, mut worst_gap) = (0usize, 0usize, 0usize, f64::INFINITY);
    for k in 0..100 {
        let gates = random_circuit(&mut rng, n, depth);
        let mut m_cpu = run(&program(&format!("qmps    {n}, {chi}"), n, &gates), &[]);
        let (psi_m, fb) = readout(&mut m_cpu, n);
        let mut s_cpu = run(&program(&format!("qalloc  {n}"), n, &gates), &[]);
        let (psi_e, _) = readout(&mut s_cpu, n);
        let ov = psi_e.iter().zip(&psi_m).fold(C64::ZERO, |s, (e, m)| s + e.conj() * *m);
        let nm: f64 = psi_m.iter().map(|a| a.norm_sqr()).sum();
        let f_true = ov.norm_sqr() / nm;
        // Unregistered, reported beside the registered bound: the rigorous Fubini-Study bound.
        let eps = match m_cpu.qpu.as_ref().unwrap() {
            Backend::Mps(m) => m.eps_log.clone(),
            Backend::StateVector(_) => vec![],
        };
        let theta: f64 = eps.iter().map(|e| (1.0 - e).sqrt().min(1.0).acos()).sum();
        let f_rig = if theta < std::f64::consts::FRAC_PI_2 { theta.cos().powi(2) } else { 0.0 };
        let _ = writeln!(report, "{k} {f_true:.12} {fb:.12} {f_rig:.12} {}", eps.len());
        truncated += usize::from(fb < 1.0);
        violations += usize::from(f_true < fb - 1e-9);
        rigorous_violations += usize::from(f_true < f_rig - 1e-9);
        worst_gap = worst_gap.min(f_true - fb);
    }
    let _ = writeln!(report, "# truncated {truncated}/100, F < F_bound - 1e-9 in {violations}, F < F_rigorous - 1e-9 in {rigorous_violations}, min(F - F_bound) {worst_gap:.3e}");
    std::fs::create_dir_all(dir().join("qgpu")).unwrap();
    std::fs::write(dir().join("qgpu").join("g3_result.txt"), &report).unwrap();
    println!("G3: truncated {truncated}/100; F < F_bound in {violations}; F < F_rigorous in {rigorous_violations}; min(F - F_bound) = {worst_gap:.3e}");
    assert!(truncated >= 80, "G3 is vacuous: only {truncated} circuits truncated");
    assert_eq!(violations, 0, "G3: the product bound claimed more fidelity than the state has in {violations} circuits");
}

// ---------------------------------------------------------------- G4

fn ghz_program(n: u32, shots_measure: bool) -> String {
    let mut s = format!("        .text\n        .global _start\n_start:\n        qseed   a2\n        qmps    {n}, 2\n        qg1.h   0\n");
    for q in 0..n - 1 {
        let _ = writeln!(s, "        qcx     {q}, {}", q + 1);
    }
    if shots_measure {
        s.push_str("        li      s1, 0\n");
        for q in 0..n {
            let _ = writeln!(s, "        qmeas   t0, {q}\n        add     s1, s1, t0");
        }
        let _ = writeln!(s, "        li      a3, {OUT}\n        sd      s1, 0(a3)");
    }
    let _ = writeln!(s, "        li      a3, {}\n        qtrunc  t2\n        sq      t2, 0(a3)\n        halt", OUT + 64);
    s
}

#[test]
fn g4_ghz_on_100_qubits_beyond_any_state_vector() {
    let n = 100u32;
    let mut cpu = run(&ghz_program(n, false), &[(12, 1)]);
    let fb = f64::from_bits(cpu.memory.load_256(OUT + 64).unwrap()[0]);
    let m = match cpu.qpu.as_ref().unwrap() {
        Backend::Mps(m) => m,
        Backend::StateVector(_) => panic!("qmps gave a state vector"),
    };
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let a0 = m.amp_bits(&vec![0; n as usize]).unwrap();
    let a1 = m.amp_bits(&vec![1; n as usize]).unwrap();
    let alt: Vec<u8> = (0..n).map(|q| (q % 2) as u8).collect();
    let ax = m.amp_bits(&alt).unwrap();
    println!("G4: <0..0|psi> = {:.15}, <1..1|psi> = {:.15}, <0101..|psi> = {:.1e}, F_bound = {fb}, max bond {}", a0.re, a1.re, ax.re.hypot(ax.im), m.max_bond());
    assert!((a0.re - r).abs() <= 1e-12 && a0.im.abs() <= 1e-12);
    assert!((a1.re - r).abs() <= 1e-12 && a1.im.abs() <= 1e-12);
    assert!(ax.re.hypot(ax.im) <= 1e-12);
    assert_eq!(fb, 1.0, "GHZ fits chi = 2 exactly");

    // 1,000 shots, each a fresh preparation measuring all 100 qubits.
    let prog = ghz_program(n, true);
    let (mut zeros, mut ones) = (0usize, 0usize);
    for shot in 0..1000u64 {
        let mut c = run(&prog, &[(12, 0x6857 ^ (shot << 8))]);
        let sum = c.memory.load_u64(OUT).unwrap();
        assert!(sum == 0 || sum == 100, "shot {shot}: {sum} of 100 qubits read 1 (GHZ is all or nothing)");
        if sum == 0 {
            zeros += 1;
        } else {
            ones += 1;
        }
    }
    let p0 = zeros as f64 / 1000.0;
    let sigma = (0.25f64 / 1000.0).sqrt();
    println!("G4: 1000 shots of 100 qubits: all-0 {zeros}, all-1 {ones}, |p0 - 0.5| = {:.2} sigma", (p0 - 0.5).abs() / sigma);
    assert!((p0 - 0.5).abs() <= 4.0 * sigma);
    // A state vector of 100 qubits is refused.
    assert!(Assembler::new().assemble("        .text\n_start:\n        qalloc  100\n        halt\n").is_err());
}

// ---------------------------------------------------------------- G5

fn haar(rng: &mut QRng) -> (C64, C64) {
    let mut g = || {
        let (u1, u2) = (rng.uniform().max(1e-300), rng.uniform());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let (a, b) = (C64::new(g(), g()), C64::new(g(), g()));
    let n = (a.norm_sqr() + b.norm_sqr()).sqrt();
    (C64::new(a.re / n, a.im / n), C64::new(b.re / n, b.im / n))
}

#[test]
fn g5_feed_forward_teleportation_on_the_tensor_backend() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/programs/qpu_teleport.uasm")).unwrap();
    assert_eq!(src.matches("qalloc  3").count(), 1);
    let src = src.replace("qalloc  3", "qmps    3, 4");
    // ADR-0003 F5's generator and seed.
    let mut rng = QRng::new(20260929);
    let (mut worst, mut outcomes) = (0.0f64, [0usize; 4]);
    for trial in 0..1000u64 {
        let (a, b) = haar(&mut rng);
        let theta = 2.0 * a.re.hypot(a.im).min(1.0).acos();
        let phi = b.im.atan2(b.re) - a.im.atan2(a.re);
        let mut cpu = run(&src, &[(10, theta.to_bits()), (11, phi.to_bits()), (12, 0xC0FFEE ^ trial), (13, OUT)]);
        let amps: Vec<C64> = (0..8u64)
            .map(|i| {
                let l = cpu.memory.load_256(OUT + 32 * i).unwrap();
                C64::new(f64::from_bits(l[0]), f64::from_bits(l[1]))
            })
            .collect();
        let p = |k: usize| amps[k].norm_sqr() + amps[k + 4].norm_sqr();
        let m = (0..4).max_by(|&x, &y| p(x).partial_cmp(&p(y)).unwrap()).unwrap();
        outcomes[m] += 1;
        let ov = a.conj() * amps[m] + b.conj() * amps[m + 4];
        worst = worst.max((1.0 - ov.norm_sqr()).abs());
    }
    println!("G5: 1000 Haar states on the MPS: worst |1 - F| = {worst:.3e}, branches {outcomes:?}");
    assert!(worst <= 1e-12);
    assert!(outcomes.iter().all(|&k| k > 150));
}
