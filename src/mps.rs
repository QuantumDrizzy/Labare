// ============================================================================
// Unibit — QGPU phase 1: a matrix product state backend (ADR-0004)
// ============================================================================
//
// The same quantum instructions as the U-QPU, on a tensor network that can
// hold many qubits when entanglement is low. One tensor per qubit,
// A[q] of shape (chi_left, 2, chi_right), complex f64, little-endian qubits.
//
// The refusal: truncation is never hidden. Every SVD adds its discarded weight
// eps = sum(discarded s^2) / sum(s^2) to a ledger, and the backend reports
// F_bound = prod(1 - eps). `qtrunc` reads it.
//
// The state is kept in mixed canonical form: before any two-site operation the
// orthogonality centre is moved onto the pair, so the singular values of the
// two-site block are Schmidt coefficients and eps is the weight truly lost.
//
// Not claimed: GPU execution (phase 2, through LYTH), speed, noise.
// ============================================================================

use crate::qpu::{C64, Mat2, QGate, QRng};

/// Largest MPS register: qubit indices are u8 immediates and 128 is declared.
pub const MAX_MPS_QUBITS: u32 = 128;
/// Largest bond dimension a program may ask for.
pub const MAX_CHI: u32 = 256;

/// Jacobi stops when the Gram off-diagonal norm is below this times its Frobenius norm.
const JACOBI_TOL: f64 = 1e-15;
const JACOBI_MAX_SWEEPS: u32 = 60;
/// Singular values below this times s_max are numerical zero (their weight is still counted).
const ZERO_CUT: f64 = 1e-15;

/// A dense complex matrix, row-major.
#[derive(Clone, Debug)]
struct Mat {
    rows: usize,
    cols: usize,
    d: Vec<C64>,
}

impl Mat {
    fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, d: vec![C64::ZERO; rows * cols] }
    }
    #[inline]
    fn at(&self, r: usize, c: usize) -> C64 {
        self.d[r * self.cols + c]
    }
    #[inline]
    fn set(&mut self, r: usize, c: usize, v: C64) {
        self.d[r * self.cols + c] = v;
    }
    fn adjoint(&self) -> Mat {
        let mut t = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                t.set(c, r, self.at(r, c).conj());
            }
        }
        t
    }
}

/// Thin SVD M = U diag(s) V^dagger with s sorted descending.
/// U is rows x k, V is cols x k, k = min(rows, cols).
struct Svd {
    u: Mat,
    s: Vec<f64>,
    v: Mat,
    /// True if Jacobi stopped on the sweep cap rather than the tolerance.
    capped: bool,
}

/// One-sided (Hestenes) Jacobi on the columns of `m` (rows >= cols).
fn svd_tall(m: &Mat) -> Svd {
    let (rows, cols) = (m.rows, m.cols);
    // Work on columns.
    let mut a: Vec<Vec<C64>> = (0..cols).map(|c| (0..rows).map(|r| m.at(r, c)).collect()).collect();
    let mut v: Vec<Vec<C64>> = (0..cols)
        .map(|c| (0..cols).map(|r| if r == c { C64::ONE } else { C64::ZERO }).collect())
        .collect();
    let dot = |x: &[C64], y: &[C64]| x.iter().zip(y).fold(C64::ZERO, |s, (a, b)| s + a.conj() * *b);
    let nrm = |x: &[C64]| x.iter().map(|a| a.norm_sqr()).sum::<f64>();

    let mut capped = true;
    for _sweep in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0f64;
        let mut fro = 0.0f64;
        for p in 0..cols {
            let np = nrm(&a[p]);
            fro += np * np;
            for q in (p + 1)..cols {
                let alpha = nrm(&a[p]);
                let beta = nrm(&a[q]);
                let gamma = dot(&a[p], &a[q]);
                let g = gamma.norm_sqr().sqrt();
                off += 2.0 * g * g;
                if g <= f64::MIN_POSITIVE || g <= 1e-300 * (alpha * beta).sqrt().max(1e-300) {
                    continue;
                }
                let ph = C64::new(gamma.re / g, gamma.im / g).conj(); // conj(gamma / |gamma|)
                let zeta = (beta - alpha) / (2.0 * g);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let t = if zeta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for r in 0..rows {
                    let (xp, xq) = (a[p][r], a[q][r] * ph);
                    a[p][r] = C64::new(c * xp.re - s * xq.re, c * xp.im - s * xq.im);
                    a[q][r] = C64::new(s * xp.re + c * xq.re, s * xp.im + c * xq.im);
                }
                for r in 0..cols {
                    let (xp, xq) = (v[p][r], v[q][r] * ph);
                    v[p][r] = C64::new(c * xp.re - s * xq.re, c * xp.im - s * xq.im);
                    v[q][r] = C64::new(s * xp.re + c * xq.re, s * xp.im + c * xq.im);
                }
            }
        }
        if off.sqrt() <= JACOBI_TOL * fro.sqrt().max(f64::MIN_POSITIVE) {
            capped = false;
            break;
        }
    }
    let mut order: Vec<usize> = (0..cols).collect();
    let sig: Vec<f64> = a.iter().map(|col| nrm(col).sqrt()).collect();
    order.sort_by(|&x, &y| sig[y].partial_cmp(&sig[x]).unwrap());
    let mut u = Mat::zeros(rows, cols);
    let mut vv = Mat::zeros(cols, cols);
    let mut s = Vec::with_capacity(cols);
    for (k, &j) in order.iter().enumerate() {
        s.push(sig[j]);
        for r in 0..rows {
            let x = if sig[j] > 0.0 { C64::new(a[j][r].re / sig[j], a[j][r].im / sig[j]) } else { C64::ZERO };
            u.set(r, k, x);
        }
        for r in 0..cols {
            vv.set(r, k, v[j][r]);
        }
    }
    Svd { u, s, v: vv, capped }
}

fn svd(m: &Mat) -> Svd {
    if m.rows >= m.cols {
        svd_tall(m)
    } else {
        // M^dagger = V S U^dagger
        let t = svd_tall(&m.adjoint());
        Svd { u: t.v, s: t.s, v: t.u, capped: t.capped }
    }
}

/// One site tensor, (l, 2, r), stored as [a][s][b] -> (a * 2 + s) * r + b.
#[derive(Clone, Debug)]
struct Site {
    l: usize,
    r: usize,
    d: Vec<C64>,
}

impl Site {
    #[inline]
    fn at(&self, a: usize, s: usize, b: usize) -> C64 {
        self.d[(a * 2 + s) * self.r + b]
    }
}

/// The MPS register.
#[derive(Clone, Debug)]
pub struct Mps {
    pub n: u32,
    pub chi_max: usize,
    sites: Vec<Site>,
    centre: usize,
    /// prod(1 - eps_k) over every SVD so far.
    pub fid_bound: f64,
    /// Every eps_k, in order, for analysis (ADR-0004 G3).
    pub eps_log: Vec<f64>,
    /// SVDs that stopped on the Jacobi sweep cap (reported, ADR-0004 section 3).
    pub jacobi_capped: u64,
    pub rng: QRng,
}

impl Mps {
    /// |0...0> on n qubits with bond cap chi_max.
    pub fn new(n: u32, chi_max: u32) -> Result<Self, String> {
        if n == 0 || n > MAX_MPS_QUBITS {
            return Err(format!("qmps: {n} qubits is outside 1..={MAX_MPS_QUBITS}"));
        }
        if chi_max == 0 || chi_max > MAX_CHI {
            return Err(format!("qmps: chi {chi_max} is outside 1..={MAX_CHI}"));
        }
        let site = Site { l: 1, r: 1, d: vec![C64::ONE, C64::ZERO] };
        Ok(Self {
            n,
            chi_max: chi_max as usize,
            sites: vec![site; n as usize],
            centre: 0,
            fid_bound: 1.0,
            eps_log: Vec::new(),
            jacobi_capped: 0,
            rng: QRng::new(0),
        })
    }

    fn check(&self, q: u32) -> Result<(), String> {
        if q < self.n {
            Ok(())
        } else {
            Err(format!("qubit {q} does not exist in a {}-qubit MPS", self.n))
        }
    }

    /// Largest bond dimension currently held.
    pub fn max_bond(&self) -> usize {
        self.sites.iter().map(|s| s.r).max().unwrap_or(1)
    }

    /// One-qubit gate: exact, local, keeps the canonical form.
    pub fn apply1(&mut self, q: u32, m: &Mat2) -> Result<(), String> {
        self.check(q)?;
        let st = &mut self.sites[q as usize];
        let (l, r) = (st.l, st.r);
        let old = st.d.clone();
        for a in 0..l {
            for b in 0..r {
                let x0 = old[(a * 2) * r + b];
                let x1 = old[(a * 2 + 1) * r + b];
                st.d[(a * 2) * r + b] = m[0][0] * x0 + m[0][1] * x1;
                st.d[(a * 2 + 1) * r + b] = m[1][0] * x0 + m[1][1] * x1;
            }
        }
        Ok(())
    }

    /// Record the truncation of one SVD and return how many values to keep.
    fn keep(&mut self, s: &[f64], cap: usize) -> (usize, f64) {
        let total: f64 = s.iter().map(|x| x * x).sum();
        let smax = s.first().copied().unwrap_or(0.0);
        let mut k = s.iter().take_while(|&&x| x > ZERO_CUT * smax).count().clamp(1, cap);
        k = k.min(s.len());
        let kept: f64 = s[..k].iter().map(|x| x * x).sum();
        let eps = if total > 0.0 { ((total - kept) / total).max(0.0) } else { 0.0 };
        if eps > 0.0 {
            self.fid_bound *= 1.0 - eps;
            self.eps_log.push(eps);
        }
        // Renormalise the kept part to the weight it came from.
        let scale = if kept > 0.0 { (total / kept).sqrt() } else { 1.0 };
        (k, scale)
    }

    /// Move the orthogonality centre to site `to` (exact up to numerical zero).
    fn move_centre(&mut self, to: usize) {
        while self.centre < to {
            let i = self.centre;
            let st = self.sites[i].clone();
            // (l*2, r) = U S V^dagger; A[i] = U, A[i+1] = S V^dagger A[i+1]
            let mut m = Mat::zeros(st.l * 2, st.r);
            for a in 0..st.l {
                for s in 0..2 {
                    for b in 0..st.r {
                        m.set(a * 2 + s, b, st.at(a, s, b));
                    }
                }
            }
            let f = svd(&m);
            self.jacobi_capped += u64::from(f.capped);
            let (k, _scale) = self.keep(&f.s, usize::MAX);
            self.sites[i] = Site { l: st.l, r: k, d: (0..st.l * 2 * k).map(|x| f.u.at(x / k, x % k)).collect() };
            let nx = self.sites[i + 1].clone();
            let mut d = vec![C64::ZERO; k * 2 * nx.r];
            for kk in 0..k {
                for s in 0..2 {
                    for c in 0..nx.r {
                        let mut acc = C64::ZERO;
                        for b in 0..st.r {
                            // (S V^dagger)[kk][b] = s_kk * conj(V[b][kk])
                            let sv = f.v.at(b, kk).conj();
                            acc = acc + C64::new(f.s[kk] * sv.re, f.s[kk] * sv.im) * nx.at(b, s, c);
                        }
                        d[(kk * 2 + s) * nx.r + c] = acc;
                    }
                }
            }
            self.sites[i + 1] = Site { l: k, r: nx.r, d };
            self.centre += 1;
        }
        while self.centre > to {
            let i = self.centre;
            let st = self.sites[i].clone();
            // (l, 2*r) = U S V^dagger; A[i] = V^dagger, A[i-1] = A[i-1] U S
            let mut m = Mat::zeros(st.l, 2 * st.r);
            for a in 0..st.l {
                for s in 0..2 {
                    for b in 0..st.r {
                        m.set(a, s * st.r + b, st.at(a, s, b));
                    }
                }
            }
            let f = svd(&m);
            self.jacobi_capped += u64::from(f.capped);
            let (k, _scale) = self.keep(&f.s, usize::MAX);
            let mut d = vec![C64::ZERO; k * 2 * st.r];
            for kk in 0..k {
                for s in 0..2 {
                    for b in 0..st.r {
                        d[(kk * 2 + s) * st.r + b] = f.v.at(s * st.r + b, kk).conj();
                    }
                }
            }
            self.sites[i] = Site { l: k, r: st.r, d };
            let pv = self.sites[i - 1].clone();
            let mut d = vec![C64::ZERO; pv.l * 2 * k];
            for a in 0..pv.l {
                for s in 0..2 {
                    for kk in 0..k {
                        let mut acc = C64::ZERO;
                        for b in 0..pv.r {
                            let us = f.u.at(b, kk);
                            acc = acc + pv.at(a, s, b) * C64::new(us.re * f.s[kk], us.im * f.s[kk]);
                        }
                        d[(a * 2 + s) * k + kk] = acc;
                    }
                }
            }
            self.sites[i - 1] = Site { l: pv.l, r: k, d };
            self.centre -= 1;
        }
    }

    /// Two-site gate on (q, q+1). `g[out][in]`, index = s_q + 2 * s_(q+1).
    fn apply2_adjacent(&mut self, q: usize, g: &[[C64; 4]; 4]) {
        self.move_centre(q);
        let (a_s, b_s) = (self.sites[q].clone(), self.sites[q + 1].clone());
        let (l, r) = (a_s.l, b_s.r);
        // theta[a][s1][s2][c]
        let mut th = vec![C64::ZERO; l * 4 * r];
        for a in 0..l {
            for s1 in 0..2 {
                for s2 in 0..2 {
                    for c in 0..r {
                        let mut acc = C64::ZERO;
                        for b in 0..a_s.r {
                            acc = acc + a_s.at(a, s1, b) * b_s.at(b, s2, c);
                        }
                        th[((a * 2 + s1) * 2 + s2) * r + c] = acc;
                    }
                }
            }
        }
        // apply g, reshape to M[(a,o1)][(o2,c)]
        let mut m = Mat::zeros(l * 2, 2 * r);
        for a in 0..l {
            for c in 0..r {
                for o in 0..4 {
                    let mut acc = C64::ZERO;
                    for i in 0..4 {
                        let x = th[((a * 2 + (i & 1)) * 2 + (i >> 1)) * r + c];
                        acc = acc + g[o][i] * x;
                    }
                    m.set(a * 2 + (o & 1), (o >> 1) * r + c, acc);
                }
            }
        }
        let f = svd(&m);
        self.jacobi_capped += u64::from(f.capped);
        let (k, scale) = self.keep(&f.s, self.chi_max);
        let mut da = vec![C64::ZERO; l * 2 * k];
        for a in 0..l {
            for s in 0..2 {
                for kk in 0..k {
                    da[(a * 2 + s) * k + kk] = f.u.at(a * 2 + s, kk);
                }
            }
        }
        let mut db = vec![C64::ZERO; k * 2 * r];
        for kk in 0..k {
            let sk = f.s[kk] * scale;
            for s in 0..2 {
                for c in 0..r {
                    let vv = f.v.at(s * r + c, kk).conj();
                    db[(kk * 2 + s) * r + c] = C64::new(sk * vv.re, sk * vv.im);
                }
            }
        }
        self.sites[q] = Site { l, r: k, d: da };
        self.sites[q + 1] = Site { l: k, r, d: db };
        self.centre = q + 1;
    }

    fn swap_adjacent(&mut self, q: usize) {
        let (o, l) = (C64::ZERO, C64::ONE);
        let g = [[l, o, o, o], [o, o, l, o], [o, l, o, o], [o, o, o, l]];
        self.apply2_adjacent(q, &g);
    }

    /// A two-qubit gate on any pair: SWAP the far qubit next to the near one,
    /// apply, SWAP back. `oriented(lo_is_first)` gives the 4x4 matrix for the
    /// pair as it sits at (lo, lo+1).
    fn apply2(&mut self, a: u32, b: u32, oriented: impl Fn(bool) -> [[C64; 4]; 4]) -> Result<(), String> {
        self.check(a)?;
        self.check(b)?;
        if a == b {
            return Err("two-qubit gate on one qubit".into());
        }
        let (lo, hi) = (a.min(b) as usize, a.max(b) as usize);
        for p in (lo + 1..hi).rev() {
            self.swap_adjacent(p);
        }
        self.apply2_adjacent(lo, &oriented(a < b));
        for p in lo + 1..hi {
            self.swap_adjacent(p);
        }
        Ok(())
    }

    /// CNOT (control c, target t).
    pub fn cx(&mut self, c: u32, t: u32) -> Result<(), String> {
        let (o, l) = (C64::ZERO, C64::ONE);
        self.apply2(c, t, |control_is_lo| {
            if control_is_lo {
                // control at s_q (bit 0 of the index): 1 <-> 3
                [[l, o, o, o], [o, o, o, l], [o, o, l, o], [o, l, o, o]]
            } else {
                // control at s_(q+1) (bit 1): 2 <-> 3
                [[l, o, o, o], [o, l, o, o], [o, o, o, l], [o, o, l, o]]
            }
        })
    }

    /// CZ, symmetric.
    pub fn cz(&mut self, a: u32, b: u32) -> Result<(), String> {
        let (o, l) = (C64::ZERO, C64::ONE);
        self.apply2(a, b, |_| [[l, o, o, o], [o, l, o, o], [o, o, l, o], [o, o, o, C64::new(-1.0, 0.0)]])
    }

    /// <psi| P_s(q) |psi> by a full transfer-matrix contraction (P_s projects qubit q on s),
    /// or <psi|psi> when `proj` is None.
    fn expect(&self, proj: Option<(usize, usize)>) -> f64 {
        // E[a][a'] over bra/ket bond indices.
        let mut e = vec![C64::ONE];
        let mut dim = 1usize;
        for (q, st) in self.sites.iter().enumerate() {
            let mut ne = vec![C64::ZERO; st.r * st.r];
            for s in 0..2 {
                if let Some((pq, ps)) = proj {
                    if pq == q && ps != s {
                        continue;
                    }
                }
                for a in 0..dim {
                    for ap in 0..dim {
                        let eaa = e[a * dim + ap];
                        if eaa == C64::ZERO {
                            continue;
                        }
                        for b in 0..st.r {
                            let x = eaa * st.at(a, s, b);
                            for bp in 0..st.r {
                                ne[b * st.r + bp] = ne[b * st.r + bp] + x * st.at(ap, s, bp).conj();
                            }
                        }
                    }
                }
            }
            e = ne;
            dim = st.r;
        }
        e[0].re
    }

    /// Measure q in Z: P(1) from the full contraction, outcome from the RNG,
    /// project A[q] and renormalise the state.
    pub fn measure(&mut self, q: u32) -> Result<u8, String> {
        self.check(q)?;
        let qi = q as usize;
        // With the centre on q, projecting A[q] leaves every other site orthonormal.
        self.move_centre(qi);
        let norm = self.expect(None);
        let p1 = (self.expect(Some((qi, 1))) / norm).clamp(0.0, 1.0);
        let outcome = u8::from(self.rng.uniform() < p1);
        let p = if outcome == 1 { p1 } else { 1.0 - p1 };
        let scale = 1.0 / (p * norm).sqrt();
        let st = &mut self.sites[qi];
        let r = st.r;
        for a in 0..st.l {
            for s in 0..2 {
                for b in 0..r {
                    let i = (a * 2 + s) * r + b;
                    st.d[i] = if s as u8 == outcome { C64::new(st.d[i].re * scale, st.d[i].im * scale) } else { C64::ZERO };
                }
            }
        }
        Ok(outcome)
    }

    pub fn reset(&mut self, q: u32) -> Result<u8, String> {
        let m = self.measure(q)?;
        if m == 1 {
            self.apply1(q, &QGate::X.matrix())?;
        }
        Ok(m)
    }

    /// Amplitude of a basis state given as one bit per qubit (any width). `qamp`
    /// takes a 64-bit index; this reads registers wider than 64 qubits.
    pub fn amp_bits(&self, bits: &[u8]) -> Result<C64, String> {
        if bits.len() != self.n as usize {
            return Err(format!("amp_bits: {} bits for {} qubits", bits.len(), self.n));
        }
        let mut v = vec![C64::ONE];
        for (st, &s) in self.sites.iter().zip(bits) {
            let mut nv = vec![C64::ZERO; st.r];
            for (a, va) in v.iter().enumerate() {
                for (bb, nb) in nv.iter_mut().enumerate() {
                    *nb = *nb + *va * st.at(a, s as usize, bb);
                }
            }
            v = nv;
        }
        Ok(v[0])
    }

    /// Amplitude of a basis index (bit q = qubit q). Indices up to 2^64 - 1;
    /// qubits 64..n are read as 0.
    pub fn amp(&self, index: u64) -> Result<C64, String> {
        let mut v = vec![C64::ONE];
        for (q, st) in self.sites.iter().enumerate() {
            let s = if q < 64 { ((index >> q) & 1) as usize } else { 0 };
            let mut nv = vec![C64::ZERO; st.r];
            for (a, va) in v.iter().enumerate() {
                for (b, nb) in nv.iter_mut().enumerate() {
                    *nb = *nb + *va * st.at(a, s, b);
                }
            }
            v = nv;
        }
        Ok(v[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svd_reconstructs_a_random_complex_matrix() {
        let mut rng = QRng::new(7);
        for (rows, cols) in [(6, 4), (4, 6), (8, 8), (1, 5), (5, 1)] {
            let mut m = Mat::zeros(rows, cols);
            for x in m.d.iter_mut() {
                *x = C64::new(rng.uniform() - 0.5, rng.uniform() - 0.5);
            }
            let f = svd(&m);
            for r in 0..rows {
                for c in 0..cols {
                    let mut acc = C64::ZERO;
                    for k in 0..f.s.len() {
                        let us = f.u.at(r, k);
                        acc = acc + C64::new(us.re * f.s[k], us.im * f.s[k]) * f.v.at(c, k).conj();
                    }
                    assert!((acc.re - m.at(r, c).re).abs() < 1e-13 && (acc.im - m.at(r, c).im).abs() < 1e-13);
                }
            }
            assert!(f.s.windows(2).all(|w| w[0] >= w[1]), "sorted");
        }
    }
}
