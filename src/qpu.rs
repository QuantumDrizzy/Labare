// ============================================================================
// Unibit — U-QPU phase 1: the reference backend (ADR-0003)
// ============================================================================
//
// An exact state vector in f64 complex, written for clarity, not speed. Every
// later backend (cuStateVec on the GPU, an MPS for the QGPU rung, a real QPU)
// is held to this one. Qubit order is little-endian, as in Qiskit: qubit q is
// bit q of the basis index.
//
// Not claimed: speed, noise, timing. Quantum instructions cost one dispatch
// cycle in the CPU's metrics; durations against coherence are phase 2.
// ============================================================================

/// A complex amplitude.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };

    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    #[inline]
    pub fn conj(self) -> C64 {
        C64::new(self.re, -self.im)
    }
    #[inline]
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
}

impl std::ops::Mul for C64 {
    type Output = C64;
    #[inline]
    fn mul(self, o: C64) -> C64 {
        C64::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

impl std::ops::Add for C64 {
    type Output = C64;
    #[inline]
    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
}

/// A one-qubit gate as a 2×2 matrix, row-major: [[m00, m01], [m10, m11]].
pub type Mat2 = [[C64; 2]; 2];

/// The fixed one-qubit gates of phase 1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QGate {
    H,
    X,
    Y,
    Z,
    S,
    Sdg,
    T,
    Tdg,
}

impl QGate {
    pub const ALL: [QGate; 8] = [QGate::H, QGate::X, QGate::Y, QGate::Z, QGate::S, QGate::Sdg, QGate::T, QGate::Tdg];

    pub fn name(self) -> &'static str {
        match self {
            QGate::H => "h",
            QGate::X => "x",
            QGate::Y => "y",
            QGate::Z => "z",
            QGate::S => "s",
            QGate::Sdg => "sdg",
            QGate::T => "t",
            QGate::Tdg => "tdg",
        }
    }

    pub fn from_name(s: &str) -> Option<QGate> {
        QGate::ALL.into_iter().find(|g| g.name() == s)
    }

    pub fn code(self) -> u8 {
        QGate::ALL.iter().position(|g| *g == self).unwrap() as u8
    }

    pub fn from_code(c: u8) -> Option<QGate> {
        QGate::ALL.get(c as usize).copied()
    }

    /// The textbook matrix.
    pub fn matrix(self) -> Mat2 {
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let o = C64::ZERO;
        let l = C64::ONE;
        match self {
            QGate::H => [[C64::new(r, 0.0), C64::new(r, 0.0)], [C64::new(r, 0.0), C64::new(-r, 0.0)]],
            QGate::X => [[o, l], [l, o]],
            QGate::Y => [[o, C64::new(0.0, -1.0)], [C64::new(0.0, 1.0), o]],
            QGate::Z => [[l, o], [o, C64::new(-1.0, 0.0)]],
            QGate::S => [[l, o], [o, C64::new(0.0, 1.0)]],
            QGate::Sdg => [[l, o], [o, C64::new(0.0, -1.0)]],
            QGate::T => [[l, o], [o, C64::new(r, r)]],
            QGate::Tdg => [[l, o], [o, C64::new(r, -r)]],
        }
    }
}

/// Rotation axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QAxis {
    X,
    Y,
    Z,
}

impl QAxis {
    pub fn name(self) -> &'static str {
        match self {
            QAxis::X => "x",
            QAxis::Y => "y",
            QAxis::Z => "z",
        }
    }
    pub fn from_name(s: &str) -> Option<QAxis> {
        match s {
            "x" => Some(QAxis::X),
            "y" => Some(QAxis::Y),
            "z" => Some(QAxis::Z),
            _ => None,
        }
    }
    pub fn code(self) -> u8 {
        self as u8
    }
    pub fn from_code(c: u8) -> Option<QAxis> {
        [QAxis::X, QAxis::Y, QAxis::Z].get(c as usize).copied()
    }

    /// Rx, Ry, Rz(θ) = exp(-i θ σ/2), the convention Qiskit uses.
    pub fn matrix(self, theta: f64) -> Mat2 {
        let (c, s) = ((theta / 2.0).cos(), (theta / 2.0).sin());
        match self {
            QAxis::X => [[C64::new(c, 0.0), C64::new(0.0, -s)], [C64::new(0.0, -s), C64::new(c, 0.0)]],
            QAxis::Y => [[C64::new(c, 0.0), C64::new(-s, 0.0)], [C64::new(s, 0.0), C64::new(c, 0.0)]],
            QAxis::Z => [[C64::new(c, -s), C64::ZERO], [C64::ZERO, C64::new(c, s)]],
        }
    }
}

/// Largest register the host backend accepts: 2^24 amplitudes × 16 B = 256 MiB.
pub const MAX_QUBITS: u32 = 24;

/// SplitMix64: a small, fixed, documented generator so a measurement sequence
/// replays exactly from its seed.
#[derive(Clone, Debug)]
pub struct QRng {
    state: u64,
}

impl QRng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1) with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

/// The exact state vector.
#[derive(Clone, Debug)]
pub struct StateVector {
    pub n: u32,
    pub amps: Vec<C64>,
    pub rng: QRng,
}

impl StateVector {
    /// |0…0⟩ on n qubits.
    pub fn new(n: u32) -> Result<Self, String> {
        if n == 0 || n > MAX_QUBITS {
            return Err(format!("qalloc: {n} qubits is outside 1..={MAX_QUBITS}"));
        }
        let mut amps = vec![C64::ZERO; 1usize << n];
        amps[0] = C64::ONE;
        Ok(Self { n, amps, rng: QRng::new(0) })
    }

    fn check(&self, q: u32) -> Result<(), String> {
        if q < self.n {
            Ok(())
        } else {
            Err(format!("qubit {q} does not exist in a {}-qubit register", self.n))
        }
    }

    /// Apply a one-qubit matrix to qubit q.
    pub fn apply1(&mut self, q: u32, m: &Mat2) -> Result<(), String> {
        self.check(q)?;
        let bit = 1usize << q;
        for i in 0..self.amps.len() {
            if i & bit != 0 {
                continue;
            }
            let (a0, a1) = (self.amps[i], self.amps[i | bit]);
            self.amps[i] = m[0][0] * a0 + m[0][1] * a1;
            self.amps[i | bit] = m[1][0] * a0 + m[1][1] * a1;
        }
        Ok(())
    }

    /// CNOT: flip target t where control c is 1.
    pub fn cx(&mut self, c: u32, t: u32) -> Result<(), String> {
        self.check(c)?;
        self.check(t)?;
        if c == t {
            return Err("qcx: control and target are the same qubit".into());
        }
        let (cb, tb) = (1usize << c, 1usize << t);
        for i in 0..self.amps.len() {
            if i & cb != 0 && i & tb == 0 {
                self.amps.swap(i, i | tb);
            }
        }
        Ok(())
    }

    /// CZ: phase −1 where both qubits are 1. Symmetric.
    pub fn cz(&mut self, a: u32, b: u32) -> Result<(), String> {
        self.check(a)?;
        self.check(b)?;
        if a == b {
            return Err("qcz: the two qubits are the same".into());
        }
        let mask = (1usize << a) | (1usize << b);
        for (i, amp) in self.amps.iter_mut().enumerate() {
            if i & mask == mask {
                *amp = C64::new(-amp.re, -amp.im);
            }
        }
        Ok(())
    }

    /// Probability that qubit q reads 1.
    pub fn prob_one(&self, q: u32) -> f64 {
        let bit = 1usize << q;
        self.amps.iter().enumerate().filter(|(i, _)| i & bit != 0).map(|(_, a)| a.norm_sqr()).sum()
    }

    /// Measure qubit q in Z: draw the outcome, collapse, renormalise.
    pub fn measure(&mut self, q: u32) -> Result<u8, String> {
        self.check(q)?;
        let p1 = self.prob_one(q);
        let outcome = u8::from(self.rng.uniform() < p1);
        let p = if outcome == 1 { p1 } else { 1.0 - p1 };
        let scale = 1.0 / p.sqrt();
        let bit = 1usize << q;
        for (i, amp) in self.amps.iter_mut().enumerate() {
            if ((i & bit != 0) as u8) == outcome {
                *amp = C64::new(amp.re * scale, amp.im * scale);
            } else {
                *amp = C64::ZERO;
            }
        }
        Ok(outcome)
    }

    /// Measure q and bring it to |0⟩.
    pub fn reset(&mut self, q: u32) -> Result<u8, String> {
        let m = self.measure(q)?;
        if m == 1 {
            self.apply1(q, &QGate::X.matrix())?;
        }
        Ok(m)
    }

    /// Amplitude of a basis index.
    pub fn amp(&self, index: u64) -> Result<C64, String> {
        self.amps
            .get(index as usize)
            .copied()
            .ok_or_else(|| format!("qamp: basis index {index} is outside 0..{}", self.amps.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bell_pair_has_the_two_amplitudes_it_should() {
        let mut s = StateVector::new(2).unwrap();
        s.apply1(0, &QGate::H.matrix()).unwrap();
        s.cx(0, 1).unwrap();
        let r = std::f64::consts::FRAC_1_SQRT_2;
        assert!((s.amps[0].re - r).abs() < 1e-15 && (s.amps[3].re - r).abs() < 1e-15);
        assert_eq!(s.amps[1], C64::ZERO);
        assert_eq!(s.amps[2], C64::ZERO);
    }
}
