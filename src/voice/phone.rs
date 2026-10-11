//! The 39 phones of ARPAbet, and what each one sounds like to a formant
//! synthesiser.
//!
//! The order is `tools/lexicon.py`'s `PHONES` exactly, because a packed
//! phone is an index into it -- reorder one and the whole lexicon says
//! different words.
//!
//! Formants are Peterson & Barney (1952)'s vowel averages and the textbook
//! loci for consonants, measured on men, scaled by `FEMALE` -- a shorter
//! vocal tract puts every resonance higher, about 15%. Fricatives are a
//! noise band at a centre and width, which is the whole of what a formant
//! synthesiser can say about them.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum P {
    AA, AE, AH, AO, AW, AY, EH, ER, EY, IH, IY, OW, OY, UH, UW,
    B, CH, D, DH, F, G, HH, JH, K, L, M, N, NG, P, R, S, SH,
    T, TH, V, W, Y, Z, ZH,
}

const ALL: [P; 39] = [
    P::AA, P::AE, P::AH, P::AO, P::AW, P::AY, P::EH, P::ER, P::EY, P::IH, P::IY, P::OW, P::OY, P::UH, P::UW,
    P::B, P::CH, P::D, P::DH, P::F, P::G, P::HH, P::JH, P::K, P::L, P::M, P::N, P::NG, P::P, P::R, P::S, P::SH,
    P::T, P::TH, P::V, P::W, P::Y, P::Z, P::ZH,
];

const NAMES: [&str; 39] = [
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH",
    "T", "TH", "V", "W", "Y", "Z", "ZH",
];

impl P {
    pub fn name(self) -> &'static str {
        NAMES[self as usize]
    }
    pub fn vowel(self) -> bool {
        (self as u8) < 15
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Phone {
    pub p: P,
    /// 0 unstressed, 1 primary, 2 secondary, 3 a consonant.
    pub stress: u8,
}

impl Phone {
    pub fn new(p: P, stress: u8) -> Phone {
        Phone { p, stress }
    }
    pub fn from_code(c: u8) -> Phone {
        Phone { p: ALL[((c & 63) as usize).min(38)], stress: c >> 6 }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Manner {
    Vowel,
    Approximant,
    Nasal,
    Fricative,
    Stop,
    Affricate,
    Aspirate,
}

/// One phone's acoustic target.
#[derive(Clone, Copy)]
pub struct Target {
    pub manner: Manner,
    pub voiced: bool,
    /// F1..F3 at the start and, for a diphthong, the end. Hz, male.
    pub f: [f32; 3],
    pub f_end: [f32; 3],
    /// Frication: centre and bandwidth (Hz), and level 0..1.
    pub fric: (f32, f32, f32),
    /// Duration in ms, stressed; unstressed vowels take about 60% of it.
    pub ms: f32,
}

/// Male to female formant scale.
pub const FEMALE: f32 = 1.15;

const fn v(f1: f32, f2: f32, f3: f32, ms: f32) -> Target {
    Target { manner: Manner::Vowel, voiced: true, f: [f1, f2, f3], f_end: [f1, f2, f3], fric: (0.0, 0.0, 0.0), ms }
}
const fn d(a: [f32; 3], b: [f32; 3], ms: f32) -> Target {
    Target { manner: Manner::Vowel, voiced: true, f: a, f_end: b, fric: (0.0, 0.0, 0.0), ms }
}
const fn ap(f1: f32, f2: f32, f3: f32, ms: f32) -> Target {
    Target { manner: Manner::Approximant, voiced: true, f: [f1, f2, f3], f_end: [f1, f2, f3], fric: (0.0, 0.0, 0.0), ms }
}
const fn nas(f2: f32, f3: f32) -> Target {
    Target { manner: Manner::Nasal, voiced: true, f: [270.0, f2, f3], f_end: [270.0, f2, f3], fric: (0.0, 0.0, 0.0), ms: 65.0 }
}
const fn fr(voiced: bool, f2: f32, centre: f32, bw: f32, level: f32, ms: f32) -> Target {
    Target {
        manner: Manner::Fricative,
        voiced,
        f: [300.0, f2, 2600.0],
        f_end: [300.0, f2, 2600.0],
        fric: (centre, bw, level),
        ms,
    }
}
const fn st(voiced: bool, f2: f32, burst: f32) -> Target {
    Target {
        manner: Manner::Stop,
        voiced,
        f: [250.0, f2, 2600.0],
        f_end: [250.0, f2, 2600.0],
        fric: (burst, 1500.0, 0.6),
        ms: if voiced { 60.0 } else { 75.0 },
    }
}

pub fn target(p: P) -> Target {
    match p {
        P::AA => v(730.0, 1090.0, 2440.0, 140.0),
        P::AE => v(660.0, 1720.0, 2410.0, 140.0),
        P::AH => v(520.0, 1190.0, 2390.0, 100.0),
        P::AO => v(570.0, 840.0, 2410.0, 140.0),
        P::AW => d([730.0, 1090.0, 2440.0], [300.0, 870.0, 2240.0], 180.0),
        P::AY => d([730.0, 1090.0, 2440.0], [300.0, 2200.0, 2900.0], 180.0),
        P::EH => v(530.0, 1840.0, 2480.0, 120.0),
        P::ER => v(490.0, 1350.0, 1690.0, 130.0),
        P::EY => d([480.0, 1900.0, 2500.0], [300.0, 2250.0, 2950.0], 160.0),
        P::IH => v(390.0, 1990.0, 2550.0, 100.0),
        P::IY => v(270.0, 2290.0, 3010.0, 130.0),
        P::OW => d([570.0, 900.0, 2400.0], [330.0, 800.0, 2250.0], 160.0),
        P::OY => d([570.0, 840.0, 2410.0], [300.0, 2200.0, 2900.0], 190.0),
        P::UH => v(440.0, 1020.0, 2240.0, 110.0),
        P::UW => v(300.0, 870.0, 2240.0, 130.0),
        P::L => ap(360.0, 1300.0, 2700.0, 70.0),
        P::R => ap(330.0, 1060.0, 1380.0, 70.0),
        P::W => ap(290.0, 610.0, 2150.0, 60.0),
        P::Y => ap(260.0, 2070.0, 3020.0, 60.0),
        P::M => nas(1000.0, 2200.0),
        P::N => nas(1600.0, 2600.0),
        P::NG => nas(2000.0, 2700.0),
        P::S => fr(false, 1600.0, 6000.0, 2000.0, 0.8, 110.0),
        P::Z => fr(true, 1600.0, 6000.0, 2000.0, 0.45, 90.0),
        P::SH => fr(false, 1800.0, 3200.0, 1500.0, 0.8, 110.0),
        P::ZH => fr(true, 1800.0, 3200.0, 1500.0, 0.45, 90.0),
        P::F => fr(false, 1100.0, 7000.0, 4000.0, 0.25, 100.0),
        P::V => fr(true, 1100.0, 7000.0, 4000.0, 0.15, 70.0),
        P::TH => fr(false, 1400.0, 6500.0, 4000.0, 0.2, 100.0),
        P::DH => fr(true, 1400.0, 6500.0, 4000.0, 0.12, 50.0),
        P::HH => Target {
            manner: Manner::Aspirate,
            voiced: false,
            f: [500.0, 1500.0, 2500.0],
            f_end: [500.0, 1500.0, 2500.0],
            fric: (0.0, 0.0, 0.0),
            ms: 70.0,
        },
        P::P => st(false, 900.0, 1000.0),
        P::B => st(true, 900.0, 1000.0),
        P::T => st(false, 1700.0, 4500.0),
        P::D => st(true, 1700.0, 4500.0),
        P::K => st(false, 2000.0, 2200.0),
        P::G => st(true, 2000.0, 2200.0),
        P::CH => Target {
            manner: Manner::Affricate,
            voiced: false,
            f: [300.0, 1800.0, 2600.0],
            f_end: [300.0, 1800.0, 2600.0],
            fric: (3200.0, 1500.0, 0.8),
            ms: 120.0,
        },
        P::JH => Target {
            manner: Manner::Affricate,
            voiced: true,
            f: [300.0, 1800.0, 2600.0],
            f_end: [300.0, 1800.0, 2600.0],
            fric: (3200.0, 1500.0, 0.5),
            ms: 100.0,
        },
    }
}
