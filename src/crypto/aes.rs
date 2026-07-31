//! AES-128 and AES-256, plus RFC 3394 key wrap.
//!
//! ### The timing caveat, stated up front
//!
//! This is a table-driven AES. The S-box lookup is indexed by a byte that
//! depends on the key, so which cache line is touched depends on the key, and
//! anything able to measure cache timing on this machine can recover it. That
//! is the exact objection that made ChaCha20 the right choice for TLS.
//!
//! It is here anyway because WPA2 has no alternative: CCMP is AES and a
//! supplicant does not get to negotiate something else. The mitigating facts
//! are that GLaDOS runs one program on one core with nothing else to do the
//! measuring, and that the alternative -- no wireless at all -- is worse. If
//! this system ever runs untrusted code, this becomes a real problem and the
//! answer is AES-NI, which this CPU has.

use alloc::vec::Vec;

const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

fn inv_sbox() -> [u8; 256] {
    let mut inv = [0u8; 256];
    for i in 0..256 {
        inv[SBOX[i] as usize] = i as u8;
    }
    inv
}

/// Multiply in GF(2^8) with the AES polynomial.
fn xtime(a: u8) -> u8 {
    if a & 0x80 != 0 {
        (a << 1) ^ 0x1B
    } else {
        a << 1
    }
}

fn gmul(mut a: u8, mut b: u8) -> u8 {
    let mut p = 0u8;
    for _ in 0..8 {
        if b & 1 != 0 {
            p ^= a;
        }
        a = xtime(a);
        b >>= 1;
    }
    p
}

pub struct Aes {
    round_keys: Vec<[u8; 16]>,
    rounds: usize,
}

impl Aes {
    pub fn new(key: &[u8]) -> Option<Aes> {
        let (nk, rounds) = match key.len() {
            16 => (4usize, 10usize),
            32 => (8, 14),
            _ => return None,
        };
        let total = 4 * (rounds + 1);
        let mut w = alloc::vec![[0u8; 4]; total];
        for i in 0..nk {
            w[i] = [key[4 * i], key[4 * i + 1], key[4 * i + 2], key[4 * i + 3]];
        }
        let mut rcon = 1u8;
        for i in nk..total {
            let mut t = w[i - 1];
            if i % nk == 0 {
                t = [
                    SBOX[t[1] as usize] ^ rcon,
                    SBOX[t[2] as usize],
                    SBOX[t[3] as usize],
                    SBOX[t[0] as usize],
                ];
                rcon = xtime(rcon);
            } else if nk > 6 && i % nk == 4 {
                t = [
                    SBOX[t[0] as usize],
                    SBOX[t[1] as usize],
                    SBOX[t[2] as usize],
                    SBOX[t[3] as usize],
                ];
            }
            for k in 0..4 {
                w[i][k] = w[i - nk][k] ^ t[k];
            }
        }

        let mut round_keys = Vec::with_capacity(rounds + 1);
        for r in 0..=rounds {
            let mut rk = [0u8; 16];
            for c in 0..4 {
                rk[c * 4..c * 4 + 4].copy_from_slice(&w[r * 4 + c]);
            }
            round_keys.push(rk);
        }
        Some(Aes { round_keys, rounds })
    }

    pub fn encrypt_block(&self, block: &mut [u8; 16]) {
        for i in 0..16 {
            block[i] ^= self.round_keys[0][i];
        }
        for r in 1..=self.rounds {
            for b in block.iter_mut() {
                *b = SBOX[*b as usize];
            }
            shift_rows(block);
            if r != self.rounds {
                mix_columns(block);
            }
            for i in 0..16 {
                block[i] ^= self.round_keys[r][i];
            }
        }
    }

    pub fn decrypt_block(&self, block: &mut [u8; 16]) {
        let inv = inv_sbox();
        for i in 0..16 {
            block[i] ^= self.round_keys[self.rounds][i];
        }
        for r in (1..=self.rounds).rev() {
            inv_shift_rows(block);
            for b in block.iter_mut() {
                *b = inv[*b as usize];
            }
            for i in 0..16 {
                block[i] ^= self.round_keys[r - 1][i];
            }
            if r != 1 {
                inv_mix_columns(block);
            }
        }
    }
}

fn shift_rows(s: &mut [u8; 16]) {
    let t = *s;
    for r in 1..4 {
        for c in 0..4 {
            s[c * 4 + r] = t[((c + r) % 4) * 4 + r];
        }
    }
}

fn inv_shift_rows(s: &mut [u8; 16]) {
    let t = *s;
    for r in 1..4 {
        for c in 0..4 {
            s[((c + r) % 4) * 4 + r] = t[c * 4 + r];
        }
    }
}

fn mix_columns(s: &mut [u8; 16]) {
    for c in 0..4 {
        let a: [u8; 4] = [s[c * 4], s[c * 4 + 1], s[c * 4 + 2], s[c * 4 + 3]];
        s[c * 4] = gmul(a[0], 2) ^ gmul(a[1], 3) ^ a[2] ^ a[3];
        s[c * 4 + 1] = a[0] ^ gmul(a[1], 2) ^ gmul(a[2], 3) ^ a[3];
        s[c * 4 + 2] = a[0] ^ a[1] ^ gmul(a[2], 2) ^ gmul(a[3], 3);
        s[c * 4 + 3] = gmul(a[0], 3) ^ a[1] ^ a[2] ^ gmul(a[3], 2);
    }
}

fn inv_mix_columns(s: &mut [u8; 16]) {
    for c in 0..4 {
        let a: [u8; 4] = [s[c * 4], s[c * 4 + 1], s[c * 4 + 2], s[c * 4 + 3]];
        s[c * 4] = gmul(a[0], 14) ^ gmul(a[1], 11) ^ gmul(a[2], 13) ^ gmul(a[3], 9);
        s[c * 4 + 1] = gmul(a[0], 9) ^ gmul(a[1], 14) ^ gmul(a[2], 11) ^ gmul(a[3], 13);
        s[c * 4 + 2] = gmul(a[0], 13) ^ gmul(a[1], 9) ^ gmul(a[2], 14) ^ gmul(a[3], 11);
        s[c * 4 + 3] = gmul(a[0], 11) ^ gmul(a[1], 13) ^ gmul(a[2], 9) ^ gmul(a[3], 14);
    }
}

/// RFC 3394 AES key unwrap, which is how WPA2 delivers the group key.
///
/// The integrity check value is fixed and known, so a wrong key is detected
/// rather than producing plausible garbage -- which is the entire reason key
/// wrap exists instead of plain ECB.
pub fn key_unwrap(kek: &[u8], wrapped: &[u8]) -> Option<Vec<u8>> {
    if wrapped.len() < 16 || wrapped.len() % 8 != 0 {
        return None;
    }
    let aes = Aes::new(kek)?;
    let n = wrapped.len() / 8 - 1;
    let mut a = [0u8; 8];
    a.copy_from_slice(&wrapped[..8]);
    let mut r: Vec<[u8; 8]> = (0..n)
        .map(|i| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&wrapped[8 * (i + 1)..8 * (i + 2)]);
            b
        })
        .collect();

    for j in (0..6).rev() {
        for i in (1..=n).rev() {
            let t = (n * j + i) as u64;
            let mut block = [0u8; 16];
            for k in 0..8 {
                block[k] = a[k];
            }
            // The counter is xored into the low bytes of A before decryption.
            let tb = t.to_be_bytes();
            for k in 0..8 {
                block[k] ^= tb[k];
            }
            block[8..16].copy_from_slice(&r[i - 1]);
            aes.decrypt_block(&mut block);
            a.copy_from_slice(&block[..8]);
            r[i - 1].copy_from_slice(&block[8..16]);
        }
    }

    // The default IV from RFC 3394. Anything else means the KEK was wrong.
    if a != [0xA6; 8] {
        return None;
    }
    let mut out = Vec::with_capacity(n * 8);
    for b in r {
        out.extend_from_slice(&b);
    }
    Some(out)
}

pub fn selftest() -> bool {
    // FIPS 197 Appendix C.1, AES-128.
    let key: [u8; 16] = core::array::from_fn(|i| i as u8);
    let mut block: [u8; 16] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee,
        0xff,
    ];
    let want: [u8; 16] = [
        0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30, 0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5,
        0x5a,
    ];
    let aes = match Aes::new(&key) {
        None => return false,
        Some(a) => a,
    };
    aes.encrypt_block(&mut block);
    if block != want {
        return false;
    }
    // Decryption must invert it, which catches a wrong inverse table.
    aes.decrypt_block(&mut block);
    if block[..4] != [0x00, 0x11, 0x22, 0x33] {
        return false;
    }

    // RFC 3394 section 4.1: 128-bit key wrapped with a 128-bit KEK.
    let kek: [u8; 16] = core::array::from_fn(|i| i as u8);
    let wrapped: [u8; 24] = [
        0x1F, 0xA6, 0x8B, 0x0A, 0x81, 0x12, 0xB4, 0x47, 0xAE, 0xF3, 0x4B, 0xD8, 0xFB, 0x5A, 0x7B,
        0x82, 0x9D, 0x3E, 0x86, 0x23, 0x71, 0xD2, 0xCF, 0xE5,
    ];
    let want_key: [u8; 16] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
        0xFF,
    ];
    match key_unwrap(&kek, &wrapped) {
        None => false,
        Some(k) => {
            if k[..] != want_key[..] {
                return false;
            }
            // A wrong KEK must be rejected by the integrity check rather than
            // returning garbage.
            let mut bad = kek;
            bad[0] ^= 1;
            key_unwrap(&bad, &wrapped).is_none()
        }
    }
}
