//! Cryptographic primitives, written here rather than vendored.
//!
//! This is the one place in GLaDOS where "written from scratch" is a liability
//! rather than a virtue, and it is worth being blunt about why. Everywhere
//! else, a bug shows up as something not working. Here, a bug shows up as
//! something that works perfectly and is not secure -- the wrong key schedule
//! still produces bytes, the leaky comparison still returns the right answer,
//! the biased nonce still encrypts. There is no test that fails.
//!
//! So the primitives were chosen for checkability over performance:
//!
//!   * **SHA-256** already existed for the content-addressed store.
//!   * **HMAC/HKDF** are a hundred lines of structure over it.
//!   * **ChaCha20-Poly1305** rather than AES-GCM, because a software AES is a
//!     table lookup indexed by secret data and this has no tables at all.
//!   * **X25519** rather than a NIST curve, because the Montgomery ladder does
//!     identical work per scalar bit and needs no point validation beyond
//!     rejecting the all-zero output.
//!
//! Every one of them is checked against the published test vectors from its
//! RFC at boot, and `crypto` reruns them on demand. That establishes
//! correctness against the standard. It does not establish that the
//! implementation is free of side channels, and nothing here should be trusted
//! with anything that matters.

pub mod aes;
pub mod bigint;
pub mod chacha;
pub mod hkdf;
pub mod p256;
pub mod rsa;
pub mod sha1;
pub mod sha512;
pub mod x25519;

/// Run every vector. Called at boot, and by the `crypto` command.
pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, YELLOW};
    use crate::kprintln;

    console::set_color(YELLOW);
    kprintln!("\n[selftest] crypto:");

    let checks: [(&str, fn() -> bool); 17] = [
        ("sha-256   NIST vectors", crate::store::sha256::selftest),
        ("hmac/hkdf RFC 4231 and RFC 5869", hkdf::selftest),
        ("sha-384   FIPS 180-4", sha512::selftest),
        ("bigint    montgomery modexp", bigint::selftest),
        ("rsa       pkcs#1 shape and mgf1", rsa::selftest),
        ("ecdsa     FIPS 186-4 P-256, tamper rejected", p256::selftest),
        ("chacha20  RFC 8439, and a flipped bit is rejected", chacha::selftest),
        ("x25519    RFC 7748, and both sides agree", x25519::selftest),
        ("sha-1     FIPS 180-1, RFC 2202, RFC 6070", sha1::selftest),
        ("aes       FIPS 197 and RFC 3394 key wrap", aes::selftest),
        ("wpa2      IEEE 802.11i pmk and ptk", crate::net::wpa2::selftest),
        ("802.11    beacons parse, probe requests build", crate::net::ieee80211::selftest),
        ("8188eu    tx/rx descriptor bit layout", crate::dev::rtl8188eu::desc::selftest),
        ("lang      functions, lists, scope, whole programs", crate::lang::selftest),
        ("uidoc     panels round-trip as text, bad ones refused", crate::gfx::uidoc::selftest),
        ("app       manifests identify, and lineage is in the hash", crate::app::manifest::selftest),
        ("appcheck  a panel naming a missing function is caught", crate::app::check::selftest),
    ];

    let mut all = true;
    for (name, f) in checks {
        let ok = f();
        all &= ok;
        console::set_color(if ok { LTGREEN } else { LTRED });
        kprintln!("  {}   {}", if ok { "ok  " } else { "FAIL" }, name);
    }
    console::set_color(LTGRAY);
    all
}
