//! The cryptography an 802.11 stack needs, and nothing else.
//!
//! Every primitive here already exists in `crate::crypto`; this module is the
//! narrow re-export a ported tree is allowed to name. It is deliberately not a
//! general crypto surface -- there is no ChaCha20 here, no X25519, no ECDSA,
//! because 802.11 does not negotiate any of them and a seam that offers what
//! its consumer cannot use is a seam nobody can reason about.
//!
//! ### What each is for, since the names do not say
//!
//!   * **`ccm_encrypt` / `ccm_decrypt`** are CCMP, the data path. This is the
//!     one the temporal key finally gets used for.
//!   * **`cmac`** is BIP, which protects management frames under 802.11w, and
//!     EAPOL-Key descriptor version 3.
//!   * **`pmk`** is PBKDF2-HMAC-SHA1 over the passphrase and the SSID, 4096
//!     rounds. Slow on purpose and slow here too: roughly 8,192 SHA-1
//!     compressions, which is the cost of the offline dictionary resistance
//!     WPA2-PSK has instead of a real key exchange.
//!   * **`prf`** is the 802.11 PRF that expands the PMK into a PTK.
//!   * **`key_unwrap`** is RFC 3394, for the GTK arriving inside message 3.

use alloc::vec::Vec;

/// CCMP's MIC length and length-field width, so a caller never spells them.
pub use crate::crypto::ccm::{CCMP_L, CCMP_MIC_LEN};

/// Encrypt and authenticate one frame body. Answers `(ciphertext, mic)`.
pub fn ccm_encrypt(
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
    plain: &[u8],
) -> Option<(Vec<u8>, Vec<u8>)> {
    crate::crypto::ccm::encrypt(key, nonce, aad, plain, CCMP_MIC_LEN, CCMP_L)
}

/// Decrypt and verify. `None` means the MIC failed and the plaintext is gone
/// with it, which is the only safe thing to do with it.
pub fn ccm_decrypt(
    key: &[u8],
    nonce: &[u8],
    aad: &[u8],
    cipher: &[u8],
    mic: &[u8],
) -> Option<Vec<u8>> {
    crate::crypto::ccm::decrypt(key, nonce, aad, cipher, mic, CCMP_L)
}

/// AES-CMAC, for BIP and for EAPOL-Key descriptor version 3.
pub fn cmac(key: &[u8], msg: &[u8]) -> Option<[u8; 16]> {
    crate::crypto::ccm::cmac(key, msg)
}

/// HMAC-SHA1, which the EAPOL-Key MIC of descriptor version 2 is a prefix of.
pub fn hmac_sha1(key: &[u8], data: &[u8]) -> [u8; 20] {
    crate::crypto::sha1::hmac(key, data)
}

/// PBKDF2-HMAC-SHA1, 4096 rounds, SSID as salt: the WPA2-PSK master key.
pub fn pmk(passphrase: &[u8], ssid: &[u8]) -> Vec<u8> {
    crate::crypto::sha1::pbkdf2(passphrase, ssid, 4096, 32)
}

/// RFC 3394 AES key unwrap, for the GTK carried in message 3.
pub fn key_unwrap(kek: &[u8], wrapped: &[u8]) -> Option<Vec<u8>> {
    crate::crypto::aes::key_unwrap(kek, wrapped)
}
