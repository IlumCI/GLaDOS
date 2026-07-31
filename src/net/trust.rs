//! The root certificate store.
//!
//! Roots are loaded from `\GLADOS\roots.der` on the boot volume -- a plain
//! concatenation of DER certificates -- rather than compiled in. Three
//! reasons, in order of importance:
//!
//!   1. **A baked-in root cannot be removed.** Trust stores change: roots get
//!      distrusted, and the whole point of that machinery is that it can
//!      happen without rebuilding every program. A file can be replaced.
//!   2. **It is auditable.** `trust` lists exactly what is trusted, by name
//!      and fingerprint, and the file is the only source. There is nothing
//!      hidden in the binary to go looking for.
//!   3. **It keeps the decision with the user.** Whose roots those are is a
//!      policy question, and this system has exactly one person to answer it.
//!
//! The consequence, stated plainly: **with no roots file, every certificate
//! fails to validate.** That is the correct default. A TLS client that trusts
//! nothing refuses to connect; a TLS client that trusts everything is the
//! thing this whole module exists to stop being.

use super::x509;
use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED, YELLOW};
use crate::kprintln;
use crate::sync::Racy;
use alloc::vec::Vec;

pub const ROOTS_PATH: &str = "\\GLADOS\\roots.der";

struct Root {
    der: Vec<u8>,
    fingerprint: [u8; 32],
    /// The encoded subject, so a certificate's issuer field can be matched
    /// against it without canonicalising names.
    subject: Vec<u8>,
    cn: Vec<u8>,
    is_ca: bool,
}

static ROOTS: Racy<Vec<Root>> = Racy::new(Vec::new());

fn roots() -> &'static mut Vec<Root> {
    unsafe { &mut *ROOTS.get() }
}

/// Split a concatenated DER bundle into individual certificates.
///
/// Each one begins with a SEQUENCE whose length says how long it is, so the
/// boundaries are self-describing and no separator is needed.
fn split_bundle(data: &[u8]) -> Vec<&[u8]> {
    let mut out = Vec::new();
    let mut at = 0;
    while at + 4 <= data.len() {
        if data[at] != 0x30 {
            break;
        }
        let first = data[at + 1];
        let (len, header) = if first & 0x80 == 0 {
            (first as usize, 2)
        } else {
            let n = (first & 0x7F) as usize;
            if n == 0 || n > 4 || at + 2 + n > data.len() {
                break;
            }
            let mut len = 0usize;
            for i in 0..n {
                len = (len << 8) | data[at + 2 + i] as usize;
            }
            (len, 2 + n)
        };
        let end = at + header + len;
        if end > data.len() {
            break;
        }
        out.push(&data[at..end]);
        at = end;
    }
    out
}

/// Read the bundle off the boot volume. Called once, before ExitBootServices.
pub fn load(data: &[u8]) -> usize {
    let list = roots();
    list.clear();
    for der in split_bundle(data) {
        // A root that will not parse is skipped rather than fatal: a bundle
        // exported from a real system contains the occasional oddity, and one
        // bad entry should not cost the other four hundred.
        if let Ok(c) = x509::parse(der) {
            list.push(Root {
                fingerprint: c.fingerprint(),
                subject: c.subject.clone(),
                cn: c.subject_cn.clone(),
                is_ca: c.is_ca,
                der: der.to_vec(),
            });
        }
    }
    list.len()
}

pub fn count() -> usize {
    roots().len()
}

pub fn is_trusted(fingerprint: &[u8; 32]) -> bool {
    roots().iter().any(|r| &r.fingerprint == fingerprint)
}

/// Find a trusted root whose subject matches the given issuer name.
pub fn find_issuer(issuer: &[u8]) -> Option<Vec<u8>> {
    roots()
        .iter()
        .find(|r| r.is_ca && r.subject == issuer)
        .map(|r| r.der.clone())
}

pub fn report() {
    console::set_color(YELLOW);
    kprintln!("[trust]");
    console::set_color(LTGRAY);
    let list = roots();
    if list.is_empty() {
        console::set_color(LTRED);
        kprintln!("  no roots loaded -- every certificate will fail to validate");
        console::set_color(LTGRAY);
        kprintln!("  put a DER bundle at {} and reboot", ROOTS_PATH);
        kprintln!("  scripts/fetch-roots.ps1 builds one from the host's store");
        return;
    }
    console::set_color(LTGREEN);
    kprintln!("  {} root(s) trusted", list.len());
    console::set_color(LTGRAY);
    for r in list.iter().take(12) {
        let name = core::str::from_utf8(&r.cn).unwrap_or("<unnamed>");
        let h = crate::store::sha256::short_hex(&r.fingerprint);
        kprintln!("  {}  {}", core::str::from_utf8(&h).unwrap_or("?"), name);
    }
    if list.len() > 12 {
        kprintln!("  ... and {} more", list.len() - 12);
    }
}
