//! The firmware container Intel ships, parsed.
//!
//! An `iwlwifi-*.ucode` file is a small header followed by a stream of
//! type-length-value records: the microcode sections that go into the device's
//! memory, the version, how many CPUs the image drives, and several dozen
//! capability records a loader may ignore. The format is `iwl-fw-file.h`'s and
//! the type numbers here are the ones that file defines.
//!
//! **This walks and never seeks, and asserts it lands on the last byte.** That
//! is the bargain `tools/v4.py` and `convert.py` already make for the checkpoint
//! format, for the identical reason: the body carries no names, shapes or
//! lengths beyond each record's own, so a reader that disagreed with the writer
//! about one length would leave everything after it as perfectly valid records
//! of the wrong thing. A firmware image is worse than a checkpoint there. A
//! misparsed section is not a section that fails to load; it is the right number
//! of bytes written to the wrong address in a radio's memory, and what comes
//! back is silence.
//!
//! So every refusal is explicit and nothing is skipped past. A record whose
//! length runs off the end is an error, not a truncation; a trailing fragment
//! too short to be a record is an error, not padding; and a container that
//! parses to the last byte but declares no loadable section is an error too,
//! because a loader that ran it would reset the device and wait for firmware
//! that was never sent.
//!
//! Nothing here touches hardware, which is why it is the last piece of the
//! Intel port that can be checked without the laptop. `tools/iwxfw.py` is the
//! second reader -- it writes a container as well as reading one, so the pair
//! round-trips, the way `tokenizer.py --verify` diffs the kernel's tokeniser
//! against the reference library.

use alloc::string::String;
use alloc::vec::Vec;

/// `"IWL\n"` little-endian. Checked, never assumed.
pub const MAGIC: u32 = 0x0a4c_5749;

/// Four zero bytes first. The v1 header began with a version word, and no valid
/// combination of major/minor/API/serial is zero -- so a leading zero is how a
/// TLV container is told from the format it replaced. A reader that skipped this
/// would parse a v1 file as a TLV one and find garbage records.
pub const HEADER: usize = 88;
const HUMAN_READABLE: usize = 64;

/// The record types this kernel acts on. Everything else is carried as
/// `Other` rather than dropped, because "the image contained a record we
/// ignored" and "the image contained nothing we understood" are different facts
/// and only the second is a refusal.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A runtime microcode section: four bytes of destination offset, then the
    /// bytes to put there.
    SecRt,
    /// The same for the initialisation image, which runs first and is replaced.
    SecInit,
    /// How many CPUs the image drives. Decides where the section list splits.
    NumOfCpu,
    /// One word: the chains the firmware drives, in bits 16..20 (transmit) and
    /// 20..24 (receive), which the NVM's masks narrow.
    PhySku,
    /// One word: how many channels a scan request may carry.
    NScanChannels,
    /// A human-readable version, separate from the header's.
    Version,
    /// The image loader, on families that bootstrap through one.
    Iml,
    /// One word of the API bitmap, with its index.
    Api,
    /// One word of the capability bitmap, with its index.
    Capa,
    /// Paged memory, for images larger than the device's own RAM.
    Paging,
    /// Which version of each command and notification this firmware speaks.
    CmdVersions,
    /// Everything else, by number. **Most of a real image is this**, and that is
    /// not a gap: the AX201's own `iwlwifi-QuZ-a0-hr-b0-77.ucode` carries 181
    /// records of which 98 are `IWL_UCODE_TLV_DEBUG_BASE` and up -- 0x1000005
    /// onward, the debug-region descriptors a loader ignores unless it is
    /// tracing. Carrying them is what lets a transcript say what an image
    /// contained; naming them would be a table to keep in step with Intel for no
    /// behaviour.
    Other(u32),
}

impl Kind {
    pub fn of(t: u32) -> Kind {
        match t {
            19 => Kind::SecRt,
            20 => Kind::SecInit,
            27 => Kind::NumOfCpu,
            23 => Kind::PhySku,
            31 => Kind::NScanChannels,
            36 => Kind::Version,
            52 => Kind::Iml,
            29 => Kind::Api,
            30 => Kind::Capa,
            32 => Kind::Paging,
            48 => Kind::CmdVersions,
            other => Kind::Other(other),
        }
    }
    pub fn raw(&self) -> u32 {
        match self {
            Kind::SecRt => 19,
            Kind::SecInit => 20,
            Kind::NumOfCpu => 27,
            Kind::PhySku => 23,
            Kind::NScanChannels => 31,
            Kind::Version => 36,
            Kind::Iml => 52,
            Kind::Api => 29,
            Kind::Capa => 30,
            Kind::Paging => 32,
            Kind::CmdVersions => 48,
            Kind::Other(t) => *t,
        }
    }
    /// Whether this record is bytes destined for the device's memory.
    pub fn loadable(&self) -> bool {
        matches!(self, Kind::SecRt | Kind::SecInit)
    }
}

/// Two offsets that are markers rather than addresses.
///
/// A section list is split by sentinels instead of being counted, so a loader
/// that treated these as destinations would write eight bytes to an address near
/// the top of the map and then load the second CPU's code over the first's.
pub const CPU1_CPU2_SEPARATOR: u32 = 0xFFFF_CCCC;
pub const PAGING_SEPARATOR: u32 = 0xAAAA_BBBB;

/// One microcode section: where it goes, and how long it is.
///
/// The bytes are left in the caller's buffer and described by range rather than
/// copied. A runtime image is over a megabyte and this runs before there is much
/// heap; copying it to describe it would double the peak for nothing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Section {
    pub kind: Kind,
    /// Destination in device memory, or one of the two separators.
    pub offset: u32,
    /// Where the bytes are in the file.
    pub at: usize,
    pub len: usize,
}

impl Section {
    pub fn is_separator(&self) -> bool {
        self.offset == CPU1_CPU2_SEPARATOR || self.offset == PAGING_SEPARATOR
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Error {
    /// Shorter than the fixed header.
    TooShort(usize),
    /// The leading word was not zero: this is a v1/v2 image, not a TLV one.
    NotTlv(u32),
    /// The magic was wrong. Names what was found, because a byte-swapped file
    /// and a wrong file are different mistakes.
    BadMagic(u32),
    /// A record's length runs past the end of the file. Carries the record
    /// index, its declared length and what was left, because "the file is
    /// truncated" and "a length is wrong" look identical without all three.
    Overrun { at: usize, want: usize, left: usize },
    /// Bytes remain, but too few to be a record.
    Trailing(usize),
    /// A section record too short to hold its own destination offset.
    ShortSection { at: usize, len: usize },
    /// Parsed to the last byte and found nothing to load.
    NoSections,
    /// A bitmap word claimed an index the declared width does not have.
    BadBitmapIndex { at: usize, index: usize, words: usize },
    /// A second command-version table, which upstream refuses: which one
    /// governs has no answer.
    TwoVersionTables(usize),
    /// A record that must be one word was not. Upstream's EINVAL, for the PHY
    /// SKU and the scan-channel count: a different length is a different layout.
    BadWord { at: usize, len: usize },
    /// More scan channels than a scan request has room for. Upstream's ERANGE.
    TooManyScanChannels(u32),
}

impl Error {
    pub fn why(&self) -> String {
        match self {
            Error::TooShort(n) => alloc::format!("{} bytes is shorter than the {}-byte header", n, HEADER),
            Error::NotTlv(v) => alloc::format!("leading word {:#010x} is not zero, so this is a v1 image", v),
            Error::BadMagic(m) => alloc::format!("magic {:#010x} is not {:#010x}", m, MAGIC),
            Error::BadWord { at, len } => {
                alloc::format!("record {} should be one four-byte word and is {} bytes", at, len)
            }
            Error::TooManyScanChannels(n) => {
                alloc::format!("{} scan channels declared, past the {} a request holds", n, MAX_SCAN_CHANNELS)
            }
            Error::TwoVersionTables(at) => {
                alloc::format!("record {} is a second command-version table", at)
            }
            Error::Overrun { at, want, left } => {
                alloc::format!("record {} declares {} bytes with {} left", at, want, left)
            }
            Error::Trailing(n) => alloc::format!("{} byte(s) left over, too few for a record", n),
            Error::ShortSection { at, len } => {
                alloc::format!("record {} is a section of {} bytes, too short for a destination", at, len)
            }
            Error::NoSections => String::from("parsed whole, and declares nothing to load"),
            Error::BadBitmapIndex { at, index, words } => alloc::format!(
                "record {} declares bitmap word {} where there are {}",
                at, index, words
            ),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Image {
    /// The header's own version string, NUL-trimmed.
    pub human: String,
    pub ver: u32,
    pub build: u32,
    pub sections: Vec<Section>,
    /// How many CPUs the image drives, if it said.
    pub cpus: Option<u32>,
    /// Which optional behaviours this firmware has, as a bitmap.
    ///
    /// **Three of the configuration commands are gated on it**, so a driver
    /// without it either sends a command firmware will reject or skips one it
    /// wanted. 160 bits in five words, which is upstream's own width.
    pub capa: [u32; CAPA_WORDS],
    /// Which API revisions it implements. 128 bits in four words.
    pub api: [u32; API_WORDS],
    /// Where the image loader is in the file, as `(at, len)`.
    ///
    /// **Located rather than copied, like the sections**, and reported rather
    /// than merely counted: `Kind::Iml` was recognised from the day this parser
    /// was written and its bytes were unreachable, so an AX210 part -- which
    /// boots *through* the loader and has no fallback -- could not have been
    /// started from an image this had parsed perfectly.
    pub iml: Option<(usize, usize)>,
    /// Every record type seen, in order, including the ignored ones. Kept so a
    /// transcript can say what an image contained rather than what was used.
    pub records: Vec<(Kind, usize)>,
    /// `(group, command, command version, notification version)`, as declared.
    ///
    /// **A capability bit says a feature exists; this says what shape its
    /// command has**, and the scan request alone has had a dozen. A driver that
    /// sends the layout it was written for to firmware expecting another gets a
    /// command error at best and a scan configured from misread fields at worst.
    pub cmd_versions: Vec<(u8, u8, u8, u8)>,
    /// The PHY SKU word, if the image carried one.
    pub phy_config: Option<u32>,
    /// How many channels a scan may name, if the image said.
    pub n_scan_channels: Option<u32>,
}

/// What a scan request holds at most, upstream's `IWX_MAX_SCAN_CHANNELS`.
pub const MAX_SCAN_CHANNELS: u32 = 67;
/// What a firmware that does not say is taken to allow, upstream's default.
pub const DEFAULT_SCAN_CHANNELS: u32 = 40;

fn narrow(fw: Option<u8>, nvm: u8) -> u8 {
    // A firmware with no PHY SKU record leaves the NVM to decide alone, rather
    // than upstream's zero-initialised word, which would mask every chain off.
    let fw = fw.unwrap_or(0xf);
    if nvm != 0 { fw & nvm } else { fw }
}

impl Image {
    /// The transmit chains to use: the firmware's, narrowed by the NVM's when
    /// the NVM names any. Upstream's `iwx_fw_valid_tx_ant`. An NVM reporting
    /// zero left this driver sending a zero mask -- no chain at all.
    pub fn valid_tx_ant(&self, nvm: u8) -> u8 {
        narrow(self.phy_config.map(|c| ((c >> 16) & 0xf) as u8), nvm)
    }

    /// And the receive chains, the same way.
    pub fn valid_rx_ant(&self, nvm: u8) -> u8 {
        narrow(self.phy_config.map(|c| ((c >> 20) & 0xf) as u8), nvm)
    }

    /// How many channels a scan may name.
    pub fn scan_channels(&self) -> usize {
        self.n_scan_channels.unwrap_or(DEFAULT_SCAN_CHANNELS) as usize
    }

    /// The version of a command this firmware declares, if it declares one.
    /// `None` is upstream's `IWX_FW_CMD_VER_UNKNOWN`, and callers treat it as
    /// the oldest layout, which is what firmware old enough not to say speaks.
    pub fn cmd_ver(&self, group: u8, cmd: u8) -> Option<u8> {
        self.cmd_versions.iter().find(|v| v.0 == group && v.1 == cmd).map(|v| v.2)
    }

    /// The version of the notification or response that command produces.
    pub fn notif_ver(&self, group: u8, cmd: u8) -> Option<u8> {
        self.cmd_versions.iter().find(|v| v.0 == group && v.1 == cmd).map(|v| v.3)
    }

    /// The sections that are really destinations, with the separators removed.
    pub fn loadable(&self) -> Vec<Section> {
        self.sections.iter().copied().filter(|s| !s.is_separator()).collect()
    }
    pub fn bytes(&self) -> usize {
        self.loadable().iter().map(|s| s.len).sum()
    }

    /// Does this firmware have capability `n`?
    ///
    /// A bit past the declared width answers **false** rather than indexing out
    /// of range. That is the safe direction: a capability nobody declared is one
    /// not to rely on, where a panic in a fault-free path would take the machine
    /// for a question about an optional feature.
    pub fn has_capa(&self, n: usize) -> bool {
        self.capa.get(n / 32).map(|w| w & (1 << (n % 32)) != 0).unwrap_or(false)
    }

    /// Does it implement API revision `n`?
    pub fn has_api(&self, n: usize) -> bool {
        self.api.get(n / 32).map(|w| w & (1 << (n % 32)) != 0).unwrap_or(false)
    }
}

/// How wide the two bitmaps are, in words. Upstream declares 160 capability bits
/// and 128 API bits; both are rounded up to a whole word.
pub const CAPA_WORDS: usize = (160 + 31) / 32;
pub const API_WORDS: usize = (128 + 31) / 32;

/// The capability bits anything in this tree reads. Named rather than numbered at
/// the use site, because a bare 74 in a condition is a number nobody can check.
pub mod capa {
    /// Firmware applies a learned regulatory profile.
    pub const LAR_SUPPORT: usize = 1;
    /// The regulatory reply is the twenty-byte version 4, not the sixteen-byte
    /// version 3.
    pub const MCC_UPDATE_11AX_SUPPORT: usize = 89;
    /// Dynamic queue allocation, which the queue-enable command depends on.
    pub const DQA_SUPPORT: usize = 12;
    /// Firmware handles the critical-temperature shutdown itself.
    pub const CT_KILL_BY_FW: usize = 74;
    /// The multi-link API. Named because upstream refuses to trust it on this
    /// family -- its comment says the API-77 firmware on some older devices claims
    /// support and does not work -- so a driver reading the bit alone would enable
    /// a path Intel's own driver declines.
    pub const MLD_API_SUPPORT: usize = 110;
}

/// The API revisions anything here reads.
pub mod api {
    /// Version 4 of the NVM response, with 110 channels rather than 51.
    pub const REGULATORY_NVM_INFO: usize = 48;
    pub const REDUCED_SCAN_CONFIG: usize = 56;
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Parse a container.
///
/// Walks from the header to the last byte and refuses anything that does not
/// land exactly there. `len` is `u32` on the wire and every arithmetic step is
/// checked, because a record declaring `0xFFFFFFFF` must be a refusal and not a
/// wrapped addition that reads as a short record.
pub fn parse(b: &[u8]) -> Result<Image, Error> {
    if b.len() < HEADER {
        return Err(Error::TooShort(b.len()));
    }
    let zero = le32(b, 0);
    if zero != 0 {
        return Err(Error::NotTlv(zero));
    }
    let magic = le32(b, 4);
    if magic != MAGIC {
        return Err(Error::BadMagic(magic));
    }
    let human = {
        let raw = &b[8..8 + HUMAN_READABLE];
        let end = raw.iter().position(|&c| c == 0).unwrap_or(HUMAN_READABLE);
        // Non-UTF-8 is a version string somebody mangled, not a reason to refuse
        // a firmware image: it is decoration, and the loader uses the records.
        String::from_utf8_lossy(&raw[..end]).into_owned()
    };
    let ver = le32(b, 8 + HUMAN_READABLE);
    let build = le32(b, 12 + HUMAN_READABLE);

    let mut sections: Vec<Section> = Vec::new();
    let mut records: Vec<(Kind, usize)> = Vec::new();
    let mut cpus = None;
    let mut iml = None;
    let mut capa = [0u32; CAPA_WORDS];
    let mut api = [0u32; API_WORDS];
    let mut cmd_versions: Vec<(u8, u8, u8, u8)> = Vec::new();
    let mut phy_config = None;
    let mut n_scan_channels = None;
    let mut at = HEADER;
    let mut n = 0usize;

    while at < b.len() {
        let left = b.len() - at;
        // A record is at least its own type and length.
        if left < 8 {
            return Err(Error::Trailing(left));
        }
        let kind = Kind::of(le32(b, at));
        let len = le32(b, at + 4) as usize;
        let body = at + 8;
        // Checked rather than added: `len` is attacker-shaped, and `body + len`
        // on a 32-bit length in a `usize` could wrap on a small target.
        if len > b.len().saturating_sub(body) {
            return Err(Error::Overrun { at: n, want: len, left: b.len() - body });
        }
        records.push((kind, len));
        match kind {
            Kind::SecRt | Kind::SecInit => {
                if len < 4 {
                    return Err(Error::ShortSection { at: n, len });
                }
                sections.push(Section {
                    kind,
                    offset: le32(b, body),
                    at: body + 4,
                    len: len - 4,
                });
            }
            Kind::NumOfCpu if len >= 4 => cpus = Some(le32(b, body)),
            Kind::PhySku | Kind::NScanChannels if len != 4 => return Err(Error::BadWord { at: n, len }),
            Kind::PhySku => phy_config = Some(le32(b, body)),
            Kind::NScanChannels => {
                let v = le32(b, body);
                if v > MAX_SCAN_CHANNELS {
                    return Err(Error::TooManyScanChannels(v));
                }
                n_scan_channels = Some(v);
            }
            // **The last one wins**, which is upstream's rule: it frees any
            // earlier loader and keeps the newest. A file with two is malformed
            // and the choice still has to be defined, because "the first" and
            // "the last" are different blobs and only one of them will run.
            Kind::Iml if len > 0 => iml = Some((body, len)),
            // **Both bitmaps arrive one word at a time, each carrying its own
            // index.** A record is eight bytes -- an index then the word -- and
            // there is one record per populated word, so a reader that took the
            // first and stopped would see only capabilities 0 to 31. An index past
            // the declared width is refused rather than dropped, which is
            // upstream's own answer: a firmware declaring capability 200 is one
            // this cannot reason about, and silently ignoring the word would leave
            // a driver confident about a bitmap it had not fully read.
            Kind::Capa if len == 8 => {
                let i = le32(b, body) as usize;
                if i >= CAPA_WORDS {
                    return Err(Error::BadBitmapIndex { at: n, index: i, words: CAPA_WORDS });
                }
                capa[i] = le32(b, body + 4);
            }
            // Four bytes an entry: command, group, command version, notification
            // version -- command *before* group, which is the order the struct
            // has and the opposite of how every lookup is spelt. A trailing
            // partial entry is dropped, as upstream drops it. Two such records
            // is refused, as upstream refuses it: which one governs is not a
            // question with an answer.
            Kind::CmdVersions => {
                if !cmd_versions.is_empty() {
                    return Err(Error::TwoVersionTables(n));
                }
                for e in b[body..body + len].chunks_exact(4) {
                    cmd_versions.push((e[1], e[0], e[2], e[3]));
                }
            }
            Kind::Api if len == 8 => {
                let i = le32(b, body) as usize;
                if i >= API_WORDS {
                    return Err(Error::BadBitmapIndex { at: n, index: i, words: API_WORDS });
                }
                api[i] = le32(b, body + 4);
            }
            _ => {}
        }
        // **Padded to four, and the padding is part of the record.** Rounding
        // up is how the next record is found; a reader that advanced by `len`
        // alone would land one to three bytes early and read a type out of the
        // middle of the previous body.
        let step = 8 + ((len + 3) & !3);
        // The same overrun check again, because the padding can be the thing
        // that runs off the end -- a last record whose body fits and whose
        // padding does not.
        if step > left {
            return Err(Error::Overrun { at: n, want: step, left });
        }
        at += step;
        n += 1;
    }
    if at != b.len() {
        return Err(Error::Trailing(b.len() - at));
    }
    if sections.is_empty() {
        return Err(Error::NoSections);
    }
    Ok(Image { human, ver, build, sections, cpus, capa, api, iml, records, cmd_versions, phy_config, n_scan_channels })
}

/// Build a container, for the suite and for nothing else.
///
/// **A writer in the kernel, which needs one only to test the reader.** It is
/// here rather than in the host tool because a claim that parses a container the
/// suite itself built is checking the reader against the format, where a claim
/// over a file on disk would be checking it against whatever happened to be
/// staged. `tools/iwxfw.py` has the same writer for the same reason, and the two
/// agreeing is the round trip that makes either trustworthy.
pub fn build(human: &str, ver: u32, build_no: u32, recs: &[(u32, Vec<u8>)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&MAGIC.to_le_bytes());
    let mut h = [0u8; HUMAN_READABLE];
    for (i, c) in human.bytes().take(HUMAN_READABLE).enumerate() {
        h[i] = c;
    }
    out.extend_from_slice(&h);
    out.extend_from_slice(&ver.to_le_bytes());
    out.extend_from_slice(&build_no.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    debug_assert_eq!(out.len(), HEADER);
    for (t, body) in recs {
        out.extend_from_slice(&t.to_le_bytes());
        out.extend_from_slice(&(body.len() as u32).to_le_bytes());
        out.extend_from_slice(body);
        while out.len() % 4 != 0 {
            out.push(0);
        }
    }
    out
}

/// A section record's body: the destination, then the bytes.
pub fn section_body(offset: u32, data: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + data.len());
    v.extend_from_slice(&offset.to_le_bytes());
    v.extend_from_slice(data);
    v
}

/// One bitmap record: an index then the word, both little-endian.
///
/// A helper because the record is bytes and the bits are `u32`, and writing
/// `1 << 12` straight into a byte vector is a compile error rather than the bit
/// somebody meant -- which the first version of these claims was.
fn bitmap_word(index: u32, word: u32) -> Vec<u8> {
    let mut v = Vec::with_capacity(8);
    v.extend_from_slice(&index.to_le_bytes());
    v.extend_from_slice(&word.to_le_bytes());
    v
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut claim = |what: &'static str, ok: bool| out.push((what, ok));

    // A container of the shape a real one has: two CPUs' runtime sections with
    // the separator between them, a cpu count, and a record this kernel ignores.
    let img = build(
        "77.1a2b3c4d.0 QuZ-a0-hr-b0-77",
        0x0100_0000,
        4242,
        &[
            (27, 2u32.to_le_bytes().to_vec()),
            (19, section_body(0x0080_0000, &[0xAA; 16])),
            (19, section_body(CPU1_CPU2_SEPARATOR, &[])),
            (19, section_body(0x0040_0000, &[0xBB; 12])),
            (61, b"phy integration".to_vec()),
        ],
    );
    let p = parse(&img);
    claim("a well-formed container parses", p.is_ok());
    let p = p.unwrap_or_else(|_| Image {
        human: String::new(), ver: 0, build: 0,
        sections: Vec::new(), cpus: None, capa: [0; CAPA_WORDS], api: [0; API_WORDS],
        iml: None, records: Vec::new(), cmd_versions: Vec::new(),
        phy_config: None, n_scan_channels: None,
    });
    claim("the version string is read and NUL-trimmed", p.human == "77.1a2b3c4d.0 QuZ-a0-hr-b0-77");
    claim("the build number survives", p.build == 4242);
    claim("the cpu count is taken from its own record", p.cpus == Some(2));
    claim("every record is remembered, including the ignored one", p.records.len() == 5);

    // --- the PHY SKU and the scan-channel count -----------------------------
    let sec = (19u32, section_body(0x0040_0000, &[0xBB; 12]));
    let with = |extra: Vec<(u32, Vec<u8>)>| {
        let mut r = alloc::vec![sec.clone()];
        r.extend(extra);
        parse(&build("x", 0, 0, &r))
    };
    // Transmit chains A and B, receive A only.
    let sku = with(alloc::vec![(23, ((0x3u32 << 16) | (0x1u32 << 20)).to_le_bytes().to_vec()), (31, 33u32.to_le_bytes().to_vec())]);
    claim(
        "the PHY SKU's chains are read and the NVM narrows them",
        sku.as_ref().map(|i| (i.valid_tx_ant(0), i.valid_tx_ant(0x2), i.valid_rx_ant(0), i.scan_channels())).ok()
            == Some((0x3, 0x2, 0x1, 33)),
    );
    claim(
        "an image with neither leaves the NVM to decide, and allows upstream's forty",
        with(alloc::vec![]).map(|i| (i.valid_tx_ant(0x1), i.scan_channels())).ok() == Some((0x1, 40)),
    );
    claim(
        "a PHY SKU that is not one word is refused, as upstream refuses it",
        matches!(with(alloc::vec![(23, alloc::vec![0; 8])]), Err(Error::BadWord { .. })),
    );
    claim(
        "and so is a scan-channel count past what a request holds",
        with(alloc::vec![(31, 68u32.to_le_bytes().to_vec())]) == Err(Error::TooManyScanChannels(68)),
    );

    // --- the command versions ----------------------------------------------
    // Command before group in each entry: the struct's order, and the opposite
    // of how every lookup is spelt, so a reader that took them the other way
    // would answer for the wrong command and still look right.
    let with_versions = build("v", 1, 1, &[
        (19, alloc::vec![0; 8]),
        (48, alloc::vec![0x0c, 0x01, 5, 0, 0x0d, 0x01, 16, 2, 0xc8, 0x01, 1, 7, 0xff]),
    ]);
    let v = parse(&with_versions);
    claim(
        "a command-version table is read command-first, and a trailing partial entry dropped",
        v.as_ref().map(|i| {
            (i.cmd_ver(0x01, 0x0c), i.cmd_ver(0x01, 0x0d), i.notif_ver(0x01, 0xc8), i.cmd_versions.len())
        }) == Ok((Some(5), Some(16), Some(7), 3)),
    );
    claim(
        "and a command it does not list has no version rather than zero",
        v.as_ref().map(|i| i.cmd_ver(0x01, 0x0e)) == Ok(None),
    );
    claim(
        "a second table is refused, as upstream refuses it",
        matches!(
            parse(&build("v", 1, 1, &[(19, alloc::vec![0; 8]), (48, alloc::vec![0; 4]), (48, alloc::vec![0; 4])])),
            Err(Error::TwoVersionTables(2))
        ),
    );

    // --- the two bitmaps ----------------------------------------------------

    // Widths, which are upstream's and are not the same number.
    claim("the capability bitmap is five words", CAPA_WORDS == 5);
    claim("and the API bitmap four", API_WORDS == 4);

    // **One record per word, each carrying its index**, which is the shape a
    // reader gets wrong by taking the first and stopping.
    let bits = build(
        "bitmaps",
        1,
        1,
        &[
            (19, section_body(0x1000, &[1])),
            // Capability 12 is in word 0, and 74 is in word 2 -- so a reader that
            // stopped at the first record would see DQA and miss CT-kill.
            (30, bitmap_word(0, 1 << 12)),
            (30, bitmap_word(2, 1 << (74 - 64))),
            (29, bitmap_word(1, 1 << (48 - 32))),
        ],
    );
    match parse(&bits) {
        Ok(i) => {
            claim("a capability in word zero is read", i.has_capa(capa::DQA_SUPPORT));
            // The one that catches a reader stopping after the first record: this
            // bit lives in word two.
            claim("and one in word two, from a second record", i.has_capa(capa::CT_KILL_BY_FW));
            claim("an undeclared capability is absent", !i.has_capa(capa::MLD_API_SUPPORT));
            claim("the API bitmap is separate from it", i.has_api(api::REGULATORY_NVM_INFO));
            claim("and does not answer for a capability of the same number", !i.has_capa(api::REGULATORY_NVM_INFO));
            // A bit past the declared width answers false rather than indexing out
            // of range, which in a fault-free path would take the machine for a
            // question about an optional feature.
            claim("a bit past the bitmap answers no rather than panicking", !i.has_capa(4096));
            claim("and so does one past the API bitmap", !i.has_api(4096));
        }
        Err(_) => claim("an image carrying bitmaps parses", false),
    }
    // An index the width does not have is refused. Dropping it would leave a
    // driver confident about a bitmap it had not fully read.
    let wide = build("too wide", 1, 1, &[
        (19, section_body(0x1000, &[1])),
        (30, bitmap_word(9, 1)),
    ]);
    claim(
        "a bitmap word past the declared width is refused by name",
        matches!(parse(&wide), Err(Error::BadBitmapIndex { index: 9, words: 5, .. })),
    );
    // A record of the wrong length is carried and not acted on, which is the
    // ordinary "recognised and ignored" path rather than a refusal: upstream
    // refuses, and here the arm simply does not match, so the bitmap stays as it
    // was. Asserted so that stays deliberate.
    let short = build("short capa", 1, 1, &[
        (19, section_body(0x1000, &[1])),
        (30, alloc::vec![0, 0, 0, 0]),
    ]);
    claim(
        "a bitmap record of the wrong length leaves the bitmap alone",
        parse(&short).map(|i| i.capa) == Ok([0; CAPA_WORDS]),
    );
    claim(
        "a record this kernel does not act on is carried, not dropped",
        p.records.iter().any(|(k, _)| *k == Kind::Other(61)),
    );

    // **The separator is not a destination**, which is the one mistake here that
    // would load the second CPU's code over the first's.
    claim("three section records are seen", p.sections.len() == 3);
    claim("but only two are loadable", p.loadable().len() == 2);
    claim("and the separator is recognised as one", p.sections[1].is_separator());
    claim("the loadable bytes are the section bodies alone", p.bytes() == 16 + 12);
    claim(
        "a section's destination comes from its first four bytes",
        p.loadable()[0].offset == 0x0080_0000 && p.loadable()[1].offset == 0x0040_0000,
    );
    claim(
        "and its bytes are described by range, not copied",
        img[p.loadable()[0].at] == 0xAA && p.loadable()[0].len == 16,
    );

    // --- the refusals, which are the reason the walk is exact ---------------
    claim("a short buffer is refused", matches!(parse(&[0u8; 8]), Err(Error::TooShort(8))));
    // A v1 image begins with a version word. Parsing one as TLV finds garbage.
    let mut v1 = img.clone();
    v1[0] = 1;
    claim("a v1 image is refused by its leading word", matches!(parse(&v1), Err(Error::NotTlv(1))));
    let mut bad = img.clone();
    bad[4] ^= 0xFF;
    claim("a wrong magic is refused and reported", matches!(parse(&bad), Err(Error::BadMagic(_))));

    // A length that runs off the end. This is the one a lenient reader treats as
    // truncation and loads anyway.
    let mut over = img.clone();
    let l = over.len();
    over[HEADER + 4..HEADER + 8].copy_from_slice(&((l as u32) * 2).to_le_bytes());
    claim("a record longer than the file is refused", matches!(parse(&over), Err(Error::Overrun { .. })));
    // And the extreme of it, which must not wrap.
    let mut huge = img.clone();
    huge[HEADER + 4..HEADER + 8].copy_from_slice(&u32::MAX.to_le_bytes());
    claim(
        "a length of 0xFFFFFFFF is refused rather than wrapping",
        matches!(parse(&huge), Err(Error::Overrun { .. })),
    );

    // A fragment too short to be a record is an error, not padding. A reader
    // that ignored it would accept a file cut mid-record.
    let mut frag = img.clone();
    frag.extend_from_slice(&[0, 0, 0, 0]);
    claim("a trailing fragment is refused", matches!(parse(&frag), Err(Error::Trailing(4))));

    // A section too short to hold its own offset.
    let short = build("x", 1, 1, &[(19, alloc::vec![0u8; 2])]);
    claim(
        "a section shorter than its destination is refused",
        matches!(parse(&short), Err(Error::ShortSection { .. })),
    );

    // **Parses whole and loads nothing.** A loader given this would reset the
    // radio and wait for firmware that was never sent, which is the failure that
    // looks like dead hardware.
    let empty = build("x", 1, 1, &[(61, b"only a capability".to_vec())]);
    claim("a container with nothing to load is refused", matches!(parse(&empty), Err(Error::NoSections)));

    // Padding is part of the record. A body of 13 bytes is followed by three
    // bytes of padding, and a reader advancing by 13 would read a type out of
    // the middle of the previous body.
    let odd = build(
        "x", 1, 1,
        &[(19, section_body(0x1000, &[0xCD; 9])), (19, section_body(0x2000, &[0xEF; 4]))],
    );
    let q = parse(&odd);
    claim("an odd-length record is padded to four and still parses", q.is_ok());
    claim(
        "and the record after it is found at the right place",
        q.map(|i| i.loadable().len() == 2 && i.loadable()[1].offset == 0x2000).unwrap_or(false),
    );

    // --- welded to the second reader ----------------------------------------
    //
    // **A writer and a reader that only ever meet their own output both "work".**
    // `parse(build(..))` succeeding proves they agree with each other and says
    // nothing about whether either agrees with Intel's format. So the container
    // this suite builds is pinned to the one `tools/iwxfw.py` builds from the
    // same arguments: same length, same digest, byte for byte.
    //
    // If either side's padding, field order or header size drifts, this claim
    // fails and the other reader is the one that says why -- which is the whole
    // point of there being two, and the bargain `tokenizer.py --verify` and
    // `v4.py` already make. The digest is `tools/iwxfw.py --selftest`'s fixture,
    // and `--emit` writes it out to be compared by hand.
    claim("the container is the length the second reader produces", img.len() == 188);
    claim(
        "and the same bytes, digest for digest",
        crate::store::sha256::hash(&img)
            == [
                0x46, 0x9c, 0x40, 0x31, 0x69, 0x19, 0xd3, 0x9d,
                0x03, 0x0d, 0x4a, 0x15, 0xc1, 0x91, 0x46, 0x8f,
                0x31, 0x78, 0x50, 0xf7, 0xdc, 0xf1, 0xf0, 0x32,
                0x33, 0xeb, 0xff, 0x85, 0xfe, 0xcc, 0xa7, 0xef,
            ],
    );
    claim("and it is a whole number of words", img.len() % 4 == 0);
    out
}
