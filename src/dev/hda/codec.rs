//! A codec's widget graph, and the path from a converter to the speaker.
//!
//! A codec does not say "here is your speaker". It says it has, say, fifty
//! widgets; each answers what kind it is and which others it takes input from;
//! pins answer, from configuration the board maker burned in, what they are
//! wired to. Playing sound is finding a pin wired to something that makes
//! noise, and walking its connection lists back to a digital-to-analog
//! converter, through whatever mixers and selectors stand between.
//!
//! **Everything here asks through a function**, `ask(node, verb) -> answer`,
//! so the walk is the same code whether the answers come from a controller or
//! from a table. The claims hand it two tables: the GF63's own ALC256, as
//! Linux read it off this machine, and QEMU's output codec, which is what the
//! driver meets under emulation. A path that was right for one and not the
//! other would show here rather than as silence on the laptop.

use alloc::vec::Vec;

use super::verb::{self, Kind, PinDefault};

#[derive(Clone, Debug)]
pub struct Widget {
    pub nid: u16,
    pub kind: Kind,
    pub caps: u32,
    pub pin_caps: u32,
    pub pin: Option<PinDefault>,
    pub conns: Vec<u16>,
    pub amp_out: u32,
}

impl Widget {
    pub fn digital(&self) -> bool {
        self.caps & verb::WCAP_DIGITAL != 0
    }
}

#[derive(Clone, Debug)]
pub struct Codec {
    pub addr: u8,
    pub vendor: u32,
    pub afg: u16,
    pub widgets: Vec<Widget>,
}

impl Codec {
    pub fn widget(&self, nid: u16) -> Option<&Widget> {
        self.widgets.iter().find(|w| w.nid == nid)
    }
}

/// The most widgets one codec is walked for. The node count is the codec's to
/// state, and a codec answering 0xFF for everything must not make this ask
/// sixty-five thousand questions.
const MAX_WIDGETS: u32 = 128;
const MAX_CONNS: u32 = 32;

/// Read a codec through `ask`. `None` when it has no audio function group.
pub fn discover(addr: u8, ask: &mut dyn FnMut(u16, u32) -> Option<u32>) -> Option<Codec> {
    let param = |ask: &mut dyn FnMut(u16, u32) -> Option<u32>, nid: u16, p: u8| ask(nid, verb::v12(verb::GET_PARAMETER, p));
    let vendor = param(ask, 0, verb::P_VENDOR)?;
    let root = param(ask, 0, verb::P_NODES)?;
    let (start, count) = ((root >> 16) & 0xFF, root & 0xFF);
    let mut afg = None;
    for nid in start..start + count.min(8) {
        if param(ask, nid as u16, verb::P_FUNCTION_TYPE).map(|t| t & 0xFF) == Some(verb::FG_AUDIO) {
            afg = Some(nid as u16);
            break;
        }
    }
    let afg = afg?;
    let nodes = param(ask, afg, verb::P_NODES)?;
    let (start, count) = ((nodes >> 16) & 0xFF, (nodes & 0xFF).min(MAX_WIDGETS));
    let mut widgets = Vec::new();
    for nid in start..start + count {
        let nid = nid as u16;
        let caps = param(ask, nid, verb::P_WIDGET_CAPS).unwrap_or(0);
        let kind = Kind::of(caps);
        let (pin_caps, pin) = if kind == Kind::Pin {
            (
                param(ask, nid, verb::P_PIN_CAPS).unwrap_or(0),
                ask(nid, verb::v12(verb::GET_CONFIG_DEFAULT, 0)).map(PinDefault::of),
            )
        } else {
            (0, None)
        };
        let conns = if caps & verb::WCAP_CONN_LIST != 0 { connections(ask, nid) } else { Vec::new() };
        let amp_out = if caps & verb::WCAP_AMP_OUT != 0 { param(ask, nid, verb::P_AMP_OUT_CAPS).unwrap_or(0) } else { 0 };
        widgets.push(Widget { nid, kind, caps, pin_caps, pin, conns, amp_out });
    }
    Some(Codec { addr, vendor, afg, widgets })
}

/// A widget's connection list, ranges expanded.
fn connections(ask: &mut dyn FnMut(u16, u32) -> Option<u32>, nid: u16) -> Vec<u16> {
    let Some(len) = ask(nid, verb::v12(verb::GET_PARAMETER, verb::P_CONN_LEN)) else { return Vec::new() };
    let long = len & 0x80 != 0;
    let n = (len & 0x7F).min(MAX_CONNS);
    let per = if long { 2 } else { 4 };
    let mut out: Vec<u16> = Vec::new();
    let mut i = 0;
    while i < n {
        let Some(word) = ask(nid, verb::v12(verb::GET_CONNECTION_ENTRY, i as u8)) else { break };
        for (j, (e, range)) in verb::connections(word, long).into_iter().enumerate() {
            if i + j as u32 >= n {
                break;
            }
            // A range entry means every node from the previous entry to this one.
            match (range, out.last().copied()) {
                (true, Some(prev)) if e > prev && e - prev <= MAX_CONNS as u16 => out.extend(prev + 1..=e),
                _ => out.push(e),
            }
        }
        i += per;
    }
    out
}

/// A route from a converter to a pin: the pin first, the converter last, and
/// for each node after the first, which entry of the node before it leads
/// there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Path {
    pub nodes: Vec<u16>,
    pub picks: Vec<u8>,
}

impl Path {
    pub fn pin(&self) -> u16 {
        self.nodes[0]
    }

    pub fn dac(&self) -> u16 {
        *self.nodes.last().unwrap_or(&0)
    }
}

/// The best output this codec has: the highest-ranked pin that can drive an
/// output and reaches an analog converter.
pub fn output_path(c: &Codec) -> Option<Path> {
    let mut pins: Vec<&Widget> = c
        .widgets
        .iter()
        .filter(|w| w.kind == Kind::Pin && w.pin_caps & verb::PCAP_OUT != 0 && w.pin.and_then(|p| p.rank()).is_some())
        .collect();
    // Highest rank first; among equals the lowest node, so the choice is the
    // same every boot.
    pins.sort_by(|a, b| {
        b.pin.and_then(|p| p.rank()).cmp(&a.pin.and_then(|p| p.rank())).then(a.nid.cmp(&b.nid))
    });
    pins.into_iter().find_map(|p| route(c, p.nid))
}

/// Breadth first from a pin to the nearest analog converter. Depth-capped, and
/// cycle-safe, because a codec's graph is the codec's to describe.
pub fn route(c: &Codec, pin: u16) -> Option<Path> {
    let mut frontier: Vec<Path> = alloc::vec![Path { nodes: alloc::vec![pin], picks: Vec::new() }];
    for _ in 0..6 {
        let mut next = Vec::new();
        for p in frontier {
            let Some(w) = c.widget(*p.nodes.last()?) else { continue };
            for (i, &n) in w.conns.iter().enumerate() {
                if p.nodes.contains(&n) {
                    continue;
                }
                let Some(t) = c.widget(n) else { continue };
                let mut q = p.clone();
                q.nodes.push(n);
                q.picks.push(i as u8);
                match t.kind {
                    Kind::Output if !t.digital() => return Some(q),
                    Kind::Mixer | Kind::Selector => next.push(q),
                    _ => {}
                }
            }
        }
        if next.is_empty() {
            return None;
        }
        frontier = next;
    }
    None
}

/// A fake codec, answering from a table, for the claims.
pub struct Table<'a> {
    /// (nid, widget caps, pin caps, pin config, connections, amp out caps)
    pub widgets: &'a [(u16, u32, u32, u32, &'a [u16], u32)],
    pub vendor: u32,
}

impl Table<'_> {
    pub fn ask(&self, nid: u16, verb: u32) -> Option<u32> {
        let code = verb >> 8;
        let p = (verb & 0xFF) as u8;
        let first = self.widgets.first().map(|w| w.0).unwrap_or(2);
        if nid == 0 {
            return match (code, p) {
                (verb::GET_PARAMETER, verb::P_VENDOR) => Some(self.vendor),
                (verb::GET_PARAMETER, verb::P_NODES) => Some((1 << 16) | 1),
                _ => Some(0),
            };
        }
        if nid == 1 {
            return match (code, p) {
                (verb::GET_PARAMETER, verb::P_FUNCTION_TYPE) => Some(verb::FG_AUDIO),
                (verb::GET_PARAMETER, verb::P_NODES) => {
                    let last = self.widgets.last().map(|w| w.0).unwrap_or(first);
                    Some(((first as u32) << 16) | (last - first + 1) as u32)
                }
                _ => Some(0),
            };
        }
        // A node in range that the table does not describe is an empty slot,
        // which is what a real codec's gaps answer.
        let Some(w) = self.widgets.iter().find(|w| w.0 == nid) else { return Some(0) };
        match (code, p) {
            (verb::GET_PARAMETER, verb::P_WIDGET_CAPS) => Some(w.1),
            (verb::GET_PARAMETER, verb::P_PIN_CAPS) => Some(w.2),
            (verb::GET_PARAMETER, verb::P_CONN_LEN) => Some(w.4.len() as u32),
            (verb::GET_PARAMETER, verb::P_AMP_OUT_CAPS) => Some(w.5),
            (verb::GET_CONFIG_DEFAULT, _) => Some(w.3),
            (verb::GET_CONNECTION_ENTRY, i) => {
                let mut word = 0u32;
                for k in 0..4 {
                    if let Some(&n) = w.4.get(i as usize + k) {
                        word |= (n as u32 & 0xFF) << (k * 8);
                    }
                }
                Some(word)
            }
            _ => Some(0),
        }
    }
}

/// The GF63's ALC256, from `/proc/asound` on this machine: the two analog
/// converters, the internal speaker (fixed, EAPD) on 0x14 fed by 0x02, the
/// headphone jack on 0x21 choosing between 0x02 and 0x03, and two unconnected
/// pins that could drive an output and must not be chosen.
pub const ALC256: &[(u16, u32, u32, u32, &[u16], u32)] = &[
    (0x02, 0x0000_041d, 0, 0, &[], 0x0000_5757),
    (0x03, 0x0000_041d, 0, 0, &[], 0x0000_5757),
    (0x06, 0x0000_0611, 0, 0, &[], 0),
    (0x12, 0x0040_040b, 0x20, 0x4000_0000, &[], 0),
    (0x13, 0x0040_040b, 0x20, 0x4111_11f0, &[], 0),
    (0x14, 0x0040_058d, 0x0001_0014, 0x9017_0110, &[0x02], 0),
    (0x1b, 0x0040_058f, 0x0001_3734, 0x4111_11f0, &[0x02, 0x03], 0),
    (0x1e, 0x0040_0781, 0x10, 0x4111_11f0, &[0x06], 0),
    (0x21, 0x0040_058d, 0x0001_001c, 0x0321_4020, &[0x02, 0x03], 0),
];

/// QEMU's `hda-output` (vendor 1af40012): one converter with an output
/// amplifier, one line-out pin fed by it. The capabilities are what `hda`
/// read off it under emulation; the pin's default configuration is a line
/// out at a jack, which is what it reported itself as.
pub const QEMU_OUTPUT: &[(u16, u32, u32, u32, &[u16], u32)] = &[
    (0x02, 0x0000_001d, 0, 0, &[], 0),
    (0x03, 0x0040_0101, 0x10, 0x0101_4010, &[0x02], 0),
];

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = Vec::new();
    let alc = Table { widgets: ALC256, vendor: 0x10ec_0256 };
    let c = discover(0, &mut |n, v| alc.ask(n, v));
    out.push((
        "the ALC256 table reads as an audio function group with every widget",
        c.as_ref().is_some_and(|c| c.vendor == 0x10ec_0256 && c.afg == 1 && c.widget(0x21).is_some()),
    ));
    let path = c.as_ref().and_then(output_path);
    out.push((
        "the GF63's output is its built-in speaker, pin 0x14, fed by converter 0x02",
        path == Some(Path { nodes: alloc::vec![0x14, 0x02], picks: alloc::vec![0] }),
    ));
    out.push((
        "the headphone jack reaches converter 0x02 through the first of its two entries",
        c.as_ref().and_then(|c| route(c, 0x21)) == Some(Path { nodes: alloc::vec![0x21, 0x02], picks: alloc::vec![0] }),
    ));
    out.push((
        "a pin the board left unconnected is never chosen, though it could drive an output",
        c.as_ref().and_then(output_path).is_some_and(|p| p.pin() != 0x1b && p.pin() != 0x1e),
    ));
    let q = Table { widgets: QEMU_OUTPUT, vendor: 0x1af4_0012 };
    let qc = discover(0, &mut |n, v| q.ask(n, v));
    out.push((
        "QEMU's output codec plays through its line out, pin 0x03 from converter 0x02",
        qc.as_ref().and_then(output_path) == Some(Path { nodes: alloc::vec![0x03, 0x02], picks: alloc::vec![0] }),
    ));
    // A mixer in between, and a cycle a codec might describe.
    const MIXED: &[(u16, u32, u32, u32, &[u16], u32)] = &[
        (0x02, 0x0000_0011, 0, 0, &[], 0),
        (0x0c, 0x0020_0103, 0, 0, &[0x0d, 0x02], 0),
        (0x0d, 0x0020_0103, 0, 0, &[0x0c], 0),
        (0x14, 0x0040_0101, 0x10, 0x9017_0110, &[0x0c], 0),
    ];
    let m = Table { widgets: MIXED, vendor: 1 };
    let mc = discover(0, &mut |n, v| m.ask(n, v));
    out.push((
        "a path through a mixer records which of its inputs leads to the converter, and a cycle ends",
        mc.as_ref().and_then(output_path) == Some(Path { nodes: alloc::vec![0x14, 0x0c, 0x02], picks: alloc::vec![0, 1] }),
    ));
    out
}
