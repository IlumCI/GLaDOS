//! Intel High Definition Audio: the controller, its codecs, and a stream out.
//!
//! The one sound path nearly every PC since 2004 has. The controller is a DMA
//! engine with a published register map; the codecs hanging off its link do
//! the conversion and own the jacks. Talking to a codec means writing
//! commands into a ring in memory (the CORB) and reading answers out of
//! another (the RIRB); playing sound means pointing a stream descriptor at a
//! list of buffers and telling a converter which stream to listen to.
//!
//! ### Why this before any codec work
//!
//! There was no sound at all. A media codec is a userspace library -- a Linux
//! program brings its own -- and what none of them can do without is a
//! device to hand samples to. This is that device.
//!
//! ### The GF63
//!
//! Its speakers are on a Realtek ALC256 behind Alder Lake's controller,
//! `8086:51c8`. Linux runs that controller in DSP mode (Sound Open Firmware),
//! which is why it reports class 04/01 rather than HD Audio's 04/03; the HD
//! Audio registers are still there, and Linux's own `dsp_driver=1` drives it
//! as a plain controller. What that loses is the digital microphone, which is
//! a DSP feature. Recognised by id here for that reason. Untested on the
//! laptop; QEMU's `ich9-intel-hda` is the controller every claim and every
//! recording below was made against.
//!
//! ### Polled, like everything else here
//!
//! Commands are answered within microseconds and a stream runs from its
//! buffer without being asked, so nothing here takes an interrupt. A stream
//! loops its buffer until it is stopped.

pub mod codec;
pub mod verb;

use alloc::string::String;
use alloc::vec::Vec;
use core::ptr::{read_volatile, write_volatile};

use crate::dev::dma::Dma;
use crate::dev::pci::{self, Device};
use crate::sync::Spin;

// Global registers.
const GCAP: u64 = 0x00;
const VMIN: u64 = 0x02;
const VMAJ: u64 = 0x03;
const GCTL: u64 = 0x08;
const STATESTS: u64 = 0x0E;
const INTCTL: u64 = 0x20;
const CORBLBASE: u64 = 0x40;
const CORBUBASE: u64 = 0x44;
const CORBWP: u64 = 0x48;
const CORBRP: u64 = 0x4A;
const CORBCTL: u64 = 0x4C;
const CORBSIZE: u64 = 0x4E;
const RIRBLBASE: u64 = 0x50;
const RIRBUBASE: u64 = 0x54;
const RIRBWP: u64 = 0x58;
const RINTCNT: u64 = 0x5A;
const RIRBCTL: u64 = 0x5C;
const RIRBSTS: u64 = 0x5D;
const RIRBSIZE: u64 = 0x5E;
const ICOI: u64 = 0x60;
const ICII: u64 = 0x64;
const ICIS: u64 = 0x68;
const DPLBASE: u64 = 0x70;
const DPUBASE: u64 = 0x74;

// Stream descriptor registers, from the descriptor's own base.
const SD_BASE: u64 = 0x80;
const SD_SIZE: u64 = 0x20;
const SD_CTL: u64 = 0x00;
const SD_STS: u64 = 0x03;
const SD_LPIB: u64 = 0x04;
const SD_CBL: u64 = 0x08;
const SD_LVI: u64 = 0x0C;
const SD_FMT: u64 = 0x12;
const SD_BDPL: u64 = 0x18;
const SD_BDPU: u64 = 0x1C;

const CTL_SRST: u32 = 1 << 0;
const CTL_RUN: u32 = 1 << 1;

/// The tag a stream is known to its converter by. One stream plays at a time,
/// so one tag; zero is reserved.
const TAG: u8 = 1;

/// Intel controllers that report class 04/01 because their DSP is enabled and
/// that are HD Audio underneath. Recognised by id rather than by class, since
/// class 04/01 is also every other multimedia audio device there is.
pub const INTEL_DSP_IDS: &[u16] = &[0x51c8, 0x51cc, 0x51cd, 0x54c8, 0x7ad0, 0x7a50, 0xa0c8, 0x43c8, 0x4dc8, 0x02c8, 0x06c8, 0x9dc8, 0xa348];

/// Every HD Audio controller on the bus, as found.
pub fn find(ecam: u64) -> Vec<Device> {
    let mut out = Vec::new();
    pci::scan(ecam, 255, |d| {
        let hda = d.class == 0x04 && d.subclass == 0x03;
        let dsp = d.class == 0x04 && d.subclass == 0x01 && d.vendor == 0x8086 && INTEL_DSP_IDS.contains(&d.device);
        if hda || dsp {
            out.push(d);
        }
    });
    out
}

/// A controller, brought up, with its codecs read.
pub struct Hda {
    pub dev: Device,
    bar: u64,
    /// `ManuallyDrop` so `Drop` decides: freed once bus mastering reads back
    /// off, kept otherwise -- the radio driver's rule, for the same reason.
    corb: core::mem::ManuallyDrop<Dma>,
    rirb: core::mem::ManuallyDrop<Dma>,
    entries: u16,
    rirb_rp: u16,
    /// Whether commands go through the rings or the immediate interface.
    immediate: bool,
    pub inputs: u8,
    pub outputs: u8,
    pub version: (u8, u8),
    /// Which link addresses answered the reset (STATESTS).
    pub present: u16,
    pub codecs: Vec<codec::Codec>,
    stream: Option<Stream>,
    ecam: u64,
}

struct Stream {
    sd: u64,
    codec: u8,
    path: codec::Path,
    _bdl: Dma,
    _buf: Dma,
}

/// Why a controller would not come up.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Fault {
    NoAperture,
    NotMapped,
    NoReset,
    NoMemory,
    NoCodec,
    NoCommand,
    NoOutput,
    NoStream,
    BadFormat,
}

impl Fault {
    pub fn why(self) -> &'static str {
        match self {
            Fault::NoAperture => "the controller has no memory aperture",
            Fault::NotMapped => "its aperture could not be mapped",
            Fault::NoReset => "it would not come out of reset",
            Fault::NoMemory => "no memory for its command rings",
            Fault::NoCodec => "no codec answered on its link",
            Fault::NoCommand => "a codec is on the link and does not answer commands",
            Fault::NoOutput => "no codec has a speaker, headphone or line out wired to a converter",
            Fault::NoStream => "the controller has no output stream",
            Fault::BadFormat => "that rate or depth is not one this driver plays",
        }
    }
}

fn r8(b: u64, o: u64) -> u8 {
    unsafe { read_volatile((b + o) as *const u8) }
}
fn r16(b: u64, o: u64) -> u16 {
    unsafe { read_volatile((b + o) as *const u16) }
}
fn r32(b: u64, o: u64) -> u32 {
    unsafe { read_volatile((b + o) as *const u32) }
}
fn w8(b: u64, o: u64, v: u8) {
    unsafe { write_volatile((b + o) as *mut u8, v) }
}
fn w16(b: u64, o: u64, v: u16) {
    unsafe { write_volatile((b + o) as *mut u16, v) }
}
fn w32(b: u64, o: u64, v: u32) {
    unsafe { write_volatile((b + o) as *mut u32, v) }
}

/// Poll until `f` holds, for up to `us` microseconds.
fn wait(us: u64, mut f: impl FnMut() -> bool) -> bool {
    let mut waited = 0;
    loop {
        if f() {
            return true;
        }
        if waited >= us {
            return false;
        }
        crate::time::delay_us(10);
        waited += 10;
    }
}

impl Hda {
    /// Reset the controller, set up its command rings, and read every codec.
    pub fn up(ecam: u64, dev: Device) -> Result<Hda, Fault> {
        let bar = pci::bar(ecam, &dev, 0).filter(|&b| b != 0).ok_or(Fault::NoAperture)?;
        if !crate::mem::paging::map_range(bar, 0x4000, true) {
            return Err(Fault::NotMapped);
        }
        let _ = pci::set_d0(ecam, &dev);
        pci::enable_bus_master(ecam, &dev);
        // Intel's controllers: traffic class 0 for the link, which the
        // specification's own note and every driver set (TCSEL, bits 2:0 of
        // config offset 0x44). Left at whatever firmware chose, playback on
        // some chipsets crackles or stalls.
        if dev.vendor == 0x8086 {
            let v = pci::cfg_read32(ecam, &dev, 0x44);
            pci::cfg_write32(ecam, &dev, 0x44, v & !0x7);
        }

        // Out of reset and back in: the only way to a known state, since the
        // firmware may have left streams running.
        w32(bar, GCTL, r32(bar, GCTL) & !1);
        if !wait(100_000, || r32(bar, GCTL) & 1 == 0) {
            return Err(Fault::NoReset);
        }
        crate::time::delay_us(100);
        w32(bar, GCTL, r32(bar, GCTL) | 1);
        if !wait(100_000, || r32(bar, GCTL) & 1 == 1) {
            return Err(Fault::NoReset);
        }
        // A codec takes 25 frames -- 521 us -- to make itself known.
        crate::time::delay_us(1_000);
        let present = r16(bar, STATESTS);
        let gcap = r16(bar, GCAP);
        let outputs = ((gcap >> 12) & 0xF) as u8;
        let inputs = ((gcap >> 8) & 0xF) as u8;
        w32(bar, INTCTL, 0);
        w32(bar, DPLBASE, 0);
        w32(bar, DPUBASE, 0);

        let corb = Dma::new(1024, 128).ok_or(Fault::NoMemory)?;
        let rirb = Dma::new(2048, 128).ok_or(Fault::NoMemory)?;
        let mut h = Hda {
            dev,
            bar,
            corb: core::mem::ManuallyDrop::new(corb),
            rirb: core::mem::ManuallyDrop::new(rirb),
            entries: 256,
            rirb_rp: 0,
            immediate: false,
            inputs,
            outputs,
            version: (r8(bar, VMAJ), r8(bar, VMIN)),
            present,
            codecs: Vec::new(),
            stream: None,
            ecam,
        };
        if !h.rings() {
            // The rings are what every controller has; the immediate
            // interface is optional. A controller whose rings would not run
            // is still asked the immediate way before being given up on.
            h.immediate = true;
        }
        for addr in 0..15u8 {
            if present & (1 << addr) == 0 {
                continue;
            }
            let found = {
                let hh = &mut h;
                codec::discover(addr, &mut |nid, verb| hh.ask(addr, nid, verb))
            };
            if let Some(c) = found {
                h.codecs.push(c);
            }
        }
        if present == 0 {
            return Err(Fault::NoCodec);
        }
        // A codec on the link that answered nothing comes up anyway, so
        // `describe` can show the ring registers that say why.
        Ok(h)
    }

    /// Stop both rings, size them, point them at memory, and start them.
    fn rings(&mut self) -> bool {
        let b = self.bar;
        w8(b, CORBCTL, 0);
        w8(b, RIRBCTL, 0);
        if !wait(10_000, || r8(b, CORBCTL) & 2 == 0 && r8(b, RIRBCTL) & 2 == 0) {
            return false;
        }
        // The largest size each supports: 256 entries, 16, or 2.
        let pick = |cap: u8| -> (u8, u16) {
            if cap & 0x40 != 0 {
                (2, 256)
            } else if cap & 0x20 != 0 {
                (1, 16)
            } else {
                (0, 2)
            }
        };
        let (cs, ce) = pick(r8(b, CORBSIZE) >> 0);
        let (rs, re) = pick(r8(b, RIRBSIZE));
        if ce != re {
            return false;
        }
        self.entries = ce;
        w8(b, CORBSIZE, (r8(b, CORBSIZE) & !3) | cs);
        w8(b, RIRBSIZE, (r8(b, RIRBSIZE) & !3) | rs);
        let cp = self.corb.pa();
        let rp = self.rirb.pa();
        w32(b, CORBLBASE, cp as u32);
        w32(b, CORBUBASE, (cp >> 32) as u32);
        w32(b, RIRBLBASE, rp as u32);
        w32(b, RIRBUBASE, (rp >> 32) as u32);
        // The read pointer's reset is a handshake: set, see it set, clear, see
        // it clear. Controllers differ in how faithfully they echo it, so a
        // missing echo is waited for briefly and then not insisted on.
        w16(b, CORBRP, 1 << 15);
        let _ = wait(1_000, || r16(b, CORBRP) & (1 << 15) != 0);
        w16(b, CORBRP, 0);
        let _ = wait(1_000, || r16(b, CORBRP) & (1 << 15) == 0);
        w16(b, CORBWP, 0);
        w16(b, RIRBWP, 1 << 15);
        // An answer counts as an interrupt-worthy batch of one. Zero is
        // reserved and some controllers stop writing responses on it.
        w16(b, RINTCNT, 1);
        self.rirb_rp = 0;
        // DMA on, and the response-interrupt *flag* enabled. Not for an
        // interrupt -- INTCTL is zero, so none reaches the processor -- but
        // because a controller counts responses toward RINTCNT and stops
        // fetching commands at the limit until that flag is raised and
        // cleared. With it disabled the flag never rises, the count never
        // resets, and the second command is never fetched: found by reading
        // the rings, CORB rp 1 wp 2 and RIRB wp 1, after one answer.
        w8(b, RIRBCTL, 3);
        w8(b, CORBCTL, 2);
        wait(10_000, || r8(b, CORBCTL) & 2 != 0 && r8(b, RIRBCTL) & 2 != 0)
    }

    /// Ask a codec one thing. `None` if it did not answer in time.
    pub fn ask(&mut self, addr: u8, nid: u16, verb: u32) -> Option<u32> {
        let cmd = verb::command(addr, nid as u8, verb);
        if self.immediate {
            return self.ask_immediate(cmd);
        }
        let b = self.bar;
        let n = self.entries;
        // **The response-interrupt flag has to be cleared by somebody**, and
        // with no interrupt handler that is here. Once responses reach
        // RINTCNT the controller raises it, and a controller is entitled to
        // stop taking commands until it is acknowledged -- QEMU's does. So
        // every command starts by acknowledging whatever is pending.
        w8(b, RIRBSTS, 0x05);
        let wp = ((r16(b, CORBWP) & 0xFF) + 1) % n;
        let at = wp as usize * 4;
        self.corb.as_mut_slice()[at..at + 4].copy_from_slice(&cmd.to_le_bytes());
        // The entry is in memory before the pointer says it is there.
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        w16(b, CORBWP, wp);
        let mut waited = 0u32;
        loop {
            let hw = r16(b, RIRBWP) & 0xFF;
            while self.rirb_rp != hw {
                self.rirb_rp = (self.rirb_rp + 1) % n;
                let at = self.rirb_rp as usize * 8;
                let e = &self.rirb.as_slice()[at..at + 8];
                let resp = u32::from_le_bytes([e[0], e[1], e[2], e[3]]);
                let ex = u32::from_le_bytes([e[4], e[5], e[6], e[7]]);
                // An unsolicited response -- a jack plugged in -- is not this
                // command's answer.
                if ex & 0x10 == 0 && (ex & 0xF) as u8 == addr {
                    w8(b, RIRBSTS, 0x05);
                    return Some(resp);
                }
            }
            if waited >= 50_000 {
                return None;
            }
            crate::time::delay_us(10);
            waited += 10;
        }
    }

    fn ask_immediate(&mut self, cmd: u32) -> Option<u32> {
        let b = self.bar;
        if !wait(1_000, || r16(b, ICIS) & 1 == 0) {
            return None;
        }
        // Clear a stale valid bit, write, and fire.
        w16(b, ICIS, r16(b, ICIS) | 2);
        w32(b, ICOI, cmd);
        w16(b, ICIS, r16(b, ICIS) | 1);
        if !wait(10_000, || r16(b, ICIS) & 3 == 2) {
            return None;
        }
        Some(r32(b, ICII))
    }

    /// The best output across every codec: (codec index, path).
    pub fn output(&self) -> Option<(usize, codec::Path)> {
        let mut best: Option<(u8, usize, codec::Path)> = None;
        for (i, c) in self.codecs.iter().enumerate() {
            if let Some(p) = codec::output_path(c) {
                let rank = c.widget(p.pin()).and_then(|w| w.pin).and_then(|d| d.rank()).unwrap_or(0);
                if best.as_ref().map_or(true, |b| rank > b.0) {
                    best = Some((rank, i, p));
                }
            }
        }
        best.map(|(_, i, p)| (i, p))
    }

    /// Power the path, unmute every amplifier on it, select every connection,
    /// enable the pin, and point the converter at `TAG` in `fmt`.
    fn route(&mut self, ci: usize, p: &codec::Path, fmt: u16) -> Result<(), Fault> {
        let addr = self.codecs[ci].addr;
        let afg = self.codecs[ci].afg;
        let mut say = |h: &mut Hda, nid: u16, v: u32| h.ask(addr, nid, v).ok_or(Fault::NoCommand);
        say(self, afg, verb::v12(verb::SET_POWER_STATE, 0))?;
        for (k, &nid) in p.nodes.iter().enumerate() {
            let w = self.codecs[ci].widget(nid).cloned();
            let Some(w) = w else { continue };
            if w.caps & verb::WCAP_POWER != 0 {
                say(self, nid, verb::v12(verb::SET_POWER_STATE, 0))?;
            }
            // Which input leads on: a selector or pin selects it, a mixer
            // unmutes it.
            if let Some(&pick) = p.picks.get(k) {
                match w.kind {
                    verb::Kind::Mixer => {
                        say(self, nid, verb::v4(verb::SET_AMP, verb::amp_in(pick, 0)))?;
                    }
                    _ if w.conns.len() > 1 => {
                        say(self, nid, verb::v12(verb::SET_CONNECTION_SELECT, pick))?;
                    }
                    _ => {}
                }
            }
            if w.caps & verb::WCAP_AMP_OUT != 0 {
                // To the amplifier's 0 dB step, which is its offset: a test
                // tone at full scale through maximum gain is a speaker
                // driven past what it is for.
                let zero_db = (w.amp_out & 0x7F) as u8;
                say(self, nid, verb::v4(verb::SET_AMP, verb::amp_out(zero_db)))?;
            }
            if w.kind == verb::Kind::Pin {
                let hp = w.pin_caps & verb::PCAP_HP != 0 && w.pin.is_some_and(|d| d.device == 2);
                let ctl = verb::PIN_OUT | if hp { verb::PIN_HP } else { 0 };
                say(self, nid, verb::v12(verb::SET_PIN_CONTROL, ctl))?;
                if w.pin_caps & verb::PCAP_EAPD != 0 {
                    say(self, nid, verb::v12(verb::SET_EAPD, verb::EAPD))?;
                }
            }
        }
        let dac = p.dac();
        say(self, dac, verb::v4(verb::SET_FORMAT, fmt))?;
        say(self, dac, verb::v12(verb::SET_STREAM_CHANNEL, TAG << 4))?;
        Ok(())
    }

    /// Play `pcm` (interleaved little-endian samples in `fmt`) on the best
    /// output, looping until `stop`.
    pub fn play(&mut self, pcm: &[u8], fmt: u16) -> Result<(), Fault> {
        self.stop();
        if self.outputs == 0 {
            return Err(Fault::NoStream);
        }
        let (ci, path) = self.output().ok_or(Fault::NoOutput)?;
        self.route(ci, &path, fmt)?;
        // The first output descriptor comes after every input one.
        let sd = SD_BASE + self.inputs as u64 * SD_SIZE;
        let b = self.bar;
        // Reset the descriptor: set, see it set, clear, see it clear.
        w32(b, sd + SD_CTL, r32(b, sd + SD_CTL) & 0x00FF_FFFF & !CTL_RUN);
        let _ = wait(10_000, || r32(b, sd + SD_CTL) & CTL_RUN == 0);
        w32(b, sd + SD_CTL, (r32(b, sd + SD_CTL) & 0x00FF_FFFF) | CTL_SRST);
        let _ = wait(10_000, || r32(b, sd + SD_CTL) & CTL_SRST != 0);
        w32(b, sd + SD_CTL, r32(b, sd + SD_CTL) & 0x00FF_FFFF & !CTL_SRST);
        let _ = wait(10_000, || r32(b, sd + SD_CTL) & CTL_SRST == 0);

        // The samples, in a buffer of their own, split into two halves of a
        // buffer descriptor list -- the controller wants at least two.
        let len = pcm.len() & !0x7F;
        if len < 256 {
            return Err(Fault::BadFormat);
        }
        let mut buf = Dma::new(len, 128).ok_or(Fault::NoMemory)?;
        buf.as_mut_slice().copy_from_slice(&pcm[..len]);
        let half = (len / 2) & !0x7F;
        let mut bdl = Dma::new(32, 128).ok_or(Fault::NoMemory)?;
        {
            let d = bdl.as_mut_slice();
            let parts = [(buf.pa(), half), (buf.pa() + half as u64, len - half)];
            for (i, (addr, n)) in parts.iter().enumerate() {
                d[i * 16..i * 16 + 8].copy_from_slice(&addr.to_le_bytes());
                d[i * 16 + 8..i * 16 + 12].copy_from_slice(&(*n as u32).to_le_bytes());
                d[i * 16 + 12..i * 16 + 16].copy_from_slice(&0u32.to_le_bytes());
            }
        }
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
        w32(b, sd + SD_CBL, len as u32);
        w16(b, sd + SD_LVI, 1);
        w16(b, sd + SD_FMT, fmt);
        w32(b, sd + SD_BDPL, bdl.pa() as u32);
        w32(b, sd + SD_BDPU, (bdl.pa() >> 32) as u32);
        // Clear stale status, then the tag and run, leaving the status byte
        // alone (its bits are write-one-to-clear).
        w8(b, sd + SD_STS, 0x1C);
        let ctl = (r32(b, sd + SD_CTL) & 0x000F_FFFF & !(0xF << 20)) | ((TAG as u32) << 20);
        w32(b, sd + SD_CTL, ctl);
        w32(b, sd + SD_CTL, ctl | CTL_RUN);
        let addr = self.codecs[ci].addr;
        self.stream = Some(Stream { sd, codec: addr, path, _bdl: bdl, _buf: buf });
        Ok(())
    }

    /// Where the stream is in its buffer, in bytes.
    pub fn position(&self) -> Option<u32> {
        self.stream.as_ref().map(|s| r32(self.bar, s.sd + SD_LPIB))
    }

    /// Stop the stream, waiting for it to say so before its memory goes.
    pub fn stop(&mut self) {
        if let Some(s) = self.stream.take() {
            let b = self.bar;
            w32(b, s.sd + SD_CTL, r32(b, s.sd + SD_CTL) & 0x00FF_FFFF & !CTL_RUN);
            let stopped = wait(10_000, || r32(b, s.sd + SD_CTL) & CTL_RUN == 0);
            // Mute the converter's stream, so a later stream on the tag does
            // not start into it.
            let _ = self.ask(s.codec, s.path.dac(), verb::v12(verb::SET_STREAM_CHANNEL, 0));
            if !stopped {
                // The engine would not confirm it stopped: its memory is kept
                // rather than handed back while it may still be read.
                core::mem::forget(s);
            }
        }
    }

}

impl Drop for Hda {
    /// Stop the stream and the rings and take bus mastering away, and only
    /// then hand the rings back.
    fn drop(&mut self) {
        self.stop();
        let b = self.bar;
        w8(b, CORBCTL, 0);
        w8(b, RIRBCTL, 0);
        let _ = wait(10_000, || r8(b, CORBCTL) & 2 == 0 && r8(b, RIRBCTL) & 2 == 0);
        if pci::disable_bus_master(self.ecam, &self.dev) {
            // Safety: each dropped once, here, and the device can no longer
            // reach them.
            unsafe {
                core::mem::ManuallyDrop::drop(&mut self.corb);
                core::mem::ManuallyDrop::drop(&mut self.rirb);
            }
        }
    }
}

/// The one controller up, if `hda up` brought one up.
static HELD: Spin<Option<HdaSlot>> = Spin::new(None);
struct HdaSlot(Hda);
// Moved between tasks only behind `HELD`; its raw pointers are its own DMA.
unsafe impl Send for HdaSlot {}

pub fn hold(h: Hda) {
    let mut g = HELD.lock();
    *g = None;
    *g = Some(HdaSlot(h));
}

pub fn release() -> bool {
    HELD.lock().take().is_some()
}

pub fn with<R>(f: impl FnOnce(&mut Hda) -> R) -> Option<R> {
    HELD.lock().as_mut().map(|s| f(&mut s.0))
}

/// A sine at `hz`, `ms` long, 48 kHz stereo 16-bit, at a quarter of full
/// scale. **One second exactly when looped**, so a whole number of cycles of
/// any whole-number frequency fits and the loop point does not click.
pub fn tone(hz: u32, amplitude: f32) -> Vec<u8> {
    let rate = 48_000u32;
    let mut out = Vec::with_capacity(rate as usize * 4);
    for i in 0..rate {
        let t = (i as u64 * hz as u64 % rate as u64) as f32 / rate as f32;
        let v = crate::ai::tensor::sinf(t * 2.0 * core::f32::consts::PI) * amplitude * 32767.0;
        let s = v as i16;
        out.extend_from_slice(&s.to_le_bytes());
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// One line per controller and codec, for the shell.
pub fn describe(h: &Hda) -> Vec<String> {
    let mut out = Vec::new();
    out.push(alloc::format!(
        "{:04x}:{:04x} HD Audio {}.{}, {} input and {} output stream(s), commands {}",
        h.dev.vendor, h.dev.device, h.version.0, h.version.1, h.inputs, h.outputs,
        if h.immediate { "immediate" } else { "through the rings" }
    ));
    let b = h.bar;
    out.push(alloc::format!(
        "  link {:04x}; CORB rp {:04x} wp {:04x} ctl {:02x} size {:02x}; RIRB wp {:04x} ctl {:02x} sts {:02x} size {:02x}; ICIS {:04x}",
        h.present, r16(b, CORBRP), r16(b, CORBWP), r8(b, CORBCTL), r8(b, CORBSIZE),
        r16(b, RIRBWP), r8(b, RIRBCTL), r8(b, RIRBSTS), r8(b, RIRBSIZE), r16(b, ICIS)
    ));
    if h.codecs.is_empty() {
        out.push(String::from("  codecs on the link, and none answered a command"));
    }
    for c in &h.codecs {
        let path = codec::output_path(c);
        out.push(alloc::format!(
            "  codec {} {:08x}, {} widget(s){}",
            c.addr,
            c.vendor,
            c.widgets.len(),
            match &path {
                Some(p) => {
                    let what = c.widget(p.pin()).and_then(|w| w.pin).map(|d| d.name()).unwrap_or("?");
                    alloc::format!(", plays to its {} (pin {:#04x}) from converter {:#04x}", what, p.pin(), p.dac())
                }
                None => String::from(", and no output a speaker or jack is wired to"),
            }
        ));
        for w in c.widgets.iter().filter(|w| w.kind == verb::Kind::Pin || w.kind == verb::Kind::Output) {
            let what = match (w.kind, w.pin) {
                (verb::Kind::Pin, Some(d)) => alloc::format!(
                    "pin, {} ({}){}",
                    d.name(),
                    if d.connected() { "wired" } else { "not wired" },
                    if w.pin_caps & verb::PCAP_OUT != 0 { ", can output" } else { "" }
                ),
                (verb::Kind::Output, _) => String::from(if w.digital() { "converter, digital" } else { "converter" }),
                _ => String::from("pin"),
            };
            out.push(alloc::format!("    {:#04x} caps {:08x} pcaps {:08x} conns {:x?}  {}", w.nid, w.caps, w.pin_caps, w.conns, what));
        }
    }
    out
}

pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out = verb::checks();
    out.extend(codec::checks());
    let t = tone(1000, 0.25);
    let sample = |i: usize| i16::from_le_bytes([t[i * 4], t[i * 4 + 1]]);
    out.push((
        "a test tone is one second of 48 kHz stereo, and loops on a whole cycle",
        t.len() == 48_000 * 4 && sample(0) == 0 && sample(48) == 0,
    ));
    out.push((
        "at a quarter of full scale, so a speaker is not driven at full volume",
        (0..48).map(sample).max().is_some_and(|m| m > 8000 && m < 8300),
    ));
    out
}
