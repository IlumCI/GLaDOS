//! What the machine can say about where it stands.
//!
//! Whether the system should act on itself or only advise about itself is
//! often framed as a policy choice. It is really an epistemics question: the
//! answer follows from how well the system understands its surroundings, and
//! this module is that understanding made explicit and printable. Hardware
//! posture, load regime, operator presence, and the state of the network,
//! the store and any running episode -- each field carries whether it is
//! actually known, because a downstream consumer (the planner in
//! [`super::aixi`] above all) has to decide what it owes an unknown. An
//! honest unknown outranks a confident guess.

use super::{agent, futures};
use alloc::vec::Vec;

/// How hard the machine is working, relative to what it has itself been
/// seen to do. Absolute thresholds would bake one laptop's habits in as
/// physics; relative ones travel.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Load {
    Unknown,
    Quiet,
    Steady,
    Busy,
}

impl Load {
    pub fn label(self) -> &'static str {
        match self {
            Load::Unknown => "unknown",
            Load::Quiet => "quiet",
            Load::Steady => "steady",
            Load::Busy => "busy",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    /// No touches in the recent window. Under serial-only control this is
    /// the normal state: scripted keystrokes bypass the hardware ISR, so
    /// the entropy ring -- and this reading -- stay dark by design.
    Silent,
    Idle,
    Active,
}

impl Operator {
    pub fn label(self) -> &'static str {
        match self {
            Operator::Silent => "silent",
            Operator::Idle => "idle",
            Operator::Active => "active",
        }
    }
}

pub struct Situation {
    pub uptime_s: f32,
    pub heap_used_mib: f32,
    pub heap_total_mib: f32,
    pub tasks: u32,
    pub switch_rate: f32,
    pub touch_rate: f32,
    pub load: Load,
    /// Fraction of the last few samples that agree with `load`: how stable
    /// the classification is. A regime that flips every second is not a
    /// regime, and planning against it would be planning against noise.
    pub load_stability: f32,
    /// Signed MiB per second over the recent window. The planner reads this
    /// as drift; the operator reads it as "something is allocating".
    pub heap_trend_mib_s: f32,
    pub operator: Operator,
    pub episode_running: bool,
    pub store_mounted: bool,
    pub nets: Vec<(&'static str, bool)>,
}

impl Situation {
    pub fn free_mib(&self) -> f32 {
        (self.heap_total_mib - self.heap_used_mib).max(0.0)
    }
}

const TREND_WINDOW: usize = 8;
pub(crate) const STABILITY_WINDOW: usize = 8;

/// Read everything the kernel will admit to right now. Cheap enough to call
/// from a shell command or the clock task; every field is either measured or
/// marked unknown, never inferred twice.
pub fn gather() -> Situation {
    let now = futures::snapshot_now();
    let hist = futures::history();
    let (used, total) = crate::mem::heap::HEAP.stats();

    // Regime from the switch-rate history: where the current rate sits
    // relative to what the machine has recently done.
    let rates: Vec<f32> = hist.iter().map(|s| s.vars[1]).collect();
    let mean = if rates.is_empty() {
        0.0
    } else {
        rates.iter().sum::<f32>() / rates.len() as f32
    };
    let cur = now.vars[1];
    let load = if hist.len() < 8 || mean <= 0.0 {
        Load::Unknown
    } else if cur > 1.5 * mean {
        Load::Busy
    } else if cur < 0.35 * mean {
        Load::Quiet
    } else {
        Load::Steady
    };
    let label_of = |r: f32, mean: f32| -> Load {
        if mean <= 0.0 {
            Load::Unknown
        } else if r > 1.5 * mean {
            Load::Busy
        } else if r < 0.35 * mean {
            Load::Quiet
        } else {
            Load::Steady
        }
    };
    let tail = rates.len().saturating_sub(STABILITY_WINDOW);
    let agreeing = rates[tail..]
        .iter()
        .filter(|&&r| label_of(r, mean) == load)
        .count();
    let stability = if rates[tail..].is_empty() {
        0.0
    } else {
        agreeing as f32 / (rates.len() - tail) as f32
    };

    // Heap drift over the trend window, in MiB/s. Endpoints rather than a
    // fit: the ring already smooths, and the slope only has to be honest
    // about direction and rough size.
    let k = TREND_WINDOW.min(hist.len().saturating_sub(1));
    let trend = if k == 0 {
        0.0
    } else {
        let a = hist[hist.len() - 1 - k].vars[0];
        let b = hist[hist.len() - 1].vars[0];
        (b - a) / k as f32 / 1024.0
    };

    let touch = now.vars[2];
    let operator = if touch < 0.5 {
        Operator::Silent
    } else if touch < 4.0 {
        Operator::Idle
    } else {
        Operator::Active
    };

    let mut nets = Vec::new();
    for (i, iface) in crate::net::ifaces().iter().enumerate() {
        let _ = i;
        nets.push((iface.name, iface.up));
    }

    Situation {
        uptime_s: now.t_s,
        heap_used_mib: used as f32 / (1024.0 * 1024.0),
        heap_total_mib: total as f32 / (1024.0 * 1024.0),
        tasks: crate::task::count() as u32,
        switch_rate: now.vars[1],
        touch_rate: touch,
        load,
        load_stability: stability,
        heap_trend_mib_s: trend,
        operator,
        episode_running: agent::busy(),
        store_mounted: crate::store::mounted(),
        nets,
    }
}
