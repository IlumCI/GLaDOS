//! Time, from whatever the pool is running on.
//!
//! **This exists so the pool's rules can run somewhere that is not a process.**
//! `vardiff` measured its windows with `std::time::Instant` and `pool` stamped a
//! local header with `SystemTime::now()`, and both of those *compile* for
//! `wasm32-unknown-unknown` and panic the first time they run: there is no clock
//! on that target, only whatever the host hands in. The same pool core serving
//! miners from a Cloudflare Durable Object needs a clock it can be given.
//!
//! Two questions and no more, because they are the only two anything here asks:
//! how many milliseconds have passed (monotonic, for windows and rates), and what
//! the Unix time is (for a header's timestamp). A richer interface would be one
//! more thing the two targets could disagree about.
//!
//! **A Worker's clock does not move inside one event.** Cloudflare freezes
//! `Date.now()` for the length of a request as a Spectre mitigation and advances
//! it only across I/O. Every window measured here spans many messages, so a frozen
//! instant inside one message is exactly what the code already assumes: nothing
//! in it compares two readings taken while handling a single line.

#[cfg(not(target_arch = "wasm32"))]
mod imp {
    use std::sync::OnceLock;
    use std::time::{Instant, SystemTime, UNIX_EPOCH};

    static START: OnceLock<Instant> = OnceLock::new();

    pub fn now_ms() -> u64 {
        START.get_or_init(Instant::now).elapsed().as_millis() as u64
    }

    pub fn now_us() -> u64 {
        START.get_or_init(Instant::now).elapsed().as_micros() as u64
    }

    pub fn unix_secs() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
    }
}

#[cfg(target_arch = "wasm32")]
mod imp {
    // Supplied by the JavaScript that instantiates the module. Named after the
    // project rather than after what they return, so a stray `env` import from
    // some dependency cannot collide with them.
    //
    // **The module name is required, not tidy.** Without it `rust-lld` treats
    // an `extern` function as a symbol some other object should have defined,
    // and fails the link with `undefined symbol: glados_now_ms` -- it does not
    // assume a host import on its own.
    #[link(wasm_import_module = "glados")]
    extern "C" {
        fn glados_now_ms() -> f64;
    }

    pub fn now_ms() -> u64 {
        unsafe { glados_now_ms() as u64 }
    }

    pub fn now_us() -> u64 {
        now_ms() * 1000
    }

    pub fn unix_secs() -> u64 {
        now_ms() / 1000
    }
}

/// Milliseconds since some fixed moment, never going backwards within a run.
pub fn now_ms() -> u64 {
    imp::now_ms()
}

/// Microseconds on the same scale as `now_ms`, for timing one validation.
///
/// On a Worker this is milliseconds times a thousand, and frozen within one
/// event besides, so a validation timed there reads as zero. Nothing depends on
/// it there: the only reader is the validation budget, and a Durable Object never
/// sets one -- Cloudflare bounds CPU per message itself.
pub fn now_us() -> u64 {
    imp::now_us()
}

/// Seconds since 1970, for anything that ends up inside a block header.
pub fn unix_secs() -> u64 {
    imp::unix_secs()
}
