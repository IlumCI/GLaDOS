//! The object space: which id names what, and who is allowed to pick it.
//!
//! A Wayland connection is a conversation about objects. Every message is
//! addressed to one, and almost every request makes another: a client asks the
//! registry for a compositor, the compositor for a surface, the surface for a
//! frame callback. There is no handshake and no reply carrying an id back,
//! because **the side making the object picks the number itself**, sends it as
//! a `new_id` argument, and starts using it in the next message.
//!
//! That works because the two sides allocate from ranges that cannot meet. A
//! client picks from 1 upward, a server from `0xFF000000` upward, and neither
//! ever has to ask. Object 1 is always `wl_display` and exists before either
//! side says anything, which is what gives a client somewhere to send its
//! first request.
//!
//! ### Indexed by id, which is why there is a ceiling
//!
//! A lookup happens on every single message, so the table is a `Vec` indexed
//! by the id rather than a list to be searched. Ids stay dense in practice --
//! libwayland recycles a destroyed id before growing -- so the table stays
//! about as large as the client's peak number of live objects, which for a
//! real application is hundreds.
//!
//! The catch is that a client picks the index. A first request naming id
//! `0x00FFFFFF` would ask for sixteen million slots, so there is a ceiling,
//! and an id above it is refused the same way an id in the server's range is.
//! A client that genuinely needed more than sixty-five thousand live objects
//! would be doing something no toolkit does.

use alloc::vec::Vec;

use super::wire::Error;

/// Object one, which is always `wl_display` and is never created.
pub const DISPLAY: u32 = 1;

/// The first id a server may allocate.
pub const SERVER_BASE: u32 = 0xFF00_0000;

/// The highest id a client may pick.
///
/// A ceiling rather than the protocol's `0xFEFFFFFF`, because the table is
/// indexed by the id and the client chooses it. See the note above.
pub const MAX_CLIENT_ID: u32 = 0x0001_0000;

/// How many ids the server may have out at once, for the same reason.
pub const MAX_SERVER_IDS: u32 = 0x0001_0000;

/// The interface an id speaks, and the version agreed for it.
///
/// The version is per object rather than per interface: a client binds
/// `wl_compositor` at version 4 and may hold another at version 1, and a
/// request that only exists from version 3 has to be answered against the
/// object it arrived on.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Entry {
    pub iface: &'static str,
    pub version: u32,
}

/// Everything alive on one connection.
///
/// Per connection rather than global, because ids are only meaningful inside
/// the conversation that made them. Two clients both using id 2 for their own
/// surface is ordinary and correct.
pub struct Table {
    /// Client-allocated, indexed by id. Slot 0 is never used.
    client: Vec<Option<Entry>>,
    /// Server-allocated, indexed by `id - SERVER_BASE`.
    server: Vec<Option<Entry>>,
}

impl Default for Table {
    fn default() -> Table {
        Table::new()
    }
}

impl Table {
    /// A new connection, with `wl_display` already in it.
    pub fn new() -> Table {
        let mut client = Vec::new();
        client.push(None); // id 0, which is the null object and never exists
        client.push(Some(Entry { iface: "wl_display", version: 1 }));
        Table { client, server: Vec::new() }
    }

    /// What an id names, or nothing.
    ///
    /// Id 0 is the null object and always answers nothing, which is what makes
    /// a nullable `object` argument readable without a special case at every
    /// call site.
    pub fn get(&self, id: u32) -> Option<Entry> {
        if id >= SERVER_BASE {
            let i = (id - SERVER_BASE) as usize;
            return self.server.get(i).copied().flatten();
        }
        self.client.get(id as usize).copied().flatten()
    }

    /// Bring a client-chosen id to life.
    pub fn create(&mut self, id: u32, iface: &'static str, version: u32) -> Result<(), Error> {
        if id == 0 || id > MAX_CLIENT_ID {
            return Err(Error::IdRange);
        }
        if self.get(id).is_some() {
            return Err(Error::IdInUse);
        }
        let i = id as usize;
        while self.client.len() <= i {
            self.client.push(None);
        }
        self.client[i] = Some(Entry { iface, version });
        Ok(())
    }

    /// Take an id out of the server's range.
    ///
    /// A freed slot is reused before the range grows, which is what libwayland
    /// does on its side and keeps the table the size of what is actually live
    /// rather than the size of everything ever made. A compositor hands out an
    /// id per frame callback, so a range that only ever crept upward would
    /// reach the ceiling in a few hours of animation.
    pub fn alloc(&mut self, iface: &'static str, version: u32) -> Result<u32, Error> {
        let entry = Some(Entry { iface, version });
        for (i, slot) in self.server.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = entry;
                return Ok(SERVER_BASE + i as u32);
            }
        }
        if self.server.len() as u32 >= MAX_SERVER_IDS {
            return Err(Error::IdExhausted);
        }
        self.server.push(entry);
        Ok(SERVER_BASE + (self.server.len() - 1) as u32)
    }

    /// Retire an id, so it can be used again.
    ///
    /// `wl_display` is refused: the protocol has no request that destroys it,
    /// and a connection whose object 1 was gone would have nowhere to report
    /// the error.
    pub fn destroy(&mut self, id: u32) -> Result<(), Error> {
        if id == DISPLAY {
            return Err(Error::IdRange);
        }
        if self.get(id).is_none() {
            return Err(Error::NoObject);
        }
        if id >= SERVER_BASE {
            self.server[(id - SERVER_BASE) as usize] = None;
        } else {
            self.client[id as usize] = None;
        }
        Ok(())
    }

    /// How many objects are alive.
    pub fn live(&self) -> usize {
        self.client.iter().flatten().count() + self.server.iter().flatten().count()
    }
}

/// What `diag sky` asks of the object space.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut t = Table::new();

    out.push((
        "a new connection already knows object one, because a client never creates wl_display",
        matches!(t.get(DISPLAY), Some(e) if e.iface == "wl_display"),
    ));
    out.push(("and nothing else is alive on it yet", t.live() == 1));
    out.push((
        "id zero names nothing, which is how a null object argument reads",
        t.get(0).is_none(),
    ));

    out.push((
        "a client may create an id of its own",
        t.create(2, "wl_registry", 1).is_ok(),
    ));
    out.push((
        "and it comes back carrying the interface it was made for",
        matches!(t.get(2), Some(e) if e.iface == "wl_registry" && e.version == 1),
    ));
    out.push((
        "an id already alive is refused, since two objects under one name is the whole conversation lost",
        t.create(2, "wl_surface", 1) == Err(Error::IdInUse),
    ));
    out.push((
        "so is id zero, which is reserved for the null object",
        t.create(0, "wl_surface", 1) == Err(Error::IdRange),
    ));
    out.push((
        "a client id inside the server's range is refused, because that is not the client's to pick",
        t.create(SERVER_BASE, "wl_surface", 1) == Err(Error::IdRange),
    ));
    out.push((
        "and an id above the ceiling is refused, since the table is indexed by it and the client chooses it",
        t.create(MAX_CLIENT_ID + 1, "wl_surface", 1) == Err(Error::IdRange),
    ));
    out.push((
        "an id at the ceiling is still allowed, so the boundary is where it says it is",
        t.create(MAX_CLIENT_ID, "wl_surface", 1).is_ok(),
    ));
    t.destroy(MAX_CLIENT_ID).ok();

    out.push(("destroying an id reports it gone", t.destroy(2).is_ok()));
    out.push(("after which nothing answers to it", t.get(2).is_none()));
    out.push((
        "and the number can be used again, which is what a client does rather than counting upward forever",
        t.create(2, "wl_surface", 3).is_ok(),
    ));
    out.push((
        "destroying an id nobody made is refused rather than quietly ignored",
        t.destroy(99) == Err(Error::NoObject),
    ));
    out.push((
        "wl_display cannot be destroyed, since the connection would have nowhere left to report an error",
        t.destroy(DISPLAY) == Err(Error::IdRange),
    ));

    // ---- the server's own range ----
    let a = t.alloc("wl_callback", 1);
    let b = t.alloc("wl_callback", 1);
    out.push((
        "the server allocates out of its own range, so the two sides can never collide without asking each other",
        a == Ok(SERVER_BASE) && matches!(b, Ok(id) if id >= SERVER_BASE),
    ));
    out.push(("and two allocations are two different ids", a != b));
    out.push((
        "a server id and the client id it differs from only by the base are separate objects",
        matches!(t.get(SERVER_BASE), Some(e) if e.iface == "wl_callback")
            && matches!(t.get(2), Some(e) if e.iface == "wl_surface"),
    ));
    let before = t.live();
    if let Ok(id) = a {
        t.destroy(id).ok();
    }
    out.push(("destroying a server id drops the count too", t.live() == before - 1));
    out.push((
        "and the freed slot is handed out again rather than the range creeping upward, which a callback per frame would otherwise exhaust",
        t.alloc("wl_callback", 1) == Ok(SERVER_BASE),
    ));

    out
}
