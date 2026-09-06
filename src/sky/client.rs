//! `wl_display` and `wl_registry`: how a client finds out what is here.
//!
//! These two are the whole of a Wayland connection's bootstrap and they are
//! the only interfaces a client can reach without being told about them. A
//! client opens the socket, sends `wl_display.get_registry`, and the server
//! answers with one `global` event per interface it offers. Everything else --
//! a compositor, a shared-memory pool, a window shell -- arrives by name
//! through that one list.
//!
//! **A registry name is not an object id and the two are easy to confuse.** A
//! global has a `name`, which is a number the server made up to identify an
//! entry in that list; `bind` turns a name into an object of the client's own
//! choosing. Nothing may be sent to a name.
//!
//! ### The bind argument that is three values
//!
//! `wl_registry.bind` is the one request in the core protocol whose `new_id`
//! carries its own interface, because the registry hands out objects of every
//! kind and the signature cannot say which. So the wire holds `name`, then a
//! **string** and a **version**, and only then the id. A decoder reading a
//! name and an id would take the length of `"wl_compositor"` for the object
//! number and get 14, which is a perfectly plausible id and completely wrong.
//! There is a claim below reading those middle two back off the wire.
//!
//! ### What a failure is
//!
//! One thing, and then silence. Wayland has no per-request reply, so a server
//! that refuses says so with `wl_display.error` naming the object, a code and
//! a sentence, and then the connection is over -- there is no state in which
//! a client has been told one request failed and may carry on. That is why
//! every refusal in `wire::Error` carries both a `code` for the protocol and a
//! `reason` for whoever reads the log, and why they live in one enum: a client
//! sees one error event, so there is one list of things that can be wrong.

use alloc::vec::Vec;

use super::object::{Table, DISPLAY};
use super::wire::{self, Error, Message, Writer};

pub const WL_DISPLAY: &str = "wl_display";
pub const WL_REGISTRY: &str = "wl_registry";
pub const WL_CALLBACK: &str = "wl_callback";

// wl_display
const DISPLAY_SYNC: u16 = 0;
const DISPLAY_GET_REGISTRY: u16 = 1;
const EV_ERROR: u16 = 0;
const EV_DELETE_ID: u16 = 1;

// wl_registry
const REGISTRY_BIND: u16 = 0;
const EV_GLOBAL: u16 = 0;
const EV_GLOBAL_REMOVE: u16 = 1;

// wl_callback
const EV_DONE: u16 = 0;

/// One interface this server offers, under the number it offers it as.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Global {
    /// The registry name. Made up by the server, meaningless to anything else.
    pub name: u32,
    pub iface: &'static str,
    pub version: u32,
}

/// One connection, and everything alive inside it.
pub struct Client {
    objects: Table,
    globals: Vec<Global>,
    /// Events waiting to be written to the socket.
    out: Vec<u8>,
    next_name: u32,
    serial: u32,
    /// Whether an error has already been reported, after which there is
    /// nothing more to say on this connection.
    gone: bool,
}

impl Default for Client {
    fn default() -> Client {
        Client::new()
    }
}

impl Client {
    pub fn new() -> Client {
        Client {
            objects: Table::new(),
            globals: Vec::new(),
            out: Vec::new(),
            next_name: 1,
            serial: 0,
            gone: false,
        }
    }

    pub fn alive(&self) -> bool {
        !self.gone
    }

    pub fn objects(&self) -> &Table {
        &self.objects
    }

    pub fn globals(&self) -> &[Global] {
        &self.globals
    }

    /// How many bytes of events are waiting.
    pub fn pending(&self) -> usize {
        self.out.len()
    }

    /// Take what has accumulated, for the socket to write.
    pub fn take_out(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.out)
    }

    /// Queue one event.
    ///
    /// An event that will not encode ends the connection rather than being
    /// dropped. It is a fact the client can never now be told, so its view of
    /// the object space is wrong from this moment on, and a compositor
    /// carrying on from there is one arguing with a client about whether an
    /// object exists.
    fn post(&mut self, w: Writer) {
        match w.finish() {
            Ok(bytes) => self.out.extend_from_slice(&bytes),
            Err(_) => self.gone = true,
        }
    }

    /// Offer an interface. Every registry the client already holds hears it.
    pub fn publish(&mut self, iface: &'static str, version: u32) -> u32 {
        let name = self.next_name;
        self.next_name += 1;
        self.globals.push(Global { name, iface, version });
        for id in self.objects.ids_of(WL_REGISTRY) {
            let mut w = Writer::new(id, EV_GLOBAL);
            w.uint(name);
            w.string(iface);
            w.uint(version);
            self.post(w);
        }
        name
    }

    /// Withdraw one, telling every registry.
    ///
    /// Objects already bound from it stay alive and stop working, which is
    /// what the protocol says: the client is expected to destroy them, and
    /// until it does the server may still be sent requests for a device that
    /// has gone.
    pub fn retract(&mut self, name: u32) -> bool {
        let before = self.globals.len();
        self.globals.retain(|g| g.name != name);
        if self.globals.len() == before {
            return false;
        }
        for id in self.objects.ids_of(WL_REGISTRY) {
            let mut w = Writer::new(id, EV_GLOBAL_REMOVE);
            w.uint(name);
            self.post(w);
        }
        true
    }

    /// One request, with the refusal handed back rather than sent.
    ///
    /// The claims use this. `request` is what a connection actually calls.
    pub fn dispatch(&mut self, msg: &Message) -> Result<(), Error> {
        let Some(entry) = self.objects.get(msg.object) else {
            return Err(Error::NoObject);
        };
        match entry.iface {
            WL_DISPLAY => self.display(msg),
            WL_REGISTRY => self.registry(msg),
            // An object of an interface nothing answers for yet. Refusing is
            // the honest response and it is also the useful one: the trace
            // says which interface to write next, the way `-ENOSYS` does for
            // a syscall.
            _ => Err(Error::NoMethod),
        }
    }

    /// One request, with a refusal turned into the error event and the
    /// connection closed. Answers whether it may continue.
    pub fn request(&mut self, msg: &Message) -> bool {
        if self.gone {
            return false;
        }
        if let Err(e) = self.dispatch(msg) {
            self.fail(msg.object, e);
        }
        !self.gone
    }

    /// `wl_display.error`, and then there is nothing more to say.
    fn fail(&mut self, object: u32, e: Error) {
        let mut w = Writer::new(DISPLAY, EV_ERROR);
        w.uint(object);
        w.uint(e.code());
        w.string(reason(e));
        self.post(w);
        self.gone = true;
    }

    fn display(&mut self, msg: &Message) -> Result<(), Error> {
        let mut r = msg.reader();
        match msg.opcode {
            DISPLAY_SYNC => {
                let id = r.new_id()?;
                done(&r)?;
                self.objects.create(id, WL_CALLBACK, 1)?;
                self.serial += 1;
                let mut w = Writer::new(id, EV_DONE);
                w.uint(self.serial);
                self.post(w);
                // The callback is spent the instant it fires, and saying so is
                // not tidiness. A client's own map frees the number when it
                // sees `delete_id`; a server that merely forgot the object
                // would leave that map growing by one per round trip, which
                // for anything animating is one per frame forever.
                self.objects.destroy(id)?;
                let mut w = Writer::new(DISPLAY, EV_DELETE_ID);
                w.uint(id);
                self.post(w);
                Ok(())
            }
            DISPLAY_GET_REGISTRY => {
                let id = r.new_id()?;
                done(&r)?;
                self.objects.create(id, WL_REGISTRY, 1)?;
                // Collected before sending, because `post` takes the whole of
                // `self` and the list being read lives inside it.
                let known: Vec<Global> = self.globals.clone();
                for g in known {
                    let mut w = Writer::new(id, EV_GLOBAL);
                    w.uint(g.name);
                    w.string(g.iface);
                    w.uint(g.version);
                    self.post(w);
                }
                Ok(())
            }
            _ => Err(Error::NoMethod),
        }
    }

    fn registry(&mut self, msg: &Message) -> Result<(), Error> {
        let mut r = msg.reader();
        match msg.opcode {
            REGISTRY_BIND => {
                let name = r.uint()?;
                // The interface and the version sit between the name and the
                // id, because this `new_id` has no interface in its signature
                // and so carries its own. See the note at the top.
                let want = r.string()?.ok_or(Error::NilArgument)?;
                let version = r.uint()?;
                let id = r.new_id()?;
                done(&r)?;
                let Some(g) = self.globals.iter().find(|g| g.name == name).copied() else {
                    return Err(Error::NoGlobal);
                };
                if want != g.iface {
                    return Err(Error::WrongInterface);
                }
                // Zero is refused as well as too high. Version numbering
                // starts at one, so a zero is a client that never filled the
                // field in, and creating the object anyway gives it something
                // that answers no request it knows.
                if version == 0 || version > g.version {
                    return Err(Error::BadVersion);
                }
                self.objects.create(id, g.iface, version)?;
                Ok(())
            }
            _ => Err(Error::NoMethod),
        }
    }
}

/// Refuse a request carrying more than its signature declares.
///
/// A Wayland request never grows arguments -- a changed signature is a new
/// opcode -- so bytes left over are the two sides disagreeing about what the
/// interface is, and every argument read after that point is read at the wrong
/// offset. Stopping here makes that a refusal instead of a rendering bug an
/// hour later.
fn done(r: &wire::Reader<'_>) -> Result<(), Error> {
    if r.at_end() {
        Ok(())
    } else {
        Err(Error::Trailing)
    }
}

/// What to put in `wl_display.error` for each refusal.
///
/// The code tells a client's error handler which of four kinds it was; this is
/// what a person reads afterwards, and it is the only place the reason
/// survives. Written as sentences rather than enum names for that reason.
pub fn reason(e: Error) -> &'static str {
    match e {
        Error::SizeTooSmall => "a message shorter than its own header",
        Error::SizeUnaligned => "a message size that is not a multiple of four",
        Error::Truncated => "an argument running past the end of its message",
        Error::StringLength => "a string longer than the message holding it",
        Error::StringTerminator => "a string with no terminator where its length says",
        Error::StringUtf8 => "a string that is not UTF-8",
        Error::ArrayLength => "an array longer than the message holding it",
        Error::NoDescriptor => "a request wanting a descriptor that did not arrive with it",
        Error::TooLarge => "a message too large for its own size field",
        Error::IdRange => "an id outside the range a client may allocate from",
        Error::IdInUse => "a new id that names something already alive",
        Error::NoObject => "a request for an object that is not here",
        Error::IdExhausted => "no id left to allocate",
        Error::NoMethod => "a request this interface does not have",
        Error::Trailing => "a request longer than its signature",
        Error::NilArgument => "a null argument where the request has no null",
        Error::NoGlobal => "a bind naming a global that was never published",
        Error::WrongInterface => "a bind naming an interface the global does not speak",
        Error::BadVersion => "a bind above the version advertised",
    }
}

/// Build one request, the way a client would.
///
/// Here rather than in the claims because the checks below are about the
/// server and a request builder written inline in each of them would be six
/// copies of the same encoding, agreeing by hand.
fn req(object: u32, opcode: u16, build: impl FnOnce(&mut Writer)) -> Vec<u8> {
    let mut w = Writer::new(object, opcode);
    build(&mut w);
    w.finish().unwrap_or_default()
}

/// Split a run of events, so a claim can look at what was actually sent.
fn events(bytes: &[u8]) -> Vec<(u32, u16, Vec<u8>)> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Ok(Some(m)) = wire::frame(&bytes[at..]) {
        at += m.whole();
        out.push((m.object, m.opcode, Vec::from(m.body)));
    }
    out
}

/// Hand one request to a client, by the path a connection uses.
fn send(c: &mut Client, bytes: &[u8]) -> bool {
    match wire::frame(bytes) {
        Ok(Some(m)) => c.request(&m),
        _ => false,
    }
}

/// The same, reporting the refusal instead of sending it.
fn try_send(c: &mut Client, bytes: &[u8]) -> Result<(), Error> {
    match wire::frame(bytes) {
        Ok(Some(m)) => c.dispatch(&m),
        Ok(None) => Err(Error::Truncated),
        Err(e) => Err(e),
    }
}

/// What `diag sky` asks of the two bootstrap interfaces.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // ---- the registry, and what a client learns from it ----
    let mut c = Client::new();
    let shm = c.publish("wl_shm", 1);
    let comp = c.publish("wl_compositor", 4);
    out.push(("two globals get two names", shm != comp));
    out.push((
        "a global published before anybody is listening reaches nobody yet",
        c.pending() == 0,
    ));

    let ok = send(&mut c, &req(DISPLAY, DISPLAY_GET_REGISTRY, |w| {
        w.new_id(2);
    }));
    out.push(("get_registry is answered", ok));
    out.push((
        "and makes an object of the interface it names",
        matches!(c.objects().get(2), Some(e) if e.iface == WL_REGISTRY),
    ));
    let evs = events(&c.take_out());
    out.push((
        "a new registry is told about every global that already exists",
        evs.len() == 2 && evs.iter().all(|e| e.0 == 2 && e.1 == EV_GLOBAL),
    ));
    let mut r = wire::Reader::new(&evs[1].2);
    out.push((
        "and each event carries the name, the interface and the version, in that order",
        r.uint() == Ok(comp)
            && matches!(r.string(), Ok(Some(ref s)) if s == "wl_compositor")
            && r.uint() == Ok(4)
            && r.at_end(),
    ));

    let seat = c.publish("wl_seat", 7);
    let evs = events(&c.take_out());
    out.push((
        "a global published afterwards reaches the registry already open",
        evs.len() == 1 && evs[0].1 == EV_GLOBAL,
    ));
    send(&mut c, &req(DISPLAY, DISPLAY_GET_REGISTRY, |w| {
        w.new_id(3);
    }));
    c.take_out();
    let removed = c.retract(seat);
    let evs = events(&c.take_out());
    out.push((
        "retracting one tells every registry the client holds, and it holds two",
        removed && evs.len() == 2 && evs.iter().all(|e| e.1 == EV_GLOBAL_REMOVE),
    ));
    out.push(("retracting a name nobody published says so", !c.retract(seat)));

    // ---- bind ----
    let mut c = Client::new();
    let comp = c.publish("wl_compositor", 4);
    send(&mut c, &req(DISPLAY, DISPLAY_GET_REGISTRY, |w| {
        w.new_id(2);
    }));
    c.take_out();
    let bind = |name: u32, iface: &str, version: u32, id: u32| {
        req(2, REGISTRY_BIND, move |w| {
            w.uint(name);
            w.string(iface);
            w.uint(version);
            w.new_id(id);
        })
    };
    // The wire itself, before anything is asked of the server: name, then a
    // string and a version, and only then the id. Reading a name and an id
    // would take 14 -- the length of "wl_compositor" and its terminator -- for
    // the object number, which is a plausible id and entirely wrong.
    let wire_bytes = bind(comp, "wl_compositor", 4, 40);
    let mut r = wire::Reader::new(&wire_bytes[wire::HEADER..]);
    let (a, b, cc, d) = (r.uint(), r.string(), r.uint(), r.new_id());
    out.push((
        "on the wire, bind carries an interface and a version between the name and the id, because this new_id has no interface in its signature",
        a == Ok(comp)
            && matches!(b, Ok(Some(ref s)) if s == "wl_compositor")
            && cc == Ok(4)
            && d == Ok(40)
            && r.at_end(),
    ));

    out.push((
        "binding a published global is answered",
        send(&mut c, &bind(comp, "wl_compositor", 4, 40)),
    ));
    out.push((
        "and the object speaks the interface the global named, at the version asked for",
        matches!(c.objects().get(40), Some(e) if e.iface == "wl_compositor" && e.version == 4),
    ));
    // Where a server reading a name and then an id would have put it: 14 is
    // the length of "wl_compositor" with its terminator, which is a plausible
    // object number and the wrong one. This is the claim that watches the
    // server rather than the encoding -- the one above reads the bytes with
    // its own reader and would pass either way.
    out.push((
        "and nothing landed at the length of the interface name, which is where reading past the string would have put it",
        c.objects().get(14).is_none(),
    ));
    out.push((
        "binding below the advertised version is allowed, which is how an old client talks to a new server",
        try_send(&mut c, &bind(comp, "wl_compositor", 1, 41)).is_ok(),
    ));
    out.push((
        "binding above it is refused",
        try_send(&mut c, &bind(comp, "wl_compositor", 5, 42)) == Err(Error::BadVersion),
    ));
    out.push((
        "and at version zero, which is a field never filled in rather than a request for the earliest",
        try_send(&mut c, &bind(comp, "wl_compositor", 0, 43)) == Err(Error::BadVersion),
    ));
    out.push((
        "binding a name nobody published is refused",
        try_send(&mut c, &bind(comp + 99, "wl_compositor", 1, 44)) == Err(Error::NoGlobal),
    ));
    out.push((
        "binding with the wrong interface for that name is refused, since the two must agree",
        try_send(&mut c, &bind(comp, "wl_shm", 1, 45)) == Err(Error::WrongInterface),
    ));
    out.push((
        "and a nil interface is refused rather than matched against nothing",
        try_send(
            &mut c,
            &req(2, REGISTRY_BIND, |w| {
                w.uint(comp);
                w.nil_string();
                w.uint(1);
                w.new_id(46);
            }),
        ) == Err(Error::NilArgument),
    ));
    out.push((
        "a bind onto an id already alive is refused",
        try_send(&mut c, &bind(comp, "wl_compositor", 1, 40)) == Err(Error::IdInUse),
    ));
    out.push((
        "and an object bound from a global answers nothing yet, rather than accepting requests silently",
        try_send(&mut c, &req(40, 0, |_| {})) == Err(Error::NoMethod),
    ));

    // ---- sync ----
    let mut c = Client::new();
    out.push((
        "sync is answered",
        send(&mut c, &req(DISPLAY, DISPLAY_SYNC, |w| { w.new_id(2); })),
    ));
    let evs = events(&c.take_out());
    out.push((
        "with done on the callback, then delete_id on the display",
        evs.len() == 2
            && evs[0] == (2, EV_DONE, evs[0].2.clone())
            && evs[1].0 == DISPLAY
            && evs[1].1 == EV_DELETE_ID,
    ));
    out.push((
        "delete_id names the callback, so the client can free the number",
        wire::Reader::new(&evs[1].2).uint() == Ok(2),
    ));
    out.push((
        "the callback is gone from the server's side too",
        c.objects().get(2).is_none(),
    ));
    out.push((
        "so the same id can be used for the next one",
        send(&mut c, &req(DISPLAY, DISPLAY_SYNC, |w| { w.new_id(2); })),
    ));
    let evs = events(&c.take_out());
    let mut first = wire::Reader::new(&evs[0].2);
    out.push((
        "and the serial moves, so two round trips are told apart",
        first.uint() == Ok(2),
    ));

    // ---- refusals, and what a client is told ----
    let mut c = Client::new();
    out.push((
        "a request to an object nobody made is refused",
        try_send(&mut c, &req(7, 0, |_| {})) == Err(Error::NoObject),
    ));
    out.push((
        "a request to object zero is refused, that being the null object",
        try_send(&mut c, &req(0, 0, |_| {})) == Err(Error::NoObject),
    ));
    out.push((
        "an opcode wl_display does not have is refused",
        try_send(&mut c, &req(DISPLAY, 9, |_| {})) == Err(Error::NoMethod),
    ));
    out.push((
        "a request carrying more than its signature declares is refused",
        try_send(
            &mut c,
            &req(DISPLAY, DISPLAY_SYNC, |w| {
                w.new_id(2);
                w.uint(0);
            }),
        ) == Err(Error::Trailing),
    ));
    out.push((
        "sync onto an id already alive is refused",
        try_send(&mut c, &req(DISPLAY, DISPLAY_SYNC, |w| { w.new_id(DISPLAY); }))
            == Err(Error::IdInUse),
    ));

    let mut c = Client::new();
    let alive = send(&mut c, &req(7, 0, |_| {}));
    let evs = events(&c.take_out());
    let mut r = wire::Reader::new(&evs[0].2);
    out.push((
        "a refusal becomes one error event on the display, naming the object that failed",
        evs.len() == 1 && evs[0].0 == DISPLAY && evs[0].1 == EV_ERROR && r.uint() == Ok(7),
    ));
    out.push((
        "carrying the protocol's own code",
        r.uint() == Ok(wire::ERR_INVALID_OBJECT),
    ));
    out.push((
        "and a sentence saying which refusal it was",
        matches!(r.string(), Ok(Some(ref s)) if s == reason(Error::NoObject)) && r.at_end(),
    ));
    out.push((
        "after which the connection is over, because Wayland has no way to say one request failed and carry on",
        !alive && !c.alive(),
    ));
    let before = c.pending();
    out.push((
        "and a further request on it does nothing at all",
        !send(&mut c, &req(DISPLAY, DISPLAY_SYNC, |w| { w.new_id(2); })) && c.pending() == before,
    ));

    out
}
