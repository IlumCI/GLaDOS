//! The Wayland wire format: eight bytes of header, then arguments.
//!
//! A message is an object id, an opcode, a length, and a payload whose shape
//! comes from the interface that object belongs to. Everything is 32-bit
//! aligned and in host byte order, which here means little-endian on both
//! sides of the socket because both sides are this machine.
//!
//!     0        4        6        8
//!     +--------+--------+--------+-----------------------------+
//!     | object | opcode |  size  | arguments, padded to 4      |
//!     +--------+--------+--------+-----------------------------+
//!
//! The opcode and the size share one word, size in the high half. **The size
//! counts the header**, so the smallest legal message is eight bytes and a
//! request with no arguments is exactly that.
//!
//! ### The size field belongs to whoever sent it
//!
//! That is the whole security story of this file. A client picks the number,
//! and a reader that believes it can be told a message is four bytes long and
//! loop forever on a buffer that never advances, or told it is sixty thousand
//! and read whatever memory follows. So `frame` refuses a size below the
//! header and a size that is not a multiple of four, and it treats a size
//! larger than what has arrived as **not yet** rather than as an error, which
//! is the honest answer on a stream where the rest is still in flight.
//!
//! Every argument reader checks against the message's own end for the same
//! reason. A string carries its own length too, and that length is also the
//! client's.
//!
//! ### An `fd` argument occupies no bytes at all
//!
//! Descriptors travel in the ancillary data of the socket, so the argument is
//! present in the interface's signature and absent from the payload. A reader
//! has nothing to advance past, which is why `fd` takes the queue as a
//! parameter and this file names no descriptor type anywhere. It stays honest
//! about ordering: `linux::unix` hands over a batch of descriptors only once
//! the bytes they were attached behind have been read, so by the time a
//! message is framed here, its descriptors are already in the queue. An empty
//! queue at an `fd` argument is a real protocol failure and not a wait.

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec::Vec;

/// Object id, opcode and size.
pub const HEADER: usize = 8;

/// The largest message the size field can describe, header included.
pub const MAX_MESSAGE: usize = 0xFFFF;

/// What a message can be wrong about, named by the field that was wrong.
///
/// Wayland's own failure path is the `wl_display.error` event, which carries a
/// code from a very short list. Most of what can go wrong down here has no
/// good code on that list, because a stream that cannot be parsed cannot carry
/// a considered explanation either. `code` maps each of these onto the nearest
/// one so a server can say something before hanging up, and hanging up is the
/// expected response to anything structural.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Error {
    /// A size smaller than the header it sits in.
    SizeTooSmall,
    /// A size that is not a multiple of four, so the next header would start
    /// misaligned and every message after it would be garbage.
    SizeUnaligned,
    /// An argument ran past the end of its message.
    Truncated,
    /// A string's length ran past the end of its message.
    StringLength,
    /// A string's length did not land on the terminator it counts.
    StringTerminator,
    /// A string was not UTF-8.
    StringUtf8,
    /// An array's length ran past the end of its message.
    ArrayLength,
    /// An `fd` argument, and no descriptor had arrived with the bytes.
    NoDescriptor,
    /// A message that would not fit in the size field.
    TooLarge,
    /// An id outside the range its sender may allocate from.
    IdRange,
    /// A new id that names something already alive.
    IdInUse,
    /// A message for an object that is not there.
    NoObject,
    /// No id left in the range the server allocates from.
    IdExhausted,
    /// An opcode the interface does not have.
    NoMethod,
    /// Bytes left after the last argument the request declares.
    Trailing,
    /// A null where the request has no null to offer.
    NilArgument,
    /// A registry name nothing was ever published under.
    NoGlobal,
    /// A bind naming an interface the global does not speak.
    WrongInterface,
    /// A bind above the version advertised, or at nothing.
    BadVersion,
}

/// `wl_display.error` codes, from the core protocol.
pub const ERR_INVALID_OBJECT: u32 = 0;
pub const ERR_INVALID_METHOD: u32 = 1;
pub const ERR_NO_MEMORY: u32 = 2;
pub const ERR_IMPLEMENTATION: u32 = 3;

impl Error {
    /// The `wl_display.error` code to send before disconnecting.
    pub fn code(self) -> u32 {
        match self {
            Error::NoObject
            | Error::IdInUse
            | Error::IdRange
            | Error::NoGlobal
            | Error::WrongInterface
            | Error::BadVersion => ERR_INVALID_OBJECT,
            Error::NoMethod => ERR_INVALID_METHOD,
            Error::TooLarge | Error::IdExhausted => ERR_NO_MEMORY,
            _ => ERR_IMPLEMENTATION,
        }
    }
}

/// `wl_fixed_t`: a signed 24.8 fixed point number.
///
/// Wayland uses these for anything that can land between pixels, which is
/// pointer coordinates and surface offsets under a scaled output. The integer
/// part is 24 bits signed, so it holds a little over eight million, which is
/// several thousand screens wide.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Fixed(pub i32);

impl Fixed {
    /// A whole number of units.
    ///
    /// A shift rather than a multiply because the multiply overflows for
    /// anything outside the 24-bit range and this build has overflow checks
    /// on. Discarding the high bits is what C does here and what the format
    /// can represent.
    pub const fn from_int(v: i32) -> Fixed {
        Fixed(v << 8)
    }

    /// The whole part, **truncated toward zero**.
    ///
    /// `wl_fixed_to_int` is a division by 256 in C, so -1.5 comes back as -1.
    /// An arithmetic shift would floor it to -2 instead, and the two agree for
    /// every positive value -- which is exactly what makes the difference easy
    /// to introduce and hard to notice, since a pointer only crosses zero when
    /// it leaves the screen. There is a claim below holding the two apart.
    pub const fn to_int(self) -> i32 {
        self.0 / 256
    }

    /// The raw 24.8 value, which is what goes on the wire.
    pub const fn bits(self) -> i32 {
        self.0
    }
}

/// One whole message, borrowed out of the buffer it arrived in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Message<'a> {
    pub object: u32,
    pub opcode: u16,
    /// The arguments, without the header.
    pub body: &'a [u8],
}

impl Message<'_> {
    /// How many bytes to drop from the front of the buffer to move past this.
    pub fn whole(&self) -> usize {
        HEADER + self.body.len()
    }

    /// A reader over the arguments.
    pub fn reader(&self) -> Reader<'_> {
        Reader::new(self.body)
    }
}

/// Take one message off the front of a buffer.
///
/// `Ok(None)` means the rest has not arrived, which on a byte stream is
/// ordinary and happens constantly. `Err` means the stream is malformed and
/// there is no recovering the framing, because the field that says where the
/// next message starts is the field that is wrong.
pub fn frame(buf: &[u8]) -> Result<Option<Message<'_>>, Error> {
    if buf.len() < HEADER {
        return Ok(None);
    }
    let object = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let second = u32::from_le_bytes([buf[4], buf[5], buf[6], buf[7]]);
    let opcode = (second & 0xFFFF) as u16;
    let size = (second >> 16) as usize;
    if size < HEADER {
        return Err(Error::SizeTooSmall);
    }
    if size % 4 != 0 {
        return Err(Error::SizeUnaligned);
    }
    if buf.len() < size {
        return Ok(None);
    }
    Ok(Some(Message { object, opcode, body: &buf[HEADER..size] }))
}

/// How long `n` bytes become once padded up to a 4-byte boundary.
const fn padded(n: usize) -> usize {
    (n + 3) & !3
}

/// Arguments coming off a message.
pub struct Reader<'a> {
    body: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    pub fn new(body: &'a [u8]) -> Reader<'a> {
        Reader { body, at: 0 }
    }

    /// Whether every argument has been taken.
    ///
    /// Worth asking after decoding a request: bytes left over mean the sender
    /// and this server disagree about the interface, and continuing on that
    /// disagreement is how a protocol bug becomes a rendering bug an hour
    /// later.
    pub fn at_end(&self) -> bool {
        self.at >= self.body.len()
    }

    /// How many bytes have been consumed.
    pub fn at(&self) -> usize {
        self.at
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self.at.checked_add(n).ok_or(Error::Truncated)?;
        if end > self.body.len() {
            return Err(Error::Truncated);
        }
        let out = &self.body[self.at..end];
        self.at = end;
        Ok(out)
    }

    pub fn uint(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    pub fn int(&mut self) -> Result<i32, Error> {
        Ok(self.uint()? as i32)
    }

    pub fn fixed(&mut self) -> Result<Fixed, Error> {
        Ok(Fixed(self.uint()? as i32))
    }

    /// An object id, where zero means null.
    pub fn object(&mut self) -> Result<u32, Error> {
        self.uint()
    }

    /// An id the sender is creating.
    pub fn new_id(&mut self) -> Result<u32, Error> {
        self.uint()
    }

    /// A string, or `None` for the nil string.
    ///
    /// The length counts the terminating NUL, so an empty string is one and
    /// nil is zero. Those are genuinely different values and a few requests
    /// care about the difference, so they stay distinguishable here rather
    /// than both arriving as an empty `String`.
    ///
    /// Refused rather than repaired when it is not UTF-8: interface names get
    /// compared against `&'static str`, and a lossy conversion would let two
    /// different names compare equal.
    pub fn string(&mut self) -> Result<Option<String>, Error> {
        let len = self.uint()? as usize;
        if len == 0 {
            return Ok(None);
        }
        let b = self.take(padded(len)).map_err(|_| Error::StringLength)?;
        if b[len - 1] != 0 {
            return Err(Error::StringTerminator);
        }
        let text = core::str::from_utf8(&b[..len - 1]).map_err(|_| Error::StringUtf8)?;
        Ok(Some(String::from(text)))
    }

    /// A byte array, whose length does not count its padding.
    pub fn array(&mut self) -> Result<Vec<u8>, Error> {
        let len = self.uint()? as usize;
        let b = self.take(padded(len)).map_err(|_| Error::ArrayLength)?;
        Ok(Vec::from(&b[..len]))
    }

    /// A descriptor, out of the batch that arrived with these bytes.
    ///
    /// Takes the queue rather than holding it, because this argument occupies
    /// no space in the message and so the reader has nothing to advance. The
    /// shape keeps this file free of any descriptor type at all, which is why
    /// the claims below can exercise it with plain integers.
    pub fn fd<T>(&mut self, queue: &mut VecDeque<T>) -> Result<T, Error> {
        queue.pop_front().ok_or(Error::NoDescriptor)
    }
}

/// Arguments going onto a message.
///
/// The size cannot be known until the arguments are written, so it is left at
/// zero and patched by `finish`. A builder that asked for the size up front
/// would be asking the caller to count bytes by hand, which is a promise
/// somebody eventually gets wrong in a way nothing checks.
///
/// A refusal is held until `finish` instead of being returned from each
/// argument. That keeps a run of writes readable, and there is exactly one
/// place a message can fail to be produced.
pub struct Writer {
    buf: Vec<u8>,
    bad: Option<Error>,
}

impl Writer {
    pub fn new(object: u32, opcode: u16) -> Writer {
        let mut buf = Vec::new();
        buf.extend_from_slice(&object.to_le_bytes());
        // Size stays zero until `finish` knows it.
        buf.extend_from_slice(&(opcode as u32).to_le_bytes());
        Writer { buf, bad: None }
    }

    pub fn uint(&mut self, v: u32) -> &mut Self {
        self.buf.extend_from_slice(&v.to_le_bytes());
        self
    }

    pub fn int(&mut self, v: i32) -> &mut Self {
        self.uint(v as u32)
    }

    pub fn fixed(&mut self, v: Fixed) -> &mut Self {
        self.uint(v.bits() as u32)
    }

    pub fn object(&mut self, id: u32) -> &mut Self {
        self.uint(id)
    }

    pub fn new_id(&mut self, id: u32) -> &mut Self {
        self.uint(id)
    }

    /// A string, terminator and padding included.
    ///
    /// An interior NUL is refused. The length field would say one thing and
    /// the terminator sit somewhere else, and the far end would read a
    /// truncated name that looks perfectly valid.
    pub fn string(&mut self, text: &str) -> &mut Self {
        if text.as_bytes().contains(&0) {
            self.bad.get_or_insert(Error::StringTerminator);
            return self;
        }
        let len = text.len() + 1;
        self.uint(len as u32);
        self.buf.extend_from_slice(text.as_bytes());
        self.buf.push(0);
        while self.buf.len() % 4 != 0 {
            self.buf.push(0);
        }
        self
    }

    /// The nil string, which is a length of zero and nothing after it.
    pub fn nil_string(&mut self) -> &mut Self {
        self.uint(0)
    }

    pub fn array(&mut self, bytes: &[u8]) -> &mut Self {
        self.uint(bytes.len() as u32);
        self.buf.extend_from_slice(bytes);
        while self.buf.len() % 4 != 0 {
            self.buf.push(0);
        }
        self
    }

    /// Patch the size in and hand over the bytes.
    pub fn finish(mut self) -> Result<Vec<u8>, Error> {
        if let Some(e) = self.bad {
            return Err(e);
        }
        let size = self.buf.len();
        if size > MAX_MESSAGE {
            return Err(Error::TooLarge);
        }
        let second = u32::from_le_bytes([self.buf[4], self.buf[5], self.buf[6], self.buf[7]]);
        let patched = (second & 0xFFFF) | ((size as u32) << 16);
        self.buf[4..8].copy_from_slice(&patched.to_le_bytes());
        Ok(self.buf)
    }
}

/// What `diag sky` asks of the wire format.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();

    // ---- framing ----
    let empty: [u8; 0] = [];
    out.push((
        "a buffer shorter than a header is not yet a message, and not an error",
        matches!(frame(&empty), Ok(None)) && matches!(frame(&[1, 0, 0]), Ok(None)),
    ));

    let bare = Writer::new(7, 3).finish().unwrap_or_default();
    out.push(("a message with no arguments is exactly the header", bare.len() == HEADER));
    let framed = frame(&bare);
    out.push((
        "the object and opcode come back off the wire",
        matches!(&framed, Ok(Some(m)) if m.object == 7 && m.opcode == 3),
    ));
    out.push((
        "and the body of an empty message is empty",
        matches!(&framed, Ok(Some(m)) if m.body.is_empty()),
    ));

    // A header that says more is coming is a wait, not a refusal.
    let mut clipped = bare.clone();
    clipped[6] = 16; // size 16, only 8 bytes present
    out.push(("a header whose body has not arrived yet is a wait", matches!(frame(&clipped), Ok(None))));

    let mut small = bare.clone();
    small[6] = 4;
    out.push((
        "a size smaller than its own header is refused",
        frame(&small) == Err(Error::SizeTooSmall),
    ));
    let mut odd = bare.clone();
    odd[6] = 9;
    out.push((
        "a size that is not a multiple of four is refused, since the next header would be misaligned",
        frame(&odd) == Err(Error::SizeUnaligned),
    ));

    let mut two = Writer::new(1, 0).finish().unwrap_or_default();
    let second = {
        let mut w = Writer::new(2, 1);
        w.uint(0xAABB_CCDD);
        w.finish().unwrap_or_default()
    };
    two.extend_from_slice(&second);
    let first_len = match frame(&two) {
        Ok(Some(m)) => m.whole(),
        _ => 0,
    };
    out.push(("a message reports its whole length, header included", first_len == HEADER));
    out.push((
        "so the message behind it frames from where the first ended",
        matches!(frame(&two[first_len..]), Ok(Some(m)) if m.object == 2 && m.opcode == 1),
    ));

    // ---- fixed point ----
    out.push(("one whole unit of fixed point is 256", Fixed::from_int(1).bits() == 256));
    out.push(("and it comes back a whole unit", Fixed::from_int(1).to_int() == 1));
    let minus_one_and_a_half = Fixed(-384);
    out.push((
        "fixed point truncates toward zero, the way libwayland's division does",
        minus_one_and_a_half.to_int() == -1,
    ));
    out.push((
        "which is a different answer from a shift, and that is why it is a division",
        (minus_one_and_a_half.bits() >> 8) == -2,
    ));

    // ---- every argument type, one message ----
    let mut w = Writer::new(0x0100, 5);
    w.uint(0xDEAD_BEEF);
    w.int(-12345);
    w.fixed(Fixed::from_int(-3));
    w.object(0);
    w.new_id(42);
    w.string("wl_compositor");
    w.nil_string();
    w.string("");
    w.array(&[1, 2, 3, 4, 5]);
    let msg = w.finish().unwrap_or_default();
    out.push(("a full message is a multiple of four bytes long", msg.len() % 4 == 0));
    out.push((
        "the size field carries the whole length once finish patches it",
        u32::from_le_bytes([msg[4], msg[5], msg[6], msg[7]]) >> 16 == msg.len() as u32,
    ));

    let mut round = false;
    let mut consumed = false;
    if let Ok(Some(m)) = frame(&msg) {
        let mut r = m.reader();
        round = r.uint() == Ok(0xDEAD_BEEF)
            && r.int() == Ok(-12345)
            && r.fixed() == Ok(Fixed::from_int(-3))
            && r.object() == Ok(0)
            && r.new_id() == Ok(42)
            && matches!(r.string(), Ok(Some(ref s)) if s == "wl_compositor")
            && matches!(r.string(), Ok(None))
            && matches!(r.string(), Ok(Some(ref s)) if s.is_empty())
            && matches!(r.array(), Ok(ref a) if a == &[1, 2, 3, 4, 5]);
        consumed = r.at_end();
    }
    out.push(("every argument type round-trips through one message", round));
    out.push((
        "and the reader ends exactly at the end, so nothing was left over",
        consumed,
    ));

    // ---- strings ----
    let mut w = Writer::new(1, 0);
    w.string("abc");
    let s = w.finish().unwrap_or_default();
    out.push((
        "a string's length counts its terminator",
        u32::from_le_bytes([s[8], s[9], s[10], s[11]]) == 4,
    ));
    out.push(("and its padding is not counted, only written", s.len() == HEADER + 4 + 4));
    let mut w = Writer::new(1, 0);
    w.string("abcd");
    let s5 = w.finish().unwrap_or_default();
    out.push((
        "a string that fills the boundary is still padded, because the terminator pushes it over",
        s5.len() == HEADER + 4 + 8,
    ));

    let mut w = Writer::new(1, 0);
    w.nil_string();
    let nil = w.finish().unwrap_or_default();
    let mut w = Writer::new(1, 0);
    w.string("");
    let mt = w.finish().unwrap_or_default();
    out.push(("nil and empty are different lengths on the wire", nil.len() != mt.len()));
    out.push((
        "and they read back as different values",
        matches!(frame(&nil), Ok(Some(m)) if m.reader().string() == Ok(None))
            && matches!(frame(&mt), Ok(Some(m)) if matches!(m.reader().string(), Ok(Some(ref t)) if t.is_empty())),
    ));

    let mut w = Writer::new(1, 0);
    w.uint(64); // a length far past the message
    w.uint(0);
    let long = w.finish().unwrap_or_default();
    out.push((
        "a string whose length runs past its message is refused",
        matches!(frame(&long), Ok(Some(m)) if m.reader().string() == Err(Error::StringLength)),
    ));

    let mut w = Writer::new(1, 0);
    w.uint(4);
    w.buf.extend_from_slice(b"abcd"); // four bytes, no terminator where the length says
    let unterminated = w.finish().unwrap_or_default();
    out.push((
        "a string whose length does not land on a terminator is refused",
        matches!(frame(&unterminated), Ok(Some(m)) if m.reader().string() == Err(Error::StringTerminator)),
    ));

    let mut w = Writer::new(1, 0);
    w.uint(3);
    w.buf.extend_from_slice(&[0xFF, 0xFE, 0x00, 0x00]);
    let junk = w.finish().unwrap_or_default();
    out.push((
        "a string that is not UTF-8 is refused rather than mangled into one that compares equal to something",
        matches!(frame(&junk), Ok(Some(m)) if m.reader().string() == Err(Error::StringUtf8)),
    ));

    let mut w = Writer::new(1, 0);
    w.string("a\u{0}b");
    out.push((
        "writing a string with a NUL inside it is refused, since its length would disagree with its terminator",
        w.finish() == Err(Error::StringTerminator),
    ));

    // ---- arrays ----
    let mut w = Writer::new(1, 0);
    w.array(&[9, 9, 9]);
    let arr = w.finish().unwrap_or_default();
    out.push((
        "an array is padded to four bytes while its length stays exact",
        arr.len() == HEADER + 4 + 4
            && matches!(frame(&arr), Ok(Some(m)) if matches!(m.reader().array(), Ok(ref a) if a.len() == 3)),
    ));

    // ---- running off the end ----
    out.push((
        "an argument past the end of a message is refused",
        matches!(frame(&bare), Ok(Some(m)) if m.reader().uint() == Err(Error::Truncated)),
    ));

    // ---- descriptors ----
    let mut q: VecDeque<u32> = VecDeque::new();
    q.push_back(11);
    q.push_back(22);
    let mut r = Reader::new(&[]);
    let first = r.fd(&mut q);
    let then = r.fd(&mut q);
    out.push((
        "an fd argument consumes no bytes, so an empty body still yields one",
        first == Ok(11) && r.at() == 0,
    ));
    out.push(("descriptors come out in the order they were attached", then == Ok(22)));
    out.push((
        "an fd argument with nothing in the queue is a protocol failure and not a wait, because the bytes cannot arrive before the descriptor",
        r.fd(&mut q) == Err(Error::NoDescriptor),
    ));

    // ---- size limits ----
    let mut w = Writer::new(1, 0);
    w.array(&alloc::vec![0u8; MAX_MESSAGE]);
    out.push((
        "a message too large for the size field is refused rather than truncated into a valid-looking one",
        w.finish() == Err(Error::TooLarge),
    ));

    // ---- what a refusal tells the client ----
    out.push((
        "an id failure reports itself as an invalid object",
        Error::IdInUse.code() == ERR_INVALID_OBJECT && Error::NoObject.code() == ERR_INVALID_OBJECT,
    ));
    out.push((
        "and a malformed message reports an implementation error, since there is no code for an unparseable stream",
        Error::SizeUnaligned.code() == ERR_IMPLEMENTATION,
    ));

    out
}
