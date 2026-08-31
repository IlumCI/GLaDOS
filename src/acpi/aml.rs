//! AML: the bytecode ACPI firmware ships, and the namespace it declares.
//!
//! This half builds the namespace and does not evaluate anything. That split
//! is deliberate and it is what makes an ACPI interpreter a bounded job rather
//! than an open-ended one, because the two halves have opposite obligations.
//!
//! **The walk has to be exact.** AML is a self-describing byte stream with
//! package lengths embedded in it, so one misread length does not lose one
//! object, it desynchronises everything after it. A parser that is ninety per
//! cent right is not ninety per cent useful; it is wrong in a way that
//! produces a confident and complete-looking namespace full of names the
//! firmware never wrote. So the only acceptable result is consuming the table
//! to its final byte, and that is asserted rather than hoped for -- the same
//! bargain `tools/v4.py` makes about checkpoints, for the same reason.
//!
//! **The evaluator, later, is allowed to be partial.** It runs only methods it
//! is asked for, and an opcode it does not implement is one method returning
//! an error that names the opcode. That is a missing line rather than a
//! mystery.
//!
//! ### Why this is smaller than it looks
//!
//! Everything that declares a name is delimited by a package length: `Scope`,
//! `Device`, `Method`, `Field`, `ThermalZone` and the rest all carry their own
//! byte count. **So the namespace walk never enters a method body and never
//! parses an expression.** It steps over each body by its declared length and
//! records where it was.
//!
//! That matters because the one genuinely hard problem in AML parsing lives
//! inside method bodies: a bare name followed by arguments is a method call,
//! and how many arguments to consume depends on how that method was declared,
//! which may be in a table not yet loaded. ACPICA solves it with multiple
//! passes. Skipping bodies wholesale means never meeting it here, and by the
//! time the evaluator does meet it the namespace is complete and the arity is
//! simply known.
//!
//! ### What a node holds
//!
//! A name, a place in the tree, a kind, and the byte range of its definition.
//! Nothing is decoded eagerly: an `OperationRegion`'s offset can be an
//! expression, a `Name` can hold a package built from other names, and
//! evaluating either at load time would mean running AML before the namespace
//! that AML refers to exists. Ranges now, values on demand.

use alloc::vec::Vec;

/// A four-character ACPI name. Padded with `_`, never NUL.
pub type Seg = [u8; 4];

/// The root, spelled the way ACPI spells it.
pub const ROOT: Seg = *b"\\___";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// A name that exists only to hold others. `Scope` makes one, and so does
    /// a path like `\_SB.PCI0` naming a parent that was never declared.
    Scope,
    Device,
    Processor,
    ThermalZone,
    PowerResource,
    /// A data object. The bytes are its `DataRefObject`.
    Name,
    /// Bytes are the body. `args` is how many arguments it takes, which the
    /// evaluator needs before it can parse a call to it.
    Method { args: u8, serialized: bool },
    /// Bytes are the offset and length expressions, unevaluated.
    OpRegion { space: u8 },
    /// One named window onto a region: where it starts in bits and how wide
    /// it is.
    ///
    /// The offsets accumulate across an element list, so a reserved run moves
    /// everything after it. That is why the list is walked rather than
    /// skipped: a partial read does not lose the last few fields, it gives
    /// every field after the gap the wrong address.
    Field { region: usize, flags: u8, offset: u32, width: u32 },
    IndexField { index: usize, data: usize, flags: u8, offset: u32, width: u32 },
    BankField { region: usize, bank: usize, flags: u8, offset: u32, width: u32 },
    Mutex { level: u8 },
    Event,
    /// A field cut out of a buffer by `CreateWordField` and its relatives.
    /// Bytes are the source and index expressions.
    BufferField { bits: u16 },
    /// Points at whatever it was aliased to, resolved after the walk.
    Alias,
}

/// What a field list hangs off, so one walker serves all three kinds.
#[derive(Clone, Copy)]
enum Owner {
    Region(usize),
    Index { index: usize, data: usize },
    Bank { region: usize, bank: usize },
}

pub struct Node {
    pub name: Seg,
    pub parent: usize,
    pub children: Vec<usize>,
    pub kind: Kind,
    /// Which AML table this came from, and the byte range within its body.
    pub table: usize,
    pub body: (usize, usize),
}

pub struct Namespace {
    nodes: Vec<Node>,
    /// One entry per AML table, in load order, so a node's range can be read.
    tables: Vec<&'static [u8]>,
    /// Names defined more than once. Legal (an SSDT may extend a scope) and
    /// worth counting, because a large number means the walk is inventing
    /// them.
    pub redefinitions: usize,
}

/// Where a walk stopped, and why.
///
/// Carries the offset because an AML fault is always "which byte", and a
/// message without one costs a boot to act on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stop {
    pub table: usize,
    pub at: usize,
    pub why: Why,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Why {
    /// An opcode this walk does not know how to step over. The byte is
    /// carried, because the fix is one table row and the row needs the number.
    Unknown(u8),
    /// An extended (0x5B-prefixed) opcode this walk does not know.
    UnknownExt(u8),
    /// A package length that runs past the end of its container.
    Overrun,
    /// A name with no segments where one was required.
    BadName,
    /// The walk finished early or late. This is the one that matters.
    NotAtEnd,
    /// Nesting deeper than anything real, which means a length was misread and
    /// the walk is following rubbish.
    TooDeep,
}

impl Namespace {
    pub fn new() -> Namespace {
        let root = Node {
            name: ROOT,
            parent: 0,
            children: Vec::new(),
            kind: Kind::Scope,
            table: 0,
            body: (0, 0),
        };
        Namespace { nodes: alloc::vec![root], tables: Vec::new(), redefinitions: 0 }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn node(&self, i: usize) -> &Node {
        &self.nodes[i]
    }

    pub fn table_bytes(&self, i: usize) -> Option<&'static [u8]> {
        self.tables.get(i).copied()
    }

    /// The bytes a node's definition occupies.
    pub fn body_of(&self, i: usize) -> &'static [u8] {
        let n = &self.nodes[i];
        match self.tables.get(n.table) {
            Some(t) if n.body.1 <= t.len() => &t[n.body.0..n.body.1],
            _ => &[],
        }
    }

    /// A node's full path, for printing. Allocates; not for hot paths.
    pub fn path(&self, mut i: usize) -> alloc::string::String {
        let mut parts: Vec<Seg> = Vec::new();
        while i != 0 {
            parts.push(self.nodes[i].name);
            i = self.nodes[i].parent;
        }
        let mut s = alloc::string::String::from("\\");
        for (k, seg) in parts.iter().rev().enumerate() {
            if k > 0 {
                s.push('.');
            }
            for c in seg.iter() {
                if *c != b'_' || s.ends_with('\\') || k > 0 {
                    s.push(*c as char);
                }
            }
            while s.ends_with('_') {
                s.pop();
            }
        }
        s
    }

    fn child(&self, parent: usize, name: Seg) -> Option<usize> {
        self.nodes[parent].children.iter().copied().find(|c| self.nodes[*c].name == name)
    }

    /// Find or make a child. Returning the existing one is what lets an SSDT
    /// extend a scope the DSDT opened, which is the normal case rather than a
    /// conflict.
    fn ensure(&mut self, parent: usize, name: Seg, kind: Kind, table: usize, body: (usize, usize)) -> usize {
        if let Some(i) = self.child(parent, name) {
            if self.nodes[i].kind == Kind::Scope && kind != Kind::Scope {
                // A path mentioned before it was declared. Fill it in.
                self.nodes[i].kind = kind;
                self.nodes[i].table = table;
                self.nodes[i].body = body;
            } else if kind != Kind::Scope {
                self.redefinitions += 1;
            }
            return i;
        }
        let i = self.nodes.len();
        self.nodes.push(Node { name, parent, children: Vec::new(), kind, table, body });
        self.nodes[parent].children.push(i);
        i
    }

    /// Resolve a name the way ACPI does.
    ///
    /// A rooted name starts at the root. A relative one with several segments
    /// is taken from the current scope. A single segment searches upward
    /// through every parent, which is the rule that lets `_STA` inside a
    /// device find the device's own `_STA` and lets a shared helper be found
    /// from anywhere below it.
    pub fn resolve(&self, from: usize, path: &Path) -> Option<usize> {
        let mut here = if path.rooted { 0 } else { from };
        for _ in 0..path.parents {
            here = self.nodes[here].parent;
        }
        if path.segs.is_empty() {
            return Some(here);
        }
        if path.rooted || path.parents > 0 || path.segs.len() > 1 {
            for seg in path.segs.iter() {
                here = self.child(here, *seg)?;
            }
            return Some(here);
        }
        // One relative segment: search upward.
        let seg = path.segs[0];
        loop {
            if let Some(i) = self.child(here, seg) {
                return Some(i);
            }
            if here == 0 {
                return None;
            }
            here = self.nodes[here].parent;
        }
    }

    /// Every node whose name matches, anywhere in the tree.
    pub fn find_all(&self, name: Seg) -> Vec<usize> {
        (0..self.nodes.len()).filter(|i| self.nodes[*i].name == name).collect()
    }
}

/// A parsed NameString.
#[derive(Clone, Debug, Default)]
pub struct Path {
    pub rooted: bool,
    pub parents: usize,
    pub segs: Vec<Seg>,
}

// --- the byte reader ----------------------------------------------------

struct Reader {
    b: &'static [u8],
    at: usize,
}

impl Reader {
    fn u8(&mut self) -> Option<u8> {
        let v = *self.b.get(self.at)?;
        self.at += 1;
        Some(v)
    }

    fn peek(&self) -> Option<u8> {
        self.b.get(self.at).copied()
    }

    fn skip(&mut self, n: usize) -> Option<()> {
        if self.at + n > self.b.len() {
            return None;
        }
        self.at += n;
        Some(())
    }

    /// A PkgLength, and the offset one byte past the package it introduces.
    ///
    /// The encoding is unusual and worth spelling out: the top two bits of the
    /// first byte say how many *more* bytes follow. With none, the low six
    /// bits are the length. With some, the low *four* bits are the least
    /// significant nibble and the following bytes are more significant. The
    /// length counts the PkgLength itself, so the end is measured from where
    /// it started rather than from where it finished.
    fn pkg(&mut self) -> Option<usize> {
        let start = self.at;
        let lead = self.u8()?;
        let extra = (lead >> 6) as usize;
        let mut len = if extra == 0 {
            (lead & 0x3F) as usize
        } else {
            let mut v = (lead & 0x0F) as usize;
            for i in 0..extra {
                v |= (self.u8()? as usize) << (4 + 8 * i);
            }
            v
        };
        len = len.checked_sub(self.at - start)?;
        let end = self.at.checked_add(len)?;
        if end > self.b.len() {
            return None;
        }
        Some(end)
    }

    /// The same encoding as `pkg`, decoded as a plain number.
    ///
    /// A field list reuses it to carry a bit count rather than a byte range,
    /// so the value is the answer instead of somewhere to jump to.
    fn pkg_raw(&mut self) -> Option<u32> {
        let lead = self.u8()?;
        let extra = (lead >> 6) as usize;
        if extra == 0 {
            return Some((lead & 0x3F) as u32);
        }
        let mut v = (lead & 0x0F) as u32;
        for i in 0..extra {
            v |= (self.u8()? as u32) << (4 + 8 * i);
        }
        Some(v)
    }

    fn name(&mut self) -> Option<Path> {
        let mut p = Path::default();
        loop {
            match self.peek()? {
                b'\\' => {
                    self.at += 1;
                    p.rooted = true;
                }
                b'^' => {
                    self.at += 1;
                    p.parents += 1;
                }
                _ => break,
            }
        }
        match self.peek()? {
            // NullName: a legal empty name, used by Scope(\) and friends.
            0x00 => {
                self.at += 1;
            }
            // DualNamePrefix
            0x2E => {
                self.at += 1;
                for _ in 0..2 {
                    p.segs.push(self.seg()?);
                }
            }
            // MultiNamePrefix, count then that many segments.
            0x2F => {
                self.at += 1;
                let n = self.u8()? as usize;
                for _ in 0..n {
                    p.segs.push(self.seg()?);
                }
            }
            _ => p.segs.push(self.seg()?),
        }
        Some(p)
    }

    fn seg(&mut self) -> Option<Seg> {
        if self.at + 4 > self.b.len() {
            return None;
        }
        let mut s = [0u8; 4];
        s.copy_from_slice(&self.b[self.at..self.at + 4]);
        self.at += 4;
        // A NameSeg is a lead character then three name characters. Anything
        // else means the walk is not where it thinks it is, and saying so here
        // turns a desync into a reported offset instead of a namespace full of
        // punctuation.
        let lead_ok = s[0].is_ascii_uppercase() || s[0] == b'_';
        let rest_ok = s[1..].iter().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == b'_');
        if lead_ok && rest_ok {
            Some(s)
        } else {
            None
        }
    }

    /// Step over one term argument, whatever shape it is.
    ///
    /// Named for what it started as, which was stepping over the data object a
    /// `Name` holds. A real DSDT needed more than that almost immediately:
    ///
    ///     OperationRegion (SANV, SystemMemory, SANB, SANL)
    ///     OperationRegion (MCHR, SystemMemory, Add (GMHB, 0x6100), 0x0100)
    ///
    /// The first gives its offset as a name and the second computes one. Both
    /// are ordinary on a machine with real firmware and neither appears in
    /// QEMU's nine kilobytes, which is why a table sixty times larger was
    /// worth going and getting.
    ///
    /// This does not evaluate anything. It only needs each opcode's shape:
    /// how many term arguments follow it, and how many targets after those.
    /// That is a finite table and it is the whole of the function.
    fn skip_data(&mut self) -> Option<()> {
        let op = self.u8()?;
        let (args, targets) = match op {
            // Constants and inline literals.
            0x00 | 0x01 | 0xFF => (0, 0),
            0x0A => {
                self.skip(1)?;
                (0, 0)
            }
            0x0B => {
                self.skip(2)?;
                (0, 0)
            }
            0x0C => {
                self.skip(4)?;
                (0, 0)
            }
            0x0E => {
                self.skip(8)?;
                (0, 0)
            }
            0x0D => {
                while self.u8()? != 0 {}
                (0, 0)
            }
            // Buffer, Package, VarPackage: self-delimiting, so their contents
            // never have to be understood here.
            0x11 | 0x12 | 0x13 => {
                let end = self.pkg()?;
                self.at = end;
                (0, 0)
            }
            // Locals and arguments.
            0x60..=0x6E => (0, 0),
            // Two arguments and a target: the arithmetic and bitwise set,
            // plus Concat, Index and ToString.
            0x72 | 0x73 | 0x74 | 0x77 | 0x79 | 0x7A | 0x7B | 0x7C | 0x7D | 0x7E | 0x7F | 0x85
            | 0x88 | 0x9C => (2, 1),
            // Divide alone takes two targets, remainder before quotient.
            0x78 => (2, 2),
            // One argument and a target.
            0x80 | 0x81 | 0x82 | 0x96 | 0x97 | 0x98 | 0x99 | 0x9D => (1, 1),
            // Store is an argument and a target; RefOf and the two counters
            // are a target alone.
            0x70 => (1, 1),
            0x71 | 0x75 | 0x76 => (0, 1),
            // One argument, no target.
            0x83 | 0x87 | 0x8E => (1, 0),
            // LNot, which doubles as the prefix for the three negated
            // comparisons. Whether it takes one argument or two depends on the
            // byte after it.
            0x92 => match self.peek() {
                Some(0x93) | Some(0x94) | Some(0x95) => {
                    self.at += 1;
                    (2, 0)
                }
                _ => (1, 0),
            },
            0x90 | 0x91 | 0x93 | 0x94 | 0x95 => (2, 0),
            0x9E => (3, 1),
            0x89 => (6, 0),
            // Extended opcodes that can appear in an argument.
            0x5B => match self.u8()? {
                // Revision, Debug.
                0x30 | 0x31 => (0, 0),
                // CondRefOf takes a name and a target.
                0x12 => (1, 1),
                // FromBCD, ToBCD.
                0x28 | 0x29 => (1, 1),
                // Acquire: a target and a 16-bit timeout.
                0x23 => {
                    self.skip(2)?;
                    (0, 1)
                }
                _ => return None,
            },
            // A bare name. Stepped over as a leaf: in this position it is
            // almost always an object rather than a call, and a call whose
            // arguments were misread would leave the walk somewhere other than
            // the last byte, which is checked.
            0x5C | b'^' | b'_' | b'A'..=b'Z' | 0x2E | 0x2F => {
                self.at -= 1;
                self.name()?;
                (0, 0)
            }
            _ => return None,
        };
        for _ in 0..args {
            self.skip_data()?;
        }
        for _ in 0..targets {
            self.skip_target()?;
        }
        Some(())
    }

    /// Step over a target: nothing, a local, an argument, or a name.
    fn skip_target(&mut self) -> Option<()> {
        match self.peek()? {
            0x00 => {
                self.at += 1;
                Some(())
            }
            0x60..=0x6E => {
                self.at += 1;
                Some(())
            }
            // Index and DerefOf are legal targets and carry their own
            // arguments.
            0x88 | 0x83 => self.skip_data(),
            0x5B => {
                self.at += 1;
                self.skip(1)?;
                Some(())
            }
            _ => {
                self.name()?;
                Some(())
            }
        }
    }
}

// --- the walk -----------------------------------------------------------

/// Nesting deeper than this means a length was misread and the walk is
/// following rubbish. Real firmware nests perhaps six deep.
const MAX_DEPTH: usize = 32;

/// What one table's walk established.
pub struct Loaded {
    pub nodes: usize,
    /// Conditional blocks skipped whole. Names inside them are not defined,
    /// and the count is here so that is a visible number rather than a silent
    /// absence.
    pub skipped_conditionals: usize,
    pub stop: Option<Stop>,
}

impl Namespace {
    /// Walk one AML table and add what it declares.
    ///
    /// The bytes must outlive the namespace, which for firmware tables they
    /// do: ACPI memory is identity-mapped and never handed to any allocator.
    pub fn load(&mut self, body: &'static [u8]) -> Loaded {
        let table = self.tables.len();
        self.tables.push(body);
        let before = self.nodes.len();
        let mut r = Reader { b: body, at: 0 };
        let mut skipped = 0usize;
        let stop = self.terms(&mut r, body.len(), 0, table, 0, &mut skipped);
        // The claim that matters. Anything but the exact end means a package
        // length was misread, and every name after that point is fiction.
        let stop = match stop {
            Some(s) => Some(s),
            None if r.at != body.len() => Some(Stop { table, at: r.at, why: Why::NotAtEnd }),
            None => None,
        };
        Loaded { nodes: self.nodes.len() - before, skipped_conditionals: skipped, stop }
    }

    /// Walk a term list up to `end`, defining into `scope`.
    fn terms(
        &mut self,
        r: &mut Reader,
        end: usize,
        scope: usize,
        table: usize,
        depth: usize,
        skipped: &mut usize,
    ) -> Option<Stop> {
        if depth > MAX_DEPTH {
            return Some(Stop { table, at: r.at, why: Why::TooDeep });
        }
        while r.at < end {
            let at = r.at;
            if let Some(s) = self.term(r, scope, table, depth, skipped) {
                return Some(s);
            }
            // A term that consumed nothing would spin here forever, and a
            // length of zero is exactly what a misread produces.
            if r.at <= at || r.at > end {
                return Some(Stop { table, at, why: Why::Overrun });
            }
        }
        None
    }

    fn term(
        &mut self,
        r: &mut Reader,
        scope: usize,
        table: usize,
        depth: usize,
        skipped: &mut usize,
    ) -> Option<Stop> {
        let at = r.at;
        let op = match r.u8() {
            Some(v) => v,
            None => return Some(Stop { table, at, why: Why::Overrun }),
        };

        match op {
            // AliasOp: the source, then the new name.
            0x06 => {
                if r.name().is_none() {
                    return Some(Stop { table, at, why: Why::BadName });
                }
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                match self.place(scope, &n) {
                    Some((p, seg)) => {
                        self.ensure(p, seg, Kind::Alias, table, (at, r.at));
                    }
                    None => return Some(Stop { table, at, why: Why::BadName }),
                }
            }
            // NameOp: a name and the object it holds.
            0x08 => {
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let start = r.at;
                if r.skip_data().is_none() {
                    let b = r.b.get(start).copied().unwrap_or(0);
                    return Some(Stop { table, at: start, why: Why::Unknown(b) });
                }
                match self.place(scope, &n) {
                    Some((p, seg)) => {
                        self.ensure(p, seg, Kind::Name, table, (start, r.at));
                    }
                    None => return Some(Stop { table, at, why: Why::BadName }),
                }
            }
            // ScopeOp: open an existing name and define into it.
            0x10 => {
                let end = match r.pkg() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let inner = match self.open(scope, &n) {
                    Some(i) => i,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                if let Some(s) = self.terms(r, end, inner, table, depth + 1, skipped) {
                    return Some(s);
                }
                r.at = end;
            }
            // MethodOp: recorded, never entered. The whole reason this walk is
            // small: a method body is where the one genuinely hard AML parsing
            // problem lives, and stepping over it by length never meets it.
            0x14 => {
                let end = match r.pkg() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let flags = match r.u8() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let kind = Kind::Method { args: flags & 0x07, serialized: flags & 0x08 != 0 };
                match self.place(scope, &n) {
                    Some((p, seg)) => {
                        self.ensure(p, seg, kind, table, (r.at, end));
                    }
                    None => return Some(Stop { table, at, why: Why::BadName }),
                }
                r.at = end;
            }
            // If, Else and While at the top level.
            //
            // Skipped whole, and counted. Their predicates are expressions,
            // and stepping over an expression needs length rules this walk
            // deliberately does not have. Descending unconditionally would be
            // worse than skipping rather than better: it would define both
            // arms of a choice the firmware makes, so a machine would appear
            // to have devices it does not. Firmware uses these at the top
            // level for `_OSI` checks, and the count says whether this machine
            // is one that does.
            0xA0 | 0xA1 | 0xA2 => {
                let end = match r.pkg() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                *skipped += 1;
                r.at = end;
            }
            // CreateByteField and its relatives: a source buffer, an index,
            // and the name being cut out of it. They declare a name, so they
            // get a node rather than being stepped over.
            0x8A | 0x8B | 0x8C | 0x8D | 0x8F => {
                let bits: u16 = match op {
                    0x8A => 32,
                    0x8B => 16,
                    0x8C => 8,
                    0x8D => 1,
                    _ => 64,
                };
                let start = r.at;
                for _ in 0..2 {
                    if r.skip_data().is_none() {
                        return Some(Stop { table, at, why: Why::Overrun });
                    }
                }
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                match self.place(scope, &n) {
                    Some((p, seg)) => {
                        self.ensure(p, seg, Kind::BufferField { bits }, table, (start, r.at));
                    }
                    None => return Some(Stop { table, at, why: Why::BadName }),
                }
            }
            // NoopOp.
            0xA3 => {}
            0x5B => {
                let ext = match r.u8() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                return self.ext_term(r, ext, at, scope, table, depth, skipped);
            }
            other => {
                // A bare expression at the top level. `Store (GSIP, SIPV)` is
                // the first one a real DSDT does, and ACPI says such a term
                // executes when the table loads.
                //
                // It is stepped over instead, and that is a deliberate
                // limit rather than an omission: **loading a namespace must
                // not run anything.** Executing firmware bytecode as a side
                // effect of reading it would mean touching hardware before
                // anybody asked, on a kernel where a fault outside a guard is
                // fatal. The cost is that a name initialised by a top-level
                // store reads as whatever it was declared with.
                r.at = at;
                if r.skip_data().is_some() {
                    return None;
                }
                return Some(Stop { table, at, why: Why::Unknown(other) });
            }
        }
        None
    }

    fn ext_term(
        &mut self,
        r: &mut Reader,
        ext: u8,
        at: usize,
        scope: usize,
        table: usize,
        depth: usize,
        skipped: &mut usize,
    ) -> Option<Stop> {
        // The declarations that carry a package length and a term list, which
        // is every container kind. Handled together because they differ only
        // in how many fixed bytes sit between the name and the body.
        let container = match ext {
            0x82 => Some((Kind::Device, 0usize)),
            0x83 => Some((Kind::Processor, 6)), // id, pblk address, pblk length
            0x84 => Some((Kind::PowerResource, 3)), // system level, resource order
            0x85 => Some((Kind::ThermalZone, 0)),
            _ => None,
        };
        if let Some((kind, fixed)) = container {
            let end = match r.pkg() {
                Some(v) => v,
                None => return Some(Stop { table, at, why: Why::Overrun }),
            };
            let n = match r.name() {
                Some(v) => v,
                None => return Some(Stop { table, at, why: Why::BadName }),
            };
            if r.skip(fixed).is_none() {
                return Some(Stop { table, at, why: Why::Overrun });
            }
            let inner = match self.place(scope, &n) {
                Some((p, seg)) => self.ensure(p, seg, kind, table, (r.at, end)),
                None => return Some(Stop { table, at, why: Why::BadName }),
            };
            if let Some(s) = self.terms(r, end, inner, table, depth + 1, skipped) {
                return Some(s);
            }
            r.at = end;
            return None;
        }

        match ext {
            // CreateField: source, bit index, bit count, then the name. The
            // only one of the family that takes a length, and the only one
            // under the extended prefix.
            0x13 => {
                let start = r.at;
                for _ in 0..3 {
                    if r.skip_data().is_none() {
                        return Some(Stop { table, at, why: Why::Overrun });
                    }
                }
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                match self.place(scope, &n) {
                    Some((p, seg)) => {
                        self.ensure(p, seg, Kind::BufferField { bits: 0 }, table, (start, r.at));
                    }
                    None => return Some(Stop { table, at, why: Why::BadName }),
                }
            }
            // MutexOp: name, sync level.
            0x01 => {
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let level = match r.u8() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                match self.place(scope, &n) {
                    Some((p, seg)) => {
                        self.ensure(p, seg, Kind::Mutex { level: level & 0x0F }, table, (at, r.at));
                    }
                    None => return Some(Stop { table, at, why: Why::BadName }),
                }
            }
            // EventOp: just a name.
            0x02 => {
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                match self.place(scope, &n) {
                    Some((p, seg)) => {
                        self.ensure(p, seg, Kind::Event, table, (at, r.at));
                    }
                    None => return Some(Stop { table, at, why: Why::BadName }),
                }
            }
            // OpRegionOp: name, address space, then offset and length as term
            // arguments. Kept unevaluated: on a real machine one of these is
            // routinely computed from another name, and working it out here
            // would mean running AML before the namespace it refers to exists.
            0x80 => {
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let space = match r.u8() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let start = r.at;
                for _ in 0..2 {
                    let here = r.at;
                    if r.skip_data().is_none() {
                        let b = r.b.get(here).copied().unwrap_or(0);
                        return Some(Stop { table, at: here, why: Why::Unknown(b) });
                    }
                }
                match self.place(scope, &n) {
                    Some((p, seg)) => {
                        self.ensure(p, seg, Kind::OpRegion { space }, table, (start, r.at));
                    }
                    None => return Some(Stop { table, at, why: Why::BadName }),
                }
            }
            // FieldOp: region, flags, then the element list.
            0x81 => {
                let end = match r.pkg() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let n = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let flags = match r.u8() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let region = self.resolve(scope, &n).unwrap_or(0);
                self.add_fields(r, end, scope, table, Owner::Region(region), flags);
                r.at = end;
            }
            // IndexFieldOp: two names, flags, elements.
            0x86 => {
                let end = match r.pkg() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let i = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let d = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let flags = match r.u8() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let index = self.resolve(scope, &i).unwrap_or(0);
                let data = self.resolve(scope, &d).unwrap_or(0);
                self.add_fields(r, end, scope, table, Owner::Index { index, data }, flags);
                r.at = end;
            }
            // BankFieldOp: region, bank name, bank value, flags, elements.
            0x87 => {
                let end = match r.pkg() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let rg = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let bk = match r.name() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::BadName }),
                };
                let here = r.at;
                if r.skip_data().is_none() {
                    let b = r.b.get(here).copied().unwrap_or(0);
                    return Some(Stop { table, at: here, why: Why::Unknown(b) });
                }
                let flags = match r.u8() {
                    Some(v) => v,
                    None => return Some(Stop { table, at, why: Why::Overrun }),
                };
                let region = self.resolve(scope, &rg).unwrap_or(0);
                let bank = self.resolve(scope, &bk).unwrap_or(0);
                self.add_fields(r, end, scope, table, Owner::Bank { region, bank }, flags);
                r.at = end;
            }
            other => return Some(Stop { table, at, why: Why::UnknownExt(other) }),
        }
        None
    }

    /// Define every named element of a field list, tracking where each one is.
    ///
    /// The bit offset accumulates across the list and every element moves it:
    /// a named field by its own width, and a reserved run by the width it
    /// declares, which is what `Offset()` compiles to. That is why the list is
    /// walked rather than skipped. Reading it partially would not lose the
    /// last few fields, it would give every field after the gap the wrong
    /// address, and a battery read from the wrong address is a plausible
    /// number rather than an error.
    ///
    /// Note these lengths are **counts, not ranges**. A field list reuses the
    /// package-length encoding to carry a number of bits, so the decoded value
    /// is the answer rather than an offset to jump to. That distinction is
    /// invisible until a field is one byte wide and reads as though it were
    /// zero.
    fn add_fields(
        &mut self,
        r: &mut Reader,
        end: usize,
        scope: usize,
        table: usize,
        owner: Owner,
        flags: u8,
    ) {
        let mut bit = 0u32;
        while r.at < end {
            let at = r.at;
            match r.peek() {
                // ReservedField: a gap, given as a bit count.
                Some(0x00) => {
                    r.at += 1;
                    let Some(width) = r.pkg_raw() else { return };
                    bit += width;
                }
                // AccessField changes the access size for what follows. The
                // size is taken from the field's own flags instead, which is
                // the common case; a list that changed it midway would be read
                // at the original size, and that is written down rather than
                // silently assumed.
                Some(0x01) => {
                    r.at += 1;
                    if r.skip(2).is_none() {
                        return;
                    }
                }
                Some(0x02) => {
                    r.at += 1;
                    if r.name().is_none() {
                        return;
                    }
                }
                Some(0x03) => {
                    r.at += 1;
                    if r.skip(3).is_none() {
                        return;
                    }
                }
                Some(_) => {
                    let Some(seg) = r.seg() else { return };
                    let Some(width) = r.pkg_raw() else { return };
                    let kind = match owner {
                        Owner::Region(region) => Kind::Field { region, flags, offset: bit, width },
                        Owner::Index { index, data } => {
                            Kind::IndexField { index, data, flags, offset: bit, width }
                        }
                        Owner::Bank { region, bank } => {
                            Kind::BankField { region, bank, flags, offset: bit, width }
                        }
                    };
                    self.ensure(scope, seg, kind, table, (at, r.at));
                    bit += width;
                }
                None => return,
            }
            if r.at <= at {
                return;
            }
        }
    }

    /// The node a path names, creating any missing parents.
    ///
    /// Separate from `place` because a scope opens a node and a declaration
    /// makes one, and the two differ on the empty path. `Scope (\)` is spelled
    /// with a NullName and means the root: a real, common construct that
    /// `place` reads as a name with nothing in it. That cost a boot, and the
    /// bytes said so plainly once they were printed: `10 49 04 5c 00`.
    fn open(&mut self, scope: usize, path: &Path) -> Option<usize> {
        let mut here = if path.rooted { 0 } else { scope };
        for _ in 0..path.parents {
            here = self.nodes[here].parent;
        }
        for seg in path.segs.iter() {
            here = self.ensure(here, *seg, Kind::Scope, 0, (0, 0));
        }
        Some(here)
    }

    /// Where a declared name should hang, creating any missing parents.
    ///
    /// `Device (\_SB.PCI0.LPCB)` names three levels and the first two may
    /// never have been declared as anything. They become scopes, and are
    /// filled in if a real declaration arrives later.
    fn place(&mut self, scope: usize, path: &Path) -> Option<(usize, Seg)> {
        let mut here = if path.rooted { 0 } else { scope };
        for _ in 0..path.parents {
            here = self.nodes[here].parent;
        }
        let (last, rest) = path.segs.split_last()?;
        for seg in rest {
            here = self.ensure(here, *seg, Kind::Scope, 0, (0, 0));
        }
        Some((here, *last))
    }
}
