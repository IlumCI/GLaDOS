//! Tree-walking interpreter.
//!
//! Working beats fast. This evaluates the AST directly; the single-pass JIT
//! that replaces it can be written against exactly the same AST and checked
//! against exactly the same results, which is much easier than debugging a
//! code generator with nothing to compare it to.
//!
//! Builtins reach straight into the kernel. That is the point of a ring-0
//! single-address-space OS: `peek`, `poke`, `inb` and `outb` at the prompt are
//! a hardware debugger with no driver, no ioctl and no permission model in the
//! way. It is also exactly how you shoot yourself -- a bad `peek` faults, and
//! the M2 reporter prints the address. That is a feature.

use super::parse::{BinOp, Expr, Stmt, Type, UnOp};
use alloc::collections::BTreeSet;
use crate::gfx::console::{self, PALETTE};
use crate::gfx::{self};
use crate::{kprint, kprintln};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i64),
    Str(String),
    /// A sequence, and the only compound value there is.
    ///
    /// Applications hold collections -- rows in a list, cells on a board,
    /// lines in a document -- and a language with no way to hold one can only
    /// write calculators. Values, not references: `push` returns a new list
    /// rather than mutating a shared one, so there is no aliasing to reason
    /// about and no question of what two names pointing at the same list
    /// means. It copies, and a copy of a list that fits on a screen is
    /// nothing.
    List(Vec<Value>),
    /// A record: its type name, and its fields in declared order.
    ///
    /// The names are carried in the value rather than looked up in the type
    /// table. It costs a string per field per instance and buys something
    /// worth more: a value that can be rendered, compared and read without
    /// the interpreter that made it. `render` is called from the desktop, from
    /// `check`, and from a shell that has no `Interp` in hand.
    ///
    /// A value like every other value here. Assigning one copies it, and there
    /// is no aliasing to explain to whoever -- or whatever -- is writing the
    /// program. That is the same bargain `List` already made.
    Rec(String, Vec<(String, Value)>),
    Nil,
}

impl Value {
    /// Whether a value is allowed where this type was declared.
    ///
    /// `Any` admits everything, including `Nil`, because a program that says
    /// nothing about a parameter should behave exactly as it did before types
    /// existed.
    pub fn fits(&self, t: &crate::aiksi::parse::Type) -> bool {
        use crate::aiksi::parse::Type;
        match (t, self) {
            (Type::Any, _) => true,
            (Type::Int, Value::Int(_)) => true,
            (Type::Str, Value::Str(_)) => true,
            (Type::List, Value::List(_)) => true,
            (Type::Nil, Value::Nil) => true,
            (Type::Rec(want), Value::Rec(have, _)) => want == have,
            _ => false,
        }
    }

    /// What this value is, for an error message.
    pub fn type_name(&self) -> &str {
        match self {
            Value::Int(_) => "int",
            Value::Str(_) => "str",
            Value::List(_) => "list",
            Value::Rec(n, _) => n,
            Value::Nil => "nil",
        }
    }

    pub fn truthy(&self) -> bool {
        match self {
            Value::Int(v) => *v != 0,
            Value::Str(s) => !s.is_empty(),
            Value::List(v) => !v.is_empty(),
            // A record always exists, and one with no fields cannot be
            // declared, so there is nothing for it to be empty of.
            Value::Rec(..) => true,
            Value::Nil => false,
        }
    }

    pub fn as_int(&self) -> Result<i64, String> {
        match self {
            Value::Int(v) => Ok(*v),
            Value::Str(_) => Err("expected a number, found a string".to_string()),
            Value::List(_) => Err("expected a number, found a list".to_string()),
            Value::Rec(n, _) => Err(format!("expected a number, found a {}", n)),
            Value::Nil => Err("expected a number, found nothing".to_string()),
        }
    }

    pub fn render(&self) -> String {
        match self {
            Value::Int(v) => format!("{}", v),
            Value::Str(s) => s.clone(),
            // `Host{name: "x", port: 80}`, which is how it was written apart
            // from the type name leading instead of calling. Legible in a
            // window, in a log and in a verdict, which is where these are read.
            Value::Rec(name, fields) => {
                let mut out = String::from(name.as_str());
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(k);
                    out.push_str(": ");
                    out.push_str(&v.render());
                }
                out.push('}');
                out
            }
            Value::List(items) => {
                let mut out = String::from("[");
                for (i, v) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&v.render());
                }
                out.push(']');
                out
            }
            Value::Nil => String::new(),
        }
    }
}

/// Without this, `while (1) {}` at the prompt wedges the shell task forever.
/// There is no Ctrl-C to rescue you: the clock task keeps running because of
/// preemption, but nothing would ever schedule the shell back into a state
/// where it could read a key.
const STEP_BUDGET: u64 = 20_000_000;

/// What a program gets when something other than a person is waiting for it.
///
/// The full budget is sized for the prompt, where the operator can see a long
/// loop running and stop it. It is the wrong size for a program the desktop
/// calls: `app::document` runs an application's `rows()` on **every repaint**,
/// so a generated loop that takes a second makes the window manager feel
/// broken, and the symptom points at the compositor rather than at the
/// application. Small enough to bound a repaint, large enough that no
/// reasonable list-building loop reaches it.
pub const DRAW_BUDGET: u64 = 200_000;

/// A named procedure: its parameters and its body.
///
/// Held behind an `Rc` wherever it is stored, because a call used to take a
/// deep copy of one. `funcs.get(name).cloned()` clones the body -- the whole
/// statement tree, and every expression tree under it -- and it ran at every
/// user function call, on a structure that is immutable from the moment
/// `Stmt::Fn` declared it. `core bench` put the pair of clones a single vote
/// pays at roughly a fifth of the vote.
///
/// `Rc` and not `Arc`: there is one interpreter per call chain and none of
/// this crosses a task, which is the same single-core assumption `Racy`
/// rests on and the same grep target if that ever stops being true.
#[derive(Clone)]
struct Func {
    params: Vec<(String, Type)>,
    ret: Type,
    body: Vec<Stmt>,
}

/// A program's declarations, run once so they need not be run again.
///
/// Registering a function is not free: `Stmt::Fn` deep-copies the body,
/// because `run` walks a borrowed `Program` and storing it without a copy
/// would put a lifetime on `Interp` and thread it through every caller in the
/// kernel. Anything that runs one program many times with fresh state pays
/// that copy every time -- `voter::Core::vote` did, on every routing
/// decision, and `core bench` measured it at 813 ns of a 2,734 ns vote.
///
/// The fields are private and stay that way: this is a snapshot of another
/// interpreter's tables, not a thing to be edited.
pub struct Prepared {
    funcs: BTreeMap<String, alloc::rc::Rc<Func>>,
    recs: BTreeMap<String, Vec<(String, Type)>>,
    /// What the top level cost, so a prepared run reports what an armed one
    /// would. See `adopt`.
    steps: u64,
}

/// How deep calls may nest before it is called a runaway.
///
/// Recursion is not the reason. The step budget already stops a program that
/// loops forever, but it does not stop one that recurses forever, because
/// every frame is a fresh allocation and the kernel runs out of stack long
/// before twenty million steps -- and running out of stack in ring 0 with no
/// guard page is a triple fault, not an error message.
const MAX_DEPTH: usize = 64;

/// What a program is allowed to reach.
///
/// The gate lives here and not in `sysbox` or the shell for one reason: the
/// raw builtins are reachable from a bare expression at the prompt and from
/// any stored program `run` executes, so a check anywhere else has a hole
/// shaped like the other path. This is the only place both go through.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Caps {
    /// The operator's own interpreter. Everything, including `poke` and the
    /// I/O ports, which is the point of a ring-0 system with one address
    /// space: the prompt is a hardware debugger with nothing in the way.
    Operator,
    /// A stored application. No raw memory, no ports, no drawing outside its
    /// window, applets limited to those that do not mutate, and writes
    /// confined to its own subtree.
    Sandbox,
}

/// Record types the kernel itself hands back.
///
/// A builtin returning structured data used to have nowhere to put it, so
/// `pci_list` answered lines of text and every caller wrote the same fragile
/// `split` to take them apart. Now that records exist the kernel declares its
/// own, and they are registered in every interpreter at construction so an
/// annotation like `fn f(d: Device)` checks against something real rather than
/// against a name nobody declared.
///
/// Declared here rather than emitted by the arm that builds one, so that the
/// shape is written down in exactly one place and `words` could list it.
pub const KERNEL_RECS: &[(&str, &[(&str, Type)])] = &[
    (
        "Device",
        &[
            ("bus", Type::Int),
            ("dev", Type::Int),
            ("func", Type::Int),
            ("vendor", Type::Int),
            ("device", Type::Int),
            ("class", Type::Str),
        ],
    ),
    // The clock. `text` is the stamp `rtc_now` used to answer on its own, kept
    // as a field rather than dropped: every caller that wanted to *show* the
    // time had it, and taking it away would make each of them reimplement
    // zero-padded formatting -- which is precisely the hand-rolled string work
    // records exist to end. The numbers are what was missing; a program wanting
    // the hour was reduced to `substr(rtc_now(), 11, 2)`.
    (
        "Time",
        &[
            ("year", Type::Int),
            ("month", Type::Int),
            ("day", Type::Int),
            ("hour", Type::Int),
            ("minute", Type::Int),
            ("second", Type::Int),
            ("text", Type::Str),
        ],
    ),
    // An interface. `net_ifaces` answered a list of names, so everything else
    // an interface knows -- its address, whether it is up, what it has carried
    // -- was simply unreachable from Aiksi. This is the clearest case in the
    // sweep: not a struct flattened into text, a struct discarded.
    (
        "Iface",
        &[
            ("name", Type::Str),
            ("ip", Type::Str),
            ("netmask", Type::Str),
            ("gateway", Type::Str),
            ("dns", Type::Str),
            ("up", Type::Int),
            ("rx_packets", Type::Int),
            ("rx_bytes", Type::Int),
            ("tx_packets", Type::Int),
            ("tx_bytes", Type::Int),
            ("tx_dropped", Type::Int),
        ],
    ),
    (
        "Config",
        &[
            ("ip", Type::Str),
            ("gateway", Type::Str),
            ("netmask", Type::Str),
            ("dns", Type::Str),
        ],
    ),
    // `HEAP.stats()` is a tuple, which is a struct that lost its field names on
    // the way out. It reached Aiksi as two builtins that had to be called
    // together to mean anything.
    ("Mem", &[("used", Type::Int), ("total", Type::Int)]),
    (
        "Task",
        &[
            ("index", Type::Int),
            ("name", Type::Str),
            ("state", Type::Str),
            ("switches", Type::Int),
            // Whether this is the task asking. A program walking the list
            // otherwise has to call `task_current()` and compare indices, which
            // is the fragile join a record should remove.
            ("current", Type::Int),
        ],
    ),
    // What `ls` shows about one entry. `hash_of`, `size` and `is_dir` are three
    // calls resolving the same path three times to answer one question.
    (
        "Stat",
        &[
            ("name", Type::Str),
            ("hash", Type::Str),
            ("size", Type::Int),
            ("is_dir", Type::Int),
        ],
    ),
    // `tcp_state` and `tcp_error` are one answer in two calls, and the pair can
    // disagree: a program that reads the state, is preempted, and then reads
    // the error gets two different moments.
    ("Tcp", &[("state", Type::Str), ("error", Type::Str)]),
];

/// One of the kernel's own record shapes, by name.
///
/// A linear scan over eight entries, which beats a map that has to be built:
/// the map cost 49 allocations at every `Interp::new()` and this costs at
/// most eight string comparisons at the one moment a record type is actually
/// named. Nothing on the routing path names one at all.
fn kernel_rec(name: &str) -> Option<&'static [(&'static str, Type)]> {
    KERNEL_RECS.iter().find(|(n, _)| *n == name).map(|(_, fs)| *fs)
}

/// The fields of a record type, from whichever table declared it.
///
/// Two sources with two shapes -- a program's own `rec` owns its names, the
/// kernel's are `&'static str` -- and unifying them would mean allocating one
/// into the other's shape, which is the cost this split exists to remove. So
/// the difference is carried in a borrow instead, and the three call sites
/// read fields rather than hold the table.
enum Fields<'a> {
    Own(&'a [(String, Type)]),
    Kernel(&'static [(&'static str, Type)]),
}

impl<'a> Fields<'a> {
    fn len(&self) -> usize {
        match self {
            Fields::Own(fs) => fs.len(),
            Fields::Kernel(fs) => fs.len(),
        }
    }

    /// The name and declared type of field `i`. Callers check `len` first.
    fn at(&self, i: usize) -> (&str, &Type) {
        match self {
            Fields::Own(fs) => (fs[i].0.as_str(), &fs[i].1),
            Fields::Kernel(fs) => (fs[i].0, &fs[i].1),
        }
    }

    fn find(&self, name: &str) -> Option<&Type> {
        (0..self.len())
            .map(|i| self.at(i))
            .find(|(n, _)| *n == name)
            .map(|(_, t)| t)
    }
}

/// What a builtin touches.
///
/// Every builtin declares one, and `BUILTINS` is the only way to reach the
/// dispatch, so a builtin that is added without saying what it touches is not
/// callable at all. That inversion is the whole point.
///
/// It replaced two denylists. Those were correct for eleven raw builtins and
/// three drawing ones, and they stopped being correct the moment the language
/// was wired to the rest of the kernel: a denylist grants by default, so every
/// builtin anyone forgot to list -- sockets included -- would have been
/// reachable from a program the machine wrote for itself. The old comment said
/// as much about per-arm checks and the same argument finished the list off.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Touch {
    /// Values in, values out. Nothing outside the interpreter is read.
    Pure,
    /// Reads state that is not secret: the clock, the heap figure, the
    /// namespace, which interfaces exist.
    Read,
    /// Writes, and a sandboxed program may only write inside its own subtree.
    /// `may_write` is what enforces that; this only says a write happens.
    Write,
    /// Talks to the network.
    Net,
    /// Runs the model.
    Model,
    /// Paints outside any window.
    Draw,
    /// Reaches the machine directly, or changes the system.
    Raw,
}

impl Touch {
    /// Whether a stored program may do this unasked.
    ///
    /// Three classes and not a grant matrix. `manifest::Manifest` carries one
    /// bit for the same reason it gives: an operator approving a request has to
    /// hold the whole of it in their head, and "may write outside itself but
    /// not open sockets" is a sentence nobody can check against a program.
    /// Sandboxed or trusted is a question with an answer.
    fn sandboxable(self) -> bool {
        matches!(self, Touch::Pure | Touch::Read | Touch::Write)
    }

    fn why(self) -> &'static str {
        match self {
            Touch::Net => "talks to the network",
            Touch::Model => "runs the model",
            Touch::Draw => "draws outside any window",
            Touch::Raw => "reaches the machine directly",
            _ => "changes the system",
        }
    }
}

/// Every builtin, what it touches, and how many arguments it takes.
///
/// The single source of truth. `builtin` refuses anything absent from here
/// before it dispatches, so an arm added to the match without a row is dead
/// code and a row without an arm is a boot selftest failure. Neither can be
/// half-done, which is the property the old denylists could not offer.
///
/// Arity is `(min, max)`; `usize::MAX` for max means variadic.
pub const BUILTINS: &[(&str, Touch, usize, usize)] = &[
    // --- values -----------------------------------------------------------
    ("here", Touch::Pure, 0, 0),
    ("int", Touch::Pure, 1, 1),
    ("hex", Touch::Pure, 1, 1),
    ("list", Touch::Pure, 0, usize::MAX),
    ("len", Touch::Pure, 1, 1),
    ("get", Touch::Pure, 2, 2),
    ("push", Touch::Pure, 2, 2),
    ("set", Touch::Pure, 3, 3),
    // Transient: the console scrolls and nothing outlives the call.
    ("print", Touch::Pure, 0, usize::MAX),
    ("println", Touch::Pure, 0, usize::MAX),

    // --- reading the machine ----------------------------------------------
    ("ticks", Touch::Read, 0, 0),
    ("hz", Touch::Read, 0, 0),
    ("tasks", Touch::Read, 0, 0),
    ("heap", Touch::Read, 0, 0),
    ("width", Touch::Read, 0, 0),
    ("height", Touch::Read, 0, 0),
    ("read", Touch::Read, 1, 1),
    ("exists", Touch::Read, 1, 1),
    ("ls", Touch::Read, 1, 1),
    // Read, and then narrowed again per call by `applet_mutates`: the applet
    // table already answers "does this change anything", and asking it is
    // exact where a second list here would drift from it.
    ("applet", Touch::Read, 1, 1),

    // --- writing ----------------------------------------------------------
    // Sandboxed writes are confined by `may_write`, which resolves the path
    // first. The class says a write happens; the jail says where.
    ("write", Touch::Write, 2, 2),

    // --- the operator's terminal ------------------------------------------
    // Not sandboxable, and this is a tightening. The colour outlives the call
    // and the clear takes the operator's scrollback, so an application
    // repainting its rows could wipe the terminal underneath it. Nothing
    // stored uses either; the prompt runs with Operator caps and keeps both.
    ("cls", Touch::Raw, 0, 0),
    ("color", Touch::Raw, 1, 1),

    // --- drawing ----------------------------------------------------------
    ("pixel", Touch::Draw, 3, 3),
    ("rect", Touch::Draw, 5, 5),
    ("text", Touch::Draw, 4, 4),

    // --- text -------------------------------------------------------------
    //
    // A systems language whose only string operation is `+` cannot parse a
    // header, split a path or read a config file, so every program that needed
    // one grew a bad version of it in Aiksi. These are `str` and `char` in the
    // Rust underneath, named for what they do rather than for the method.
    ("upper", Touch::Pure, 1, 1),
    ("lower", Touch::Pure, 1, 1),
    ("trim", Touch::Pure, 1, 1),
    ("split", Touch::Pure, 2, 2),
    ("join", Touch::Pure, 2, 2),
    ("substr", Touch::Pure, 3, 3),
    ("find", Touch::Pure, 2, 2),
    ("replace", Touch::Pure, 3, 3),
    ("starts", Touch::Pure, 2, 2),
    ("ends", Touch::Pure, 2, 2),
    ("contains", Touch::Pure, 2, 2),
    ("chr", Touch::Pure, 1, 1),
    ("ord", Touch::Pure, 1, 1),
    ("repeat", Touch::Pure, 2, 2),
    ("pad", Touch::Pure, 2, 2),
    ("hexenc", Touch::Pure, 1, 1),
    ("hexdec", Touch::Pure, 1, 1),

    // --- arithmetic --------------------------------------------------------
    ("abs", Touch::Pure, 1, 1),
    ("min", Touch::Pure, 2, 2),
    ("max", Touch::Pure, 2, 2),
    ("clamp", Touch::Pure, 3, 3),
    ("sqrt", Touch::Pure, 1, 1),
    ("pow", Touch::Pure, 2, 2),

    // --- lists, beyond building one ----------------------------------------
    ("sort", Touch::Pure, 1, 1),
    ("reverse", Touch::Pure, 1, 1),
    ("slice", Touch::Pure, 3, 3),
    ("index", Touch::Pure, 2, 2),
    ("remove", Touch::Pure, 2, 2),
    ("range", Touch::Pure, 2, 2),

    // --- crate::dev::rtc, crate::time, crate::dev::lapic --------------------
    ("rtc_now", Touch::Read, 0, 0),
    ("rtc_unix", Touch::Read, 0, 0),
    ("uptime", Touch::Read, 0, 0),
    ("tsc", Touch::Read, 0, 0),
    ("tsc_mhz", Touch::Read, 0, 0),

    // --- crate::task -------------------------------------------------------
    ("task_count", Touch::Read, 0, 0),
    ("task_current", Touch::Read, 0, 0),
    ("task_switches", Touch::Read, 0, 0),
    // Yielding is not a read, but it is not a way to reach anything either:
    // the scheduler preempts at 100 Hz regardless, so this only gives up a
    // slice early. A long loop in a repaint path is bounded by the step
    // budget, not by whether it was polite.
    ("task_yield", Touch::Read, 0, 0),

    // --- crate::mem --------------------------------------------------------
    ("mem_used", Touch::Read, 0, 0),
    ("mem_total", Touch::Read, 0, 0),

    // --- crate::dev::pci ---------------------------------------------------
    ("pci_list", Touch::Read, 0, 0),

    // --- crate::net --------------------------------------------------------
    //
    // Status only, so far. Reading which interfaces exist and what address
    // this machine has tells a program about itself; it does not put a packet
    // on the wire, which is why these are Read and the sockets are not.
    ("net_ready", Touch::Read, 0, 0),
    ("net_ifaces", Touch::Read, 0, 0),
    ("net_ip", Touch::Read, 0, 0),
    ("net_gateway", Touch::Read, 0, 0),
    ("net_dns", Touch::Read, 0, 0),
    // The record forms. The scalars above stay: `mem_used()` is not improved
    // by becoming `mem_stats().used`, and a sweep that converted atomic
    // answers into records to be uniform would be making the language worse to
    // make a rule tidy. What gets a record is what is actually a struct.
    ("net_config", Touch::Read, 0, 0),
    ("mem_stats", Touch::Read, 0, 0),
    ("task_list", Touch::Read, 0, 0),
    ("stat", Touch::Read, 1, 1),

    // --- crate::net::dns, ::tcp, ::udp, ::tls -------------------------------
    //
    // The line between Read and Net is whether a packet leaves the machine.
    // Everything here puts one on the wire, so none of it is reachable from a
    // stored program without `app trust` -- which is the answer to "will it
    // write me nmap": it can, and you have to say so first.
    ("dns_resolve", Touch::Net, 1, 1),
    ("tcp_connect", Touch::Net, 3, 3),
    ("tcp_send", Touch::Net, 2, 2),
    ("tcp_recv", Touch::Net, 1, 1),
    ("tcp_close", Touch::Net, 0, 0),
    ("tcp_state", Touch::Net, 0, 0),
    ("tcp_error", Touch::Net, 0, 0),
    ("tcp_status", Touch::Net, 0, 0),
    ("http_get", Touch::Net, 3, 3),
    ("https_get", Touch::Net, 3, 3),
    ("https_identity", Touch::Net, 0, 0),
    ("udp_send", Touch::Net, 4, 4),
    ("ping", Touch::Net, 2, 2),

    // --- crate::ai ----------------------------------------------------------
    ("model_ready", Touch::Read, 0, 0),
    ("ask", Touch::Model, 2, 2),

    // --- crate::sysbox, beyond read and write -------------------------------
    ("hash_of", Touch::Read, 1, 1),
    ("size", Touch::Read, 1, 1),
    ("is_dir", Touch::Read, 1, 1),
    // Removal is a write, and confined by the same jail: `may_write` resolves
    // the path before comparing, so `../..` is not a way out of it.
    ("rm", Touch::Write, 1, 1),

    // --- the machine itself -----------------------------------------------
    ("peek8", Touch::Raw, 1, 1),
    ("peek16", Touch::Raw, 1, 1),
    ("peek32", Touch::Raw, 1, 1),
    ("peek64", Touch::Raw, 1, 1),
    ("poke8", Touch::Raw, 2, 2),
    ("poke32", Touch::Raw, 2, 2),
    ("poke64", Touch::Raw, 2, 2),
    ("inb", Touch::Raw, 1, 1),
    ("outb", Touch::Raw, 2, 2),
    ("inl", Touch::Raw, 1, 1),
    ("outl", Touch::Raw, 2, 2),
];

/// What a builtin touches, or `None` if there is no such builtin.
pub fn touch_of(name: &str) -> Option<Touch> {
    BUILTINS.iter().find(|(n, ..)| *n == name).map(|(_, t, ..)| *t)
}

/// Is this a builtin at all?
pub fn is_builtin(name: &str) -> bool {
    touch_of(name).is_some()
}

/// Every builtin a program with these capabilities may call.
pub fn available(caps: Caps) -> Vec<&'static str> {
    BUILTINS
        .iter()
        .filter(|(_, t, ..)| caps == Caps::Operator || t.sandboxable())
        .map(|(n, ..)| *n)
        .collect()
}

pub struct Interp {
    /// Innermost scope last. A name is looked up from the top down and
    /// assigned wherever it is already bound, so a function can read and
    /// update a global without ceremony, and a parameter shadows one without
    /// destroying it.
    scopes: Vec<BTreeMap<String, Value>>,
    funcs: BTreeMap<String, alloc::rc::Rc<Func>>,
    /// Set by `return`, and checked after every statement in a block. A
    /// sentinel rather than a control-flow type threaded through every
    /// signature, which for a tree-walker this size is the same thing with
    /// less to read.
    returning: Option<Value>,
    depth: usize,
    steps: u64,
    budget: u64,
    caps: Caps,
    /// The subtree a sandboxed program may write into. Absolute, with no
    /// trailing slash.
    jail: Option<String>,
    /// Scratch between builtins. See `set_note`.
    notes: BTreeMap<String, String>,
    /// Record types the *program* declared, by name, with fields in order.
    ///
    /// The ones the kernel returns are not copied in here. They used to be,
    /// and construction paid for it: `KERNEL_RECS` is eight shapes with 49
    /// names between them, so every `Interp::new()` allocated 49 `String`s
    /// and did eight tree inserts to reproduce a table that is immutable
    /// kernel data and identical in every interpreter ever built. On the
    /// routing path that is a per-decision cost -- `Core::vote` builds a
    /// fresh interpreter every time, deliberately -- and `core bench`
    /// measured it as the largest single component of a vote.
    ///
    /// Lookup therefore consults two tables (`fields_of`), and they stay
    /// disjoint because `Stmt::Rec` refuses a name the kernel returns. That
    /// guard was tidiness when both lived in one map, where a redeclaration
    /// would simply have overwritten; with two tables it is load-bearing,
    /// since otherwise a shadowed name would resolve by whichever table is
    /// searched first and nothing would say which.
    recs: BTreeMap<String, Vec<(String, Type)>>,
    /// Programs already imported, by resolved path.
    ///
    /// This is what makes `use` idempotent and what makes a cycle terminate:
    /// two files that import each other each find the other already present
    /// the second time round. A depth counter would also stop it and would
    /// stop it by failing; this stops it by being finished.
    imported: BTreeSet<String>,
}

impl Default for Interp {
    fn default() -> Self {
        Self::new()
    }
}

impl Interp {
    pub fn new() -> Self {
        Self {
            scopes: alloc::vec![BTreeMap::new()],
            funcs: BTreeMap::new(),
            returning: None,
            depth: 0,
            steps: 0,
            budget: STEP_BUDGET,
            caps: Caps::Operator,
            jail: None,
            notes: BTreeMap::new(),
            recs: BTreeMap::new(),
            imported: BTreeSet::new(),
        }
    }

    /// Where a record type is declared, program first.
    ///
    /// Order does not decide anything, because the two tables cannot hold the
    /// same name: `Stmt::Rec` refuses one the kernel returns. Program first is
    /// the cheaper miss anyway -- an empty `BTreeMap` answers without
    /// comparing a byte, which is the common case, since most programs
    /// declare no records and every interpreter now starts with none.
    fn fields_of(&self, name: &str) -> Option<Fields<'_>> {
        if let Some(fs) = self.recs.get(name) {
            return Some(Fields::Own(fs));
        }
        kernel_rec(name).map(Fields::Kernel)
    }

    /// An interpreter for a stored program, confined to one subtree.
    ///
    /// Every existing caller keeps `new` and keeps everything, which is what
    /// makes this safe to add: the prompt, the shell's session and the model's
    /// own tools are unchanged, and only code that opts in is confined.
    pub fn sandboxed(jail: &str) -> Self {
        let mut it = Self::new();
        it.caps = Caps::Sandbox;
        it.jail = Some(String::from(jail.trim_end_matches('/')));
        it
    }

    /// Lower the step budget. Cannot be raised above the default.
    pub fn with_step_budget(mut self, n: u64) -> Self {
        self.budget = n.min(STEP_BUDGET);
        self
    }

    pub fn caps(&self) -> Caps {
        self.caps
    }

    /// True if a sandboxed program may write here.
    ///
    /// The path is resolved first. A jail compared against what was typed is
    /// defeated by `../..`, which is the entire history of this kind of check.
    /// Call a function already defined in this interpreter, with values.
    ///
    /// Everything else that calls into Aiksi from Rust builds a source string
    /// and re-parses it -- `app::call_fn` quotes its argument into a literal
    /// and hands the lot back to the lexer. That is fine for a button press
    /// and wrong for anything on a hot path: a council core votes once per
    /// routing decision, and re-lexing a program to pass it a string would
    /// cost more than the vote.
    ///
    /// It also removes the quoting entirely, and quoting is where this class
    /// of bridge usually breaks.
    pub fn invoke(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        let Some(f) = self.funcs.get(name).cloned() else {
            return Err(format!("no function '{}'", name));
        };
        // A refcount bump, not a copy of the body. The clone is still needed:
        // `call_user` takes `&mut self` and the borrow of `funcs` cannot be
        // held across it.
        self.returning = None;
        self.call_user(name, &f, args)
    }

    /// Whether a function of this name is defined.
    pub fn has_fn(&self, name: &str) -> bool {
        self.funcs.contains_key(name)
    }

    /// Steps taken since the last `run`. What a cost judge measures.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// A note one builtin leaves for another.
    ///
    /// `tcp_connect` answers 1 or 0 because a refused port is a result and not
    /// an exception, and *why* it failed still has to reach `tcp_error`. A
    /// static would work and would let two programs read each other's failure;
    /// on the interpreter it dies with the interpreter, which for an
    /// application is every repaint.
    ///
    /// Not reachable from Aiksi. There is no `note()` builtin, deliberately:
    /// this is plumbing between arms, and a program that could write here
    /// could forge the reason its own last call failed.
    pub fn set_note(&mut self, key: &str, value: &str) {
        self.notes.insert(String::from(key), String::from(value));
    }

    pub fn note(&self, key: &str) -> String {
        self.notes.get(key).cloned().unwrap_or_default()
    }

    /// `may_write` for the kernel arms, which live in another module and must
    /// not each grow their own idea of where the jail is.
    pub fn may_write_pub(&self, path: &str) -> bool {
        self.may_write(path)
    }

    fn may_write(&self, path: &str) -> bool {
        let Some(jail) = &self.jail else {
            return true;
        };
        let full = crate::sysbox::resolve_path(path);
        // The subtree, and not merely the prefix: `/app/todo-evil` must not
        // pass a jail of `/app/todo`.
        full == *jail || full.starts_with(&alloc::format!("{}/", jail))
    }

    /// The global scope, which is what `vars` at the prompt means: a function's
    /// locals exist only while it is running and there is nothing to show.
    pub fn var_count(&self) -> usize {
        self.scopes[0].len()
    }

    pub fn vars(&self) -> impl Iterator<Item = (&String, &Value)> {
        self.scopes[0].iter()
    }

    pub fn fn_names(&self) -> impl Iterator<Item = &String> {
        self.funcs.keys()
    }

    /// This frame, then the global one. Never a caller's frame.
    ///
    /// **This walked the whole stack, and that was a live bug.** `call_user`
    /// pushes the callee's frame on top of the caller's without popping it, so
    /// `scopes` is the entire call chain -- and walking it innermost-first let
    /// a callee reach into whichever caller up the live stack happened to use
    /// the same name. Measured, at the prompt:
    ///
    /// ```text
    ///     fn inner() { return x }
    ///     fn outer() { x = 42 return inner() }
    ///     outer()                                  -> 42
    /// ```
    ///
    /// `inner` has no `x`. It read one belonging to a function that called it.
    /// Nothing declared that relationship and nothing could see it: the name a
    /// function resolves to depended on the dynamic call chain above it, which
    /// is not a property any reader of `inner` could work out.
    ///
    /// Two levels is what the doc on `assign` always claimed and what
    /// `selftest` always checked -- a parameter shadowing a global, a function
    /// updating a global. Neither test ever asserted reach-through, so the
    /// behaviour was incidental rather than intended.
    ///
    /// Blocks do not push scopes -- only `call_user` does -- so "this frame"
    /// and "the global frame" are the whole of it.
    fn lookup(&self, name: &str) -> Option<&Value> {
        let last = self.scopes.len().checked_sub(1)?;
        if let Some(v) = self.scopes[last].get(name) {
            return Some(v);
        }
        if last == 0 {
            return None;
        }
        self.scopes[0].get(name)
    }

    /// Bind a name where it already lives, or in this frame if it is new. So a
    /// function updating a global updates the global, and one introducing a
    /// name keeps it to itself.
    ///
    /// The second half of that sentence was false until the scope walk was
    /// narrowed. A callee assigning `y` found a caller's `y` first and wrote
    /// through to it:
    ///
    /// ```text
    ///     fn poke() { y = 99 return 0 }
    ///     fn host() { y = 1 poke() return y }
    ///     host()                                   -> 99
    /// ```
    fn assign(&mut self, name: &str, v: Value) {
        let Some(last) = self.scopes.len().checked_sub(1) else { return };
        if let Some(slot) = self.scopes[last].get_mut(name) {
            *slot = v;
            return;
        }
        if last != 0 {
            if let Some(slot) = self.scopes[0].get_mut(name) {
                *slot = v;
                return;
            }
        }
        self.scopes[last].insert(String::from(name), v);
    }

    /// Run a block, stopping early if something inside it returned.
    fn body(&mut self, stmts: &[Stmt]) -> Result<(), String> {
        for st in stmts {
            self.stmt(st)?;
            if self.returning.is_some() {
                break;
            }
        }
        Ok(())
    }

    fn call_user(&mut self, name: &str, f: &Func, args: &[Value]) -> Result<Value, String> {
        if args.len() != f.params.len() {
            return Err(format!(
                "expected {} argument(s), got {}",
                f.params.len(),
                args.len()
            ));
        }
        if self.depth >= MAX_DEPTH {
            return Err("call nesting too deep (runaway recursion?)".to_string());
        }
        let mut frame = BTreeMap::new();
        for ((p, ty), a) in f.params.iter().zip(args.iter()) {
            // Checked on the way in, where the caller's mistake is, rather
            // than wherever the value is first used. A string passed where an
            // int belongs otherwise reaches `int()`, which answers 0 by design,
            // and the program computes a wrong number four calls later with
            // nothing having failed.
            if !a.fits(ty) {
                return Err(format!(
                    "{} wants {} for '{}', got {}",
                    name,
                    ty.name(),
                    p,
                    a.type_name()
                ));
            }
            frame.insert(p.clone(), a.clone());
        }
        self.scopes.push(frame);
        self.depth += 1;
        let r = self.body(&f.body);
        self.scopes.pop();
        self.depth -= 1;
        r?;
        // A function that falls off its end yields nothing, which is what a
        // procedure called for its effect should say.
        let out = self.returning.take().unwrap_or(Value::Nil);
        if !out.fits(&f.ret) {
            return Err(format!(
                "{} returns {}, got {}",
                name,
                f.ret.name(),
                out.type_name()
            ));
        }
        Ok(out)
    }

    /// Where a sandboxed program may import from, besides its own subtree.
    ///
    /// One shared, operator-curated directory. A stored application's
    /// dependencies are then either its own files or something visible in one
    /// place, which is what makes "what does this program use" a question with
    /// an answer.
    const LIB: &'static str = "/lib";

    /// Evaluate another program into this interpreter.
    ///
    /// Textual inclusion, not a module system. There is nothing to qualify
    /// against -- no namespaces, no exports -- and inventing a prefix would
    /// mean inventing a spelling for it and then explaining it. What `use`
    /// buys is that a function written once can be called from a second
    /// program, which is the whole of what the applications here need.
    ///
    /// **The imported program runs with the importer's capabilities**, and
    /// that is the security property. A sandboxed application importing a file
    /// from `/lib` does not gain what `/lib` could do if the operator ran it;
    /// caps live on the interpreter, and there is one interpreter. So an
    /// import can never be an escalation, and the jail below is about
    /// legibility rather than safety: it keeps a stored program's dependencies
    /// somewhere a person can find them.
    pub fn import(&mut self, path: &str) -> Result<(), String> {
        // The extension is optional at the call site, because `use "/lib/text"`
        // reads better than the alternative and there is only one kind of file
        // this could mean.
        let with_ext = if path.contains('.') {
            String::from(path)
        } else {
            format!("{}.ai&xi", path)
        };
        let full = crate::sysbox::resolve_path(&with_ext);
        if self.caps == Caps::Sandbox {
            let own = self.jail.clone().unwrap_or_default();
            let inside = |root: &str| {
                !root.is_empty() && (full == root || full.starts_with(&format!("{}/", root)))
            };
            if !inside(Self::LIB) && !inside(&own) {
                return Err(format!(
                    "a stored program may only use its own files or {}, not {}",
                    Self::LIB,
                    full
                ));
            }
        }
        // Already here. Not an error: two programs importing the same helper
        // is the normal case, and a cycle terminates because the second visit
        // finds this rather than recursing.
        if self.imported.contains(&full) {
            return Ok(());
        }
        let Some(bytes) = crate::sysbox::read_blob(&full) else {
            return Err(format!("use: no such program '{}'", full));
        };
        // Marked before evaluating, not after, or a file that imports itself
        // recurses until the stack runs out -- and running out of stack in
        // ring 0 with no guard page is a triple fault, not an error message.
        self.imported.insert(full.clone());
        let src = String::from_utf8_lossy(&bytes).into_owned();
        let toks = super::lex::lex(&src).map_err(|e| format!("{}: {}", full, e))?;
        let prog = super::parse::parse(toks).map_err(|e| format!("{}: {}", full, e))?;
        // Through `stmt` rather than `run`, because `run` resets the step
        // counter -- an import would otherwise be a way to buy an unbounded
        // budget one file at a time.
        for st in &prog {
            self.stmt(st).map_err(|e| format!("{}: {}", full, e))?;
        }
        // An imported file's trailing `return` is not this program's.
        self.returning = None;
        Ok(())
    }

    /// Run a program, returning the value of the last expression statement.
    /// Whether a program's top level does nothing but declare.
    ///
    /// This is the whole condition under which preparing is equivalent to
    /// arming. A declaration's only effect is to register itself, so running
    /// such a top level once and copying the result is indistinguishable from
    /// running it again. Every other statement can read the world or change
    /// it -- an assignment computes, a call can ask the clock, `use` lexes and
    /// executes a whole other file -- and freezing any of those would turn a
    /// value re-computed per run into a value fixed at prepare time. That is a
    /// semantic change wearing an optimisation's clothes, so those programs
    /// keep arming.
    pub fn is_declarative(prog: &[Stmt]) -> bool {
        prog.iter().all(|s| matches!(s, Stmt::Fn(..) | Stmt::Rec(..)))
    }

    /// Run a declarative top level once, for a program that will be run many
    /// times over with fresh state.
    ///
    /// Refuses anything `is_declarative` refuses, rather than preparing what
    /// it can and arming the rest: a program half-prepared would have its
    /// declarations registered before its statements ran, which is not the
    /// order it was written in.
    pub fn prepare(prog: &[Stmt]) -> Result<Prepared, String> {
        if !Self::is_declarative(prog) {
            return Err(String::from("top level is more than declarations"));
        }
        let mut it = Self::new();
        it.run(prog)?;
        Ok(Prepared { funcs: it.funcs, recs: it.recs, steps: it.steps })
    }

    /// Seed from prepared declarations instead of running the top level.
    ///
    /// The step count comes with them, and that is not bookkeeping. `steps` is
    /// what a caller is charged and what the budget stops, so a prepared run
    /// that skipped the top level's ticks would answer the same and cost less
    /// -- two paths through one program disagreeing about a number the judges
    /// read. Carrying it makes the two bit-identical in the only things
    /// anything can observe: the value, the cost, and the error.
    pub fn adopt(&mut self, p: &Prepared) {
        self.funcs = p.funcs.clone();
        self.recs = p.recs.clone();
        self.steps = self.steps.saturating_add(p.steps);
    }

    pub fn run(&mut self, prog: &[Stmt]) -> Result<Value, String> {
        self.steps = 0;
        // A `return` typed at the prompt has nothing to return from. Cleared
        // here so it cannot sit armed and silently truncate the next block
        // that runs.
        self.returning = None;
        let mut last = Value::Nil;
        for s in prog {
            last = self.stmt(s)?;
        }
        Ok(last)
    }

    fn tick(&mut self) -> Result<(), String> {
        self.steps += 1;
        if self.steps > self.budget {
            return Err("execution budget exceeded (infinite loop?)".to_string());
        }
        Ok(())
    }

    fn stmt(&mut self, s: &Stmt) -> Result<Value, String> {
        self.tick()?;
        match s {
            Stmt::Expr(e) => self.expr(e),
            Stmt::Fn(name, params, ret, body) => {
                // The one deep copy that stays. `run` walks a borrowed
                // `Program`, so storing the body without copying it would put
                // a lifetime on `Interp` and thread it through every caller in
                // the kernel. Paid once per declaration instead of once per
                // call, which is where it was actually being paid.
                self.funcs.insert(
                    name.clone(),
                    alloc::rc::Rc::new(Func {
                        params: params.clone(),
                        ret: ret.clone(),
                        body: body.clone(),
                    }),
                );
                Ok(Value::Nil)
            }
            Stmt::Rec(name, fields) => {
                // A record may not take a builtin's name. A user *function*
                // may, deliberately -- a program that defines `rect` means its
                // own -- but that shadowing is per-call and reversible by
                // reading the program. A record additionally installs a
                // constructor, and `rec len { ... }` would leave `len(x)`
                // meaning something that depends on which of the two the
                // reader remembered.
                if super::eval::is_builtin(name) {
                    return Err(format!("'{}' is a builtin and cannot be a record", name));
                }
                // Nor may a program redeclare one the kernel hands back. A
                // builtin would then return a `Device` with the kernel's
                // fields while every annotation in the program checked against
                // a different shape of the same name. This is also what keeps
                // the two record tables disjoint now that they are two --
                // without it, `fields_of` would resolve a shadowed name by
                // search order and nothing would report that it had.
                if KERNEL_RECS.iter().any(|(n, _)| *n == name) {
                    return Err(format!("'{}' is a record the kernel returns", name));
                }
                self.recs.insert(name.clone(), fields.clone());
                Ok(Value::Nil)
            }
            Stmt::Use(path) => {
                self.import(path)?;
                Ok(Value::Nil)
            }
            Stmt::Return(e) => {
                let v = match e {
                    Some(x) => self.expr(x)?,
                    None => Value::Nil,
                };
                self.returning = Some(v);
                Ok(Value::Nil)
            }
            Stmt::If(cond, then, otherwise) => {
                if self.expr(cond)?.truthy() {
                    self.body(then)?;
                } else if let Some(els) = otherwise {
                    self.body(els)?;
                }
                Ok(Value::Nil)
            }
            Stmt::While(cond, body) => {
                while self.expr(cond)?.truthy() {
                    self.tick()?;
                    self.body(body)?;
                    // A `return` inside the loop leaves the function, not
                    // just this iteration. Running the body through `body`
                    // stops the iteration; this stops the loop. Missing this
                    // is silent: the loop simply runs to completion and
                    // whatever it returned is overwritten by whatever comes
                    // after it, so the function answers the wrong thing
                    // rather than failing.
                    if self.returning.is_some() {
                        break;
                    }
                }
                Ok(Value::Nil)
            }
        }
    }

    fn expr(&mut self, e: &Expr) -> Result<Value, String> {
        self.tick()?;
        match e {
            Expr::Int(v) => Ok(Value::Int(*v)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Var(name) => self
                .lookup(name)
                .cloned()
                .ok_or_else(|| format!("undefined variable '{}'", name)),
            Expr::Assign(name, rhs) => {
                let v = self.expr(rhs)?;
                self.assign(name, v.clone());
                Ok(v)
            }
            Expr::Unary(op, inner) => {
                let v = self.expr(inner)?;
                match op {
                    UnOp::Neg => Ok(Value::Int(v.as_int()?.wrapping_neg())),
                    UnOp::Not => Ok(Value::Int(if v.truthy() { 0 } else { 1 })),
                    UnOp::BitNot => Ok(Value::Int(!v.as_int()?)),
                }
            }
            Expr::Bin(op, l, r) => self.binary(*op, l, r),
            Expr::Field(target, field) => {
                let v = self.expr(target)?;
                match &v {
                    Value::Rec(ty, fields) => fields
                        .iter()
                        .find(|(n, _)| n == field)
                        .map(|(_, v)| v.clone())
                        .ok_or_else(|| format!("{} has no field '{}'", ty, field)),
                    // Not Nil. Reading a field off a number is a mistake in the
                    // program, and answering nothing would let it run on and
                    // fail somewhere that has nothing to do with the cause.
                    other => Err(format!(
                        "'.{}' wants a record, got {}",
                        field,
                        other.type_name()
                    )),
                }
            }
            Expr::SetField(target, field, value) => {
                let v = self.expr(value)?;
                let updated = {
                    let base = self.expr(target)?;
                    let Value::Rec(ty, fields) = base else {
                        return Err(format!(
                            "'.{} =' wants a record, got {}",
                            field,
                            base.type_name()
                        ));
                    };
                    // The declared type of the field still holds. A record
                    // whose fields could be replaced by anything after
                    // construction would make the annotation a comment.
                    if let Some(want) = self
                        .fields_of(&ty)
                        .and_then(|fs| fs.find(field).cloned())
                    {
                        if !v.fits(&want) {
                            return Err(format!(
                                "{}.{} wants {}, got {}",
                                ty,
                                field,
                                want.name(),
                                v.type_name()
                            ));
                        }
                    }
                    let mut out = fields.clone();
                    match out.iter_mut().find(|(n, _)| n == field) {
                        Some(slot) => slot.1 = v.clone(),
                        None => return Err(format!("{} has no field '{}'", ty, field)),
                    }
                    Value::Rec(ty, out)
                };
                // Records are values, so there is nothing shared to mutate:
                // the new record has to be put back where the old one was.
                // Only a plain variable can be assigned back to, which means
                // `f(x).y = 1` is refused rather than silently discarded.
                match &**target {
                    Expr::Var(name) => {
                        self.assign(name, updated.clone());
                        Ok(updated)
                    }
                    Expr::Field(..) => Err(
                        "assigning through a nested field is not supported -- \
                         take the inner record into a variable first"
                            .to_string(),
                    ),
                    _ => Err("left of '=' must be a variable or a field".to_string()),
                }
            }
            Expr::Call(name, args) => {
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.expr(a)?);
                }
                // User functions shadow builtins deliberately. A program that
                // defines `rect` means its own `rect`, and finding out that a
                // name was reserved is worse than losing access to a builtin
                // the program chose to replace.
                if let Some(f) = self.funcs.get(name).cloned() {
                    return self.call_user(name, &f, &vals);
                }
                // A record's name is its constructor. Between user functions
                // and builtins: a program may still define a function with the
                // record's name and mean it, and no builtin can be shadowed
                // because `rec` refuses a builtin's name outright.
                if let Some(fields) = self.fields_of(name) {
                    if vals.len() != fields.len() {
                        return Err(format!(
                            "{} takes {} field(s), got {}",
                            name,
                            fields.len(),
                            vals.len()
                        ));
                    }
                    let mut out = Vec::with_capacity(fields.len());
                    for (i, v) in vals.iter().enumerate() {
                        let (fname, ty) = fields.at(i);
                        if !v.fits(ty) {
                            return Err(format!(
                                "{}.{} wants {}, got {}",
                                name,
                                fname,
                                ty.name(),
                                v.type_name()
                            ));
                        }
                        out.push((String::from(fname), v.clone()));
                    }
                    return Ok(Value::Rec(name.clone(), out));
                }
                self.builtin(name, &vals)
            }
        }
    }

    fn binary(&mut self, op: BinOp, l: &Expr, r: &Expr) -> Result<Value, String> {
        // Short-circuit before evaluating the right side.
        if op == BinOp::LogAnd {
            let a = self.expr(l)?;
            if !a.truthy() {
                return Ok(Value::Int(0));
            }
            return Ok(Value::Int(if self.expr(r)?.truthy() { 1 } else { 0 }));
        }
        if op == BinOp::LogOr {
            let a = self.expr(l)?;
            if a.truthy() {
                return Ok(Value::Int(1));
            }
            return Ok(Value::Int(if self.expr(r)?.truthy() { 1 } else { 0 }));
        }

        let a = self.expr(l)?;
        let b = self.expr(r)?;

        // String concatenation is the one non-numeric case.
        if op == BinOp::Add {
            if let (Value::Str(x), y) = (&a, &b) {
                return Ok(Value::Str(format!("{}{}", x, y.render())));
            }
            if let (x, Value::Str(y)) = (&a, &b) {
                return Ok(Value::Str(format!("{}{}", x.render(), y)));
            }
        }
        if op == BinOp::Eq {
            return Ok(Value::Int(if a == b { 1 } else { 0 }));
        }
        if op == BinOp::Ne {
            return Ok(Value::Int(if a != b { 1 } else { 0 }));
        }

        let x = a.as_int()?;
        let y = b.as_int()?;
        let v = match op {
            BinOp::Add => x.wrapping_add(y),
            BinOp::Sub => x.wrapping_sub(y),
            BinOp::Mul => x.wrapping_mul(y),
            BinOp::Div => {
                if y == 0 {
                    return Err("division by zero".to_string());
                }
                x.wrapping_div(y)
            }
            BinOp::Rem => {
                if y == 0 {
                    return Err("remainder by zero".to_string());
                }
                x.wrapping_rem(y)
            }
            BinOp::Lt => (x < y) as i64,
            BinOp::Le => (x <= y) as i64,
            BinOp::Gt => (x > y) as i64,
            BinOp::Ge => (x >= y) as i64,
            BinOp::And => x & y,
            BinOp::Or => x | y,
            BinOp::Xor => x ^ y,
            // Shifts are masked to 63 so a silly count cannot panic.
            BinOp::Shl => x.wrapping_shl((y & 63) as u32),
            BinOp::Shr => x.wrapping_shr((y & 63) as u32),
            BinOp::Eq | BinOp::Ne | BinOp::LogAnd | BinOp::LogOr => unreachable!(),
        };
        Ok(Value::Int(v))
    }

    fn arg<'v>(args: &'v [Value], n: usize) -> Result<&'v Value, String> {
        args.get(n).ok_or_else(|| format!("missing argument {}", n + 1))
    }

    fn builtin(&mut self, name: &str, args: &[Value]) -> Result<Value, String> {
        // One gate, before anything is dispatched, and it opens only for a
        // builtin the table names. An unknown name cannot fall through to the
        // match: the match is unreachable without a row, so adding an arm and
        // forgetting the row produces dead code rather than an ungated builtin.
        let Some(touch) = touch_of(name) else {
            return Err(format!("no builtin called '{}'", name));
        };
        let (lo, hi) = BUILTINS
            .iter()
            .find(|(n, ..)| *n == name)
            .map(|(_, _, lo, hi)| (*lo, *hi))
            .unwrap_or((0, usize::MAX));
        if args.len() < lo || args.len() > hi {
            return Err(if lo == hi {
                format!("{} takes {} argument(s), got {}", name, lo, args.len())
            } else if hi == usize::MAX {
                format!("{} takes at least {}, got {}", name, lo, args.len())
            } else {
                format!("{} takes {} to {} arguments, got {}", name, lo, hi, args.len())
            });
        }
        if self.caps == Caps::Sandbox {
            if !touch.sandboxable() {
                // Name the command that actually applies. The jail root says
                // what kind of program this is: apps live under /app and are
                // approved with `app trust`, skills live under /ai/tools and
                // are approved with `skill trust`. Suggesting the wrong one is
                // worse than suggesting none -- it sends the operator to a
                // command that will not find their program.
                let jail = self.jail.as_deref().unwrap_or("");
                let leaf = jail.rsplit('/').next().unwrap_or("<program>");
                let how = if jail.starts_with("/ai/tools") {
                    "skill trust <hash>"
                } else {
                    "app trust"
                };
                return Err(format!(
                    "'{}' {} and a sandboxed program may not -- '{}' for {} if that is what you want",
                    name,
                    touch.why(),
                    how,
                    leaf
                ));
            }
            if name == "write" {
                // Checked here rather than inside the arm, so the arm cannot
                // be rewritten later without noticing the check.
                let path = args.first().map(|v| v.render()).unwrap_or_default();
                if !self.may_write(&path) {
                    return Err(format!("a stored program may not write to {}", path));
                }
            }
            if name == "applet" {
                let line = args.first().map(|v| v.render()).unwrap_or_default();
                let cmd = line.split(' ').next().unwrap_or("");
                match crate::sysbox::applet_mutates(cmd) {
                    None => return Err(format!("no applet '{}'", cmd)),
                    Some(true) => {
                        return Err(format!(
                            "'{}' changes the system and a stored program may not call it",
                            cmd
                        ))
                    }
                    Some(false) => {}
                }
            }
        }

        fn need(args: &[Value], n: usize, name: &str) -> Result<(), String> {
            if args.len() != n {
                Err(format!("{} takes {} argument(s), got {}", name, n, args.len()))
            } else {
                Ok(())
            }
        }
        fn int(args: &[Value], i: usize) -> Result<i64, String> {
            args[i].as_int()
        }
        fn colour(v: i64) -> crate::gfx::Color {
            PALETTE[(v & 0x0F) as usize]
        }

        match name {
            // --- lists ---------------------------------------------------
            //
            // Values, not references. `push` returns a new list rather than
            // changing one in place, so two names can never disagree about
            // what a list contains and nothing has to explain aliasing to
            // whoever -- or whatever -- is writing the program.
            // Where this program is allowed to keep things.
            //
            // A stored program cannot hardcode its own path: the same files
            // live at `/draft/<name>` while being written and `/app/<name>`
            // once adopted, and a literal path would be outside the jail on
            // one side of that move. Asking makes a program location
            // independent, and the jail is the authority on the answer rather
            // than a convention the program has to be told.
            "here" => Ok(Value::Str(self.jail.clone().unwrap_or_default())),
            // Text to number.
            //
            // `read` answers with what is in the file, which is text, and `+`
            // on text concatenates. A program keeping a count in a file and
            // adding one to it gets "01" and then "011", with nothing failing
            // anywhere -- the first skeleton written here did exactly that.
            // Anything unparseable is zero rather than an error: a counter
            // whose file has been emptied should start again, not refuse to
            // draw.
            "int" => {
                let t = Self::arg(args, 0)?.render();
                let t = t.trim();
                let (neg, digits) = match t.strip_prefix('-') {
                    Some(rest) => (true, rest),
                    None => (false, t),
                };
                let mut n: i64 = 0;
                let mut any = false;
                for c in digits.chars() {
                    let Some(d) = c.to_digit(10) else { break };
                    n = n.saturating_mul(10).saturating_add(d as i64);
                    any = true;
                }
                Ok(Value::Int(if any && neg { -n } else if any { n } else { 0 }))
            }
            "list" => Ok(Value::List(args.to_vec())),
            "len" => match args.first() {
                Some(Value::List(v)) => Ok(Value::Int(v.len() as i64)),
                Some(Value::Str(t)) => Ok(Value::Int(t.chars().count() as i64)),
                _ => Err("len wants a list or a string".to_string()),
            },
            "get" => {
                let (l, i) = (Self::arg(args, 0)?, Self::arg(args, 1)?.as_int()?);
                match l {
                    Value::List(v) => {
                        // Out of range is nothing, not a fault. A program
                        // walking a list it did not write should be able to
                        // ask past the end without dying.
                        Ok(v.get(i as usize).cloned().unwrap_or(Value::Nil))
                    }
                    Value::Str(t) => Ok(t
                        .chars()
                        .nth(i as usize)
                        .map(|c| Value::Str(c.to_string()))
                        .unwrap_or(Value::Nil)),
                    _ => Err("get wants a list or a string".to_string()),
                }
            }
            "push" => {
                let l = Self::arg(args, 0)?;
                let v = Self::arg(args, 1)?;
                match l {
                    Value::List(items) => {
                        let mut out = items.clone();
                        out.push(v.clone());
                        Ok(Value::List(out))
                    }
                    _ => Err("push wants a list".to_string()),
                }
            }
            "set" => {
                let l = Self::arg(args, 0)?;
                let i = Self::arg(args, 1)?.as_int()?;
                let v = Self::arg(args, 2)?;
                match l {
                    Value::List(items) => {
                        let mut out = items.clone();
                        let i = i as usize;
                        if i >= out.len() {
                            return Err("set past the end of the list".to_string());
                        }
                        out[i] = v.clone();
                        Ok(Value::List(out))
                    }
                    _ => Err("set wants a list".to_string()),
                }
            }
            "print" => {
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        kprint!(" ");
                    }
                    kprint!("{}", a.render());
                }
                Ok(Value::Nil)
            }
            "println" => {
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        kprint!(" ");
                    }
                    kprint!("{}", a.render());
                }
                kprintln!();
                Ok(Value::Nil)
            }
            "hex" => {
                need(args, 1, "hex")?;
                Ok(Value::Str(format!("{:#x}", int(args, 0)?)))
            }
            "cls" => {
                console::with(|c| c.clear());
                Ok(Value::Nil)
            }
            "color" => {
                need(args, 1, "color")?;
                console::set_color((int(args, 0)? & 0x0F) as u8);
                Ok(Value::Nil)
            }
            "ticks" => Ok(Value::Int(crate::dev::lapic::ticks() as i64)),
            "hz" => Ok(Value::Int(crate::dev::lapic::timer_hz() as i64)),
            "tasks" => Ok(Value::Int(crate::task::count() as i64)),
            "heap" => {
                let (used, total) = crate::mem::heap::HEAP.stats();
                kprintln!("  {} B used of {} B", used, total);
                Ok(Value::Int(used as i64))
            }

            // --- namespace ---
            //
            // These are what make a program in /ai/tools a skill rather than
            // a calculator: the ability to look at the namespace and change
            // it. The exposure is exactly the `cat`/`ls`/`write` applets',
            // reached through one more indirection; `run` is classified as
            // mutating, so the read-only grammar never reaches any of this.
            "read" => {
                need(args, 1, "read")?;
                let path = args[0].render();
                match crate::sysbox::read_blob(&path) {
                    Some(bytes) => Ok(Value::Str(String::from_utf8_lossy(&bytes).into_owned())),
                    None => Err(format!("read: no such file '{}'", path)),
                }
            }
            "exists" => {
                need(args, 1, "exists")?;
                let path = args[0].render();
                let yes =
                    crate::sysbox::is_dir(&path) || crate::sysbox::read_blob(&path).is_some();
                Ok(Value::Int(yes as i64))
            }
            // A list, not newline-joined text. It answered text, so every
            // caller began by splitting it back apart -- the seeded
            // `/ai/tools/count` tool hand-wrote a character-by-character line
            // counter to do exactly that, which is a fair summary of what the
            // old shape cost. `applet("ls ...")` still answers the formatted
            // listing for anything that wants to show one.
            "ls" => {
                need(args, 1, "ls")?;
                let path = args[0].render();
                if !crate::sysbox::is_dir(&path) {
                    return Err(format!("ls: '{}' is not a directory", path));
                }
                Ok(Value::List(
                    crate::sysbox::children(&path).into_iter().map(Value::Str).collect(),
                ))
            }
            "write" => {
                need(args, 2, "write")?;
                let path = args[0].render();
                let text = args[1].render();
                if crate::sysbox::write_text(&path, &text) {
                    Ok(Value::Int(text.len() as i64))
                } else {
                    Err(format!("write: could not write '{}'", path))
                }
            }
            "applet" => {
                // The program calls the OS. One string in -- "name args" --
                // the applet's captured output out as a string. This is what
                // turns a skill from a calculation into a script: a program
                // that can ls, cat, write and snap its way through the
                // namespace, compose applets, and hand the composed result
                // back to whoever ran it. Trust travels through `run`, which
                // is classified mutating regardless of program text, so the
                // read-only grammar never reaches this.
                need(args, 1, "applet")?;
                let line = args[0].render();
                let (cmd, rest) = match line.split_once(' ') {
                    Some((c, r)) => (c, r),
                    None => (line.as_str(), ""),
                };
                if !crate::sysbox::is_ready() {
                    return Err("applet: namespace not initialised".into());
                }
                if !crate::sysbox::is_applet(cmd) {
                    return Err(format!("applet: '{}' is not an applet", cmd));
                }
                console::begin_capture();
                let ran = crate::sysbox::dispatch(cmd, rest);
                let out = console::end_capture().unwrap_or_default();
                if !ran {
                    return Err(format!("applet: '{}' did not run", cmd));
                }
                Ok(Value::Str(out.trim_end().to_string()))
            }

            // --- graphics ---
            "width" => Ok(Value::Int(gfx::primary().map(|f| f.width()).unwrap_or(0) as i64)),
            "height" => Ok(Value::Int(gfx::primary().map(|f| f.height()).unwrap_or(0) as i64)),
            "pixel" => {
                need(args, 3, "pixel")?;
                if let Some(fb) = gfx::primary() {
                    let raw = fb.encode(colour(int(args, 2)?));
                    fb.put(int(args, 0)? as u32, int(args, 1)? as u32, raw);
                }
                Ok(Value::Nil)
            }
            "rect" => {
                need(args, 5, "rect")?;
                if let Some(fb) = gfx::primary() {
                    fb.rect(
                        int(args, 0)? as u32,
                        int(args, 1)? as u32,
                        int(args, 2)? as u32,
                        int(args, 3)? as u32,
                        colour(int(args, 4)?),
                    );
                }
                Ok(Value::Nil)
            }
            "text" => {
                need(args, 4, "text")?;
                if let Some(fb) = gfx::primary() {
                    fb.draw_text(
                        int(args, 0)? as u32,
                        int(args, 1)? as u32,
                        &args[2].render(),
                        colour(int(args, 3)?),
                        PALETTE[0],
                        2,
                    );
                }
                Ok(Value::Nil)
            }

            // --- raw memory. Ring 0 means these are simply loads and stores. ---
            "peek8" => {
                need(args, 1, "peek8")?;
                let a = int(args, 0)? as u64 as *const u8;
                Ok(Value::Int(unsafe { core::ptr::read_volatile(a) } as i64))
            }
            "peek16" => {
                need(args, 1, "peek16")?;
                let a = int(args, 0)? as u64 as *const u16;
                Ok(Value::Int(unsafe { core::ptr::read_volatile(a) } as i64))
            }
            "peek32" => {
                need(args, 1, "peek32")?;
                let a = int(args, 0)? as u64 as *const u32;
                Ok(Value::Int(unsafe { core::ptr::read_volatile(a) } as i64))
            }
            "peek64" => {
                need(args, 1, "peek64")?;
                let a = int(args, 0)? as u64 as *const u64;
                Ok(Value::Int(unsafe { core::ptr::read_volatile(a) } as i64))
            }
            "poke8" => {
                need(args, 2, "poke8")?;
                let a = int(args, 0)? as u64 as *mut u8;
                unsafe { core::ptr::write_volatile(a, int(args, 1)? as u8) };
                Ok(Value::Nil)
            }
            "poke32" => {
                need(args, 2, "poke32")?;
                let a = int(args, 0)? as u64 as *mut u32;
                unsafe { core::ptr::write_volatile(a, int(args, 1)? as u32) };
                Ok(Value::Nil)
            }
            "poke64" => {
                need(args, 2, "poke64")?;
                let a = int(args, 0)? as u64 as *mut u64;
                unsafe { core::ptr::write_volatile(a, int(args, 1)? as u64) };
                Ok(Value::Nil)
            }

            // --- port I/O. The EC lives behind these. ---
            "inb" => {
                need(args, 1, "inb")?;
                Ok(Value::Int(unsafe {
                    crate::cpu::port::inb(int(args, 0)? as u16)
                } as i64))
            }
            "outb" => {
                need(args, 2, "outb")?;
                unsafe { crate::cpu::port::outb(int(args, 0)? as u16, int(args, 1)? as u8) };
                Ok(Value::Nil)
            }
            "inl" => {
                need(args, 1, "inl")?;
                Ok(Value::Int(unsafe {
                    crate::cpu::port::inl(int(args, 0)? as u16)
                } as i64))
            }
            "outl" => {
                need(args, 2, "outl")?;
                unsafe { crate::cpu::port::outl(int(args, 0)? as u16, int(args, 1)? as u32) };
                Ok(Value::Nil)
            }

            // Everything that reaches a kernel subsystem. The gate and the
            // arity check have already run, so this cannot be a way around
            // either: a name only gets here by being in the table.
            other => super::kernel::call(self, other, args),
        }
    }
}

// The list the shell offers used to live here as a second array of names,
// hand-kept beside the match. It is gone: `BUILTINS` above is the one list,
// and `words` reads it. A name that appears in one place cannot disagree with
// itself.
