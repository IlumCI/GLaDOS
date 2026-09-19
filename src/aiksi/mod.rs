//! The language.
//!
//! Source -> tokens -> AST -> evaluation. In TempleOS the shell *was* the
//! compiler: what you typed at the prompt was compiled to machine code and
//! executed, with no separation between "using the system" and "programming
//! it". This is the first half of that. The second half replaces `eval` with a
//! single-pass code generator emitting x86-64 into the heap; the front end
//! here does not change when that happens.

pub mod differ;
pub mod eval;
pub mod jit;
pub mod kernel;
pub mod lex;
pub mod parse;

pub use eval::{Interp, Value};

use alloc::string::String;

/// The standard library, compiled in and seeded into `/lib` at boot.
///
/// A real file rather than a byte-string literal, because it is *source*: it
/// gets read, diffed and edited like the Rust beside it, and escaping a
/// hundred lines of Aiksi into `b"...\n\"` would make it none of those. The
/// seeded tools in `sysbox` predate this and are one line each, which is why
/// they get away with it.
///
/// Compiled in for the reason the routing corpus is: `/lib` is where a stored
/// program's dependencies are allowed to live, so a machine that has never
/// mounted a store still has to have one.
pub const LIB_PROB: &str = include_str!("lib/prob.ai&xi");

/// Plane geometry over exact coordinates, seeded beside it.
pub const LIB_GEOM: &str = include_str!("lib/geom.ai&xi");

/// Linear algebra: Gaussian elimination that cannot lie about a pivot.
pub const LIB_MAT: &str = include_str!("lib/mat.ai&xi");

/// Number theory, and modular arithmetic with no 128-bit divide anywhere.
pub const LIB_NUM: &str = include_str!("lib/num.ai&xi");

/// Polynomials, including the two operations whole numbers cannot express.
pub const LIB_POLY: &str = include_str!("lib/poly.ai&xi");

/// Physics in quantities that carry their units.
pub const LIB_PHYS: &str = include_str!("lib/phys.ai&xi");

/// Chemistry: formulas parsed, molar masses summed exactly.
pub const LIB_CHEM: &str = include_str!("lib/chem.ai&xi");

/// Every library, as (path, source).
///
/// One list, so seeding `/lib` and checking what is in it cannot disagree
/// about what exists. Adding a library is adding a row here; forgetting to
/// seed one stops being a thing that can happen separately.
pub const LIBS: &[(&str, &str)] = &[
    ("/lib/prob.ai&xi", LIB_PROB),
    ("/lib/geom.ai&xi", LIB_GEOM),
    ("/lib/mat.ai&xi", LIB_MAT),
    ("/lib/num.ai&xi", LIB_NUM),
    ("/lib/poly.ai&xi", LIB_POLY),
    ("/lib/phys.ai&xi", LIB_PHYS),
    ("/lib/chem.ai&xi", LIB_CHEM),
];

/// Every `fn name(` a library source declares, in order.
///
/// Scanned rather than parsed, which is the bargain `fmt::outline` makes and
/// for the same reason: there is nothing here a parser would be more right
/// about, since a declaration is `fn`, a space, a name and an open paren at
/// the start of a line, and a comment cannot match because it starts with a
/// slash.
pub fn declared_names(src: &str) -> alloc::vec::Vec<&str> {
    let mut out = alloc::vec::Vec::new();
    for line in src.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("fn ") {
            if let Some(q) = rest.find('(') {
                out.push(rest[..q].trim());
            }
        }
    }
    out
}

/// Lex, parse and evaluate one line, returning the value of its last expression.
pub fn eval_line(interp: &mut Interp, src: &str) -> Result<Value, String> {
    let toks = lex::lex(src)?;
    let ast = parse::parse(toks)?;
    interp.run(&ast)
}

/// What a step actually costs, which nobody has ever measured.
///
/// Every step budget in this tree -- `STEP_BUDGET` 20,000,000, `SKILL_BUDGET`
/// 5,000,000, `DRAW_BUDGET` 200,000, `VOTE_BUDGET` 20,000 -- is a step count
/// chosen by comparison to another step count. They are a chain of "one order
/// below the previous" decisions with no wall-clock number anywhere underneath
/// them, so nobody has been able to say what any of them means in time. That
/// is the first thing a compiler would need to know, and it is worth knowing
/// whether or not one is ever written.
///
/// Best of nine, min and max, for the reason `video bench` is: under emulation
/// a single sample measures the host's scheduler rather than the guest.
///
/// The step count comes from the interpreter itself rather than from counting
/// nodes here. `tick` fires once per statement, once per expression node and
/// once more per loop iteration, and a second implementation of that rule in
/// the benchmark would be a second thing to keep in step -- and would quietly
/// mis-report the moment either drifted.
/// What the tree-walking interpreter costs.
///
/// Picoseconds for a step, because a step is well under a nanosecond once the
/// loop is divided out and integer nanoseconds would read as zero.
#[derive(Clone, Copy)]
pub struct Walk {
    pub mhz: u64,
    pub new_ns: (u64, u64),
    pub loop_ns: (u64, u64),
    pub steps: u64,
    pub step_ps: u64,
}

/// A struct rather than printed lines, so `bench report` and a person read one
/// measurement. `None` when the loop reported no steps, in which case nothing
/// derived from it means anything.
pub fn measure() -> Option<Walk> {
    let mhz = crate::time::tsc_mhz().max(1);
    const RUNS: usize = 9;

    // Cycles, not microseconds: the interesting quantities here are hundreds
    // of nanoseconds and dividing by MHz first would round them to zero.
    fn best(mut f: impl FnMut()) -> (u64, u64) {
        let (mut lo, mut hi) = (u64::MAX, 0u64);
        for _ in 0..RUNS {
            let t = crate::time::rdtsc();
            f();
            let d = crate::time::rdtsc().wrapping_sub(t);
            lo = lo.min(d);
            hi = hi.max(d);
        }
        (lo, hi)
    }
    let ns = |cycles: u64| -> u64 { cycles.saturating_mul(1000) / mhz };

    // Construction alone. `Core::vote` pays this on every routing decision.
    let (c_lo, c_hi) = best(|| {
        let it = eval::Interp::new();
        core::hint::black_box(&it);
    });

    // A tight loop, timed against the interpreter's own step count.
    let src = "i = 0 while (i < 20000) { i = i + 1 } i";
    let mut steps = 0u64;
    let (l_lo, l_hi) = best(|| {
        let mut it = eval::Interp::new();
        let _ = eval_line(&mut it, src);
        steps = it.steps();
    });
    if steps == 0 {
        return None;
    }
    Some(Walk {
        mhz,
        new_ns: (ns(c_lo), ns(c_hi)),
        loop_ns: (ns(l_lo), ns(l_hi)),
        steps,
        step_ps: ns(l_lo).saturating_mul(1000) / steps,
    })
}

pub fn bench() {
    use crate::kprintln;
    let Some(m) = measure() else {
        kprintln!("  the loop reported no steps -- nothing here would mean anything");
        return;
    };
    kprintln!("  tsc {} MHz, best of 9", m.mhz);
    kprintln!("  Interp::new()          {} ns  (max {})", m.new_ns.0, m.new_ns.1);
    kprintln!(
        "  20k-iteration loop     {} us  (max {}), {} steps",
        m.loop_ns.0 / 1000,
        m.loop_ns.1 / 1000,
        m.steps
    );
    kprintln!("  one step               {} ps", m.step_ps);

    // What every budget in the tree means in time, which is the number none of
    // them could be checked against before.
    kprintln!("  a budget spent in full, at that rate:");
    for (name, budget) in [
        ("VOTE_BUDGET  20k", 20_000u64),
        ("DRAW_BUDGET 200k", 200_000),
        ("SKILL_BUDGET  5M", 5_000_000),
        ("STEP_BUDGET  20M", 20_000_000),
    ] {
        let us = budget.saturating_mul(m.step_ps) / 1_000_000;
        if us >= 1000 {
            kprintln!("    {}   {} ms", name, us / 1000);
        } else {
            kprintln!("    {}   {} us", name, us);
        }
    }
}

/// The standard library, checked by importing it and running it.
///
/// Not by reading it. `/lib/prob` is Aiksi source compiled into the image, so
/// the only thing that establishes it works is the interpreter executing it
/// against answers known from outside -- C(5,3) is 10 whatever this machine
/// thinks, and five fair tosses landing three heads is 5/16.
///
/// This is also the first thing that exercises `use` on a path that is not the
/// caller's own, and the first consumer of exact fractions outside their own
/// claims.
pub fn lib_selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let mut ok = true;
    let mut check = |what: &str, pass: bool| {
        console::set_color(if pass { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if pass { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= pass;
    };

    fn lib(expr: &str) -> Option<Value> {
        let mut it = Interp::new();
        eval_line(&mut it, &alloc::format!("use \"/lib/prob\" {}", expr)).ok()
    }
    fn li(expr: &str, want: i64) -> bool {
        matches!(lib(expr), Some(Value::Int(v)) if v == want)
    }
    fn lt(expr: &str, want: &str) -> bool {
        matches!(lib(expr), Some(v) if v.render() == want)
    }

    check(
        "the library imports at all, from a path the caller does not own",
        lib("fact(5)").is_some(),
    );
    check("factorial, and a refusal past what fits", li("fact(5)", 120) && li("fact(21)", -1));

    // The point of the multiplicative forms. 50! is not a number this machine
    // holds, so neither of these is computable through a factorial at all --
    // they are the argument for not writing one.
    check(
        "permutations past the point a factorial overflows",
        li("perm(50, 3)", 117_600),
    );
    check(
        "C(50,25) exactly, which n!/(k!(n-k)!) could not reach",
        li("comb(50, 25)", 126_410_606_437_752),
    );
    check(
        "combinations at the edges, and outside them",
        li("comb(5, 3)", 10) && li("comb(5, 0)", 1) && li("comb(5, 6)", 0),
    );

    // The claim the exact numbers exist for. A float would answer 0.3125 and
    // then not compare equal to itself after a few more operations.
    check(
        "five fair tosses landing three heads is 5/16, exactly",
        lt("binom(5, 3, 1, 2)", "5/16"),
    );
    // Every outcome, summed, is one. Nothing but exact arithmetic makes that
    // true -- in floating point it is 0.9999999999999999 or 1.0000000000000002
    // depending on the order, and neither is 1.
    check(
        "the whole distribution sums to exactly 1",
        li("atleast(5, 0, 1, 2)", 1) && li("atleast(6, 0, 1, 3)", 1),
    );
    check(
        "sampling without replacement: two of two aces from four cards",
        lt("hyper(4, 2, 2, 2)", "1/6"),
    );

    check("an exact mean, which is a fraction", lt("mean(list(1, 2))", "3/2"));
    check(
        "an even-length median is the mean of the middle two",
        lt("median(list(1, 2, 3, 4))", "5/2") && li("median(list(1, 2, 3))", 2),
    );
    check("variance over three points", lt("variance(list(1, 2, 3))", "2/3"));

    // `sort` would put 2/3 before 1/2, because structurally a numerator of 2
    // precedes one of 1. This is why the library carries its own ordering.
    check(
        "fractions order by value and not by their spelling",
        lt("get(ordered(list(rat(2,3), rat(1,2))), 0)", "1/2"),
    );
    check("a percentage is for reading, and rounds", li("pct(rat(1,4))", 25));

    // --- geometry, over exact coordinates ---------------------------------
    fn geo(expr: &str) -> Option<Value> {
        let mut it = Interp::new();
        eval_line(&mut it, &alloc::format!("use \"/lib/geom\" {}", expr)).ok()
    }
    fn gi(expr: &str, want: i64) -> bool {
        matches!(geo(expr), Some(Value::Int(v)) if v == want)
    }
    fn gt(expr: &str, want: &str) -> bool {
        matches!(geo(expr), Some(v) if v.render() == want)
    }

    const SQ: &str = "list(Pt(0,0), Pt(2,0), Pt(2,2), Pt(0,2))";
    const TRI: &str = "list(Pt(0,0), Pt(1,0), Pt(0,1))";

    check(
        "a second library imports, and declares a record type of its own",
        geo("Pt(1, 2).x").is_some(),
    );
    // The first thing anybody asks a geometry library, and the one whole-number
    // arithmetic answers 0 to.
    check(
        "the unit triangle has area 1/2, which integer division calls 0",
        gt(&alloc::format!("area({})", TRI), "1/2"),
    );
    check(
        "and a square's area is whole, so it comes back whole",
        gi(&alloc::format!("area({})", SQ), 4),
    );
    // Twice the area is a whole number whenever the coordinates are, which is
    // what lets a caller read a winding direction without touching a fraction.
    check(
        "the sign of the doubled area is the winding direction",
        gi(&alloc::format!("area2({})", SQ), 8)
            && gi("clockwise(list(Pt(0,0), Pt(0,2), Pt(2,2), Pt(2,0)))", 1)
            && gi(&alloc::format!("clockwise({})", SQ), 0),
    );
    check(
        "the orientation predicate, which every decision below is",
        gi("collinear(Pt(0,0), Pt(1,1), Pt(2,2))", 1)
            && gi("side(Pt(0,0), Pt(1,0), Pt(0,1))", 1)
            && gi("side(Pt(0,0), Pt(1,0), Pt(0,-1))", -1),
    );
    // Two lines with whole endpoints meet at a rational point, and 2/3 has no
    // exact binary representation at all -- so this is a point floating point
    // cannot put on either line.
    check(
        "lines meet exactly, at a point no float can hold",
        gt("meet(Pt(0,0), Pt(1,1), Pt(0,1), Pt(2,0))", "Pt{x: 2/3, y: 2/3}"),
    );
    check(
        "parallel lines answer nothing rather than a sentinel point",
        matches!(geo("meet(Pt(0,0), Pt(1,1), Pt(0,1), Pt(1,2))"), Some(Value::Nil)),
    );
    check(
        "the centroid of an area, which is not the mean of the corners",
        gt(&alloc::format!("centroid({})", SQ), "Pt{x: 1, y: 1}"),
    );
    check(
        "a midpoint is a fraction when the endpoints are odd apart",
        gt("mid(Pt(0,0), Pt(1,1))", "Pt{x: 1/2, y: 1/2}"),
    );
    check("squared distance, exactly", gi("dist2(Pt(0,0), Pt(3,4))", 25));
    check(
        "inside and outside, decided by the comparison and not by rounding",
        gi(&alloc::format!("inside({}, Pt(1,1))", SQ), 1)
            && gi(&alloc::format!("inside({}, Pt(3,1))", SQ), 0),
    );
    // The hull is the algorithm that most wants exactness: every decision it
    // makes is a sign of `cross`, and a rounded sign builds a "hull" that is
    // not convex. An interior point and a point sitting on an edge must both
    // come off, and a hull is its corners.
    check(
        "a convex hull drops an interior point",
        gi(&alloc::format!("len(hull(push({}, Pt(1,1))))", SQ), 4),
    );
    check(
        "and drops a point lying on an edge, because a hull is its corners",
        gi("len(hull(list(Pt(0,0), Pt(1,0), Pt(2,0), Pt(2,2), Pt(0,2))))", 4),
    );
    // Coordinates may themselves be fractions, which is the case a library
    // built on whole numbers cannot take at all.
    check(
        "the coordinates may be fractions to begin with",
        gt("area(list(Pt(0,0), Pt(rat(1,2),0), Pt(0,rat(1,2))))", "1/8"),
    );

    // --- the libraries as a set, which reading one cannot check -----------
    //
    // `use` is textual inclusion, and a user function **shadows a builtin**
    // deliberately -- `eval` says a program defining `rect` means its own
    // `rect`. So a library declaring `trim` silently takes the string builtin
    // away from every program that imports it, for a name with nothing to do
    // with text. And two libraries declaring one name means whichever was
    // imported second wins, with no message at all.
    //
    // Both of these were live while `/lib/poly` was being written: it had a
    // `trim` and an `area`, and this is what found them.
    {
        let mut shadow = "";
        let mut dupe = "";
        let mut unparsed = "";
        let mut seen: alloc::vec::Vec<&str> = alloc::vec::Vec::new();
        for (path, src) in LIBS {
            for n in declared_names(src) {
                if eval::touch_of(n).is_some() && shadow.is_empty() {
                    shadow = n;
                }
                if seen.contains(&n) && dupe.is_empty() {
                    dupe = n;
                }
                seen.push(n);
            }
            let stem = &path[..path.len() - 6];
            let mut it = Interp::new();
            if eval_line(&mut it, &alloc::format!("use \"{}\" 1", stem)).is_err()
                && unparsed.is_empty()
            {
                unparsed = path;
            }
        }
        check(
            &alloc::format!("all {} librar(ies) import and run their top level", LIBS.len()),
            unparsed.is_empty(),
        );
        check(
            &alloc::format!(
                "{} function(s), none shadowing a builtin ({})",
                seen.len(),
                if shadow.is_empty() { "none" } else { shadow }
            ),
            shadow.is_empty(),
        );
        check(
            &alloc::format!(
                "and no two libraries declare one name ({})",
                if dupe.is_empty() { "none" } else { dupe }
            ),
            dupe.is_empty(),
        );
    }

    fn lib_of(which: &str, expr: &str) -> Option<Value> {
        let mut it = Interp::new();
        eval_line(&mut it, &alloc::format!("use \"{}\" {}", which, expr)).ok()
    }
    fn vi(which: &str, expr: &str, want: i64) -> bool {
        matches!(lib_of(which, expr), Some(Value::Int(v)) if v == want)
    }
    fn vt(which: &str, expr: &str, want: &str) -> bool {
        matches!(lib_of(which, expr), Some(v) if v.render() == want)
    }

    // --- linear algebra ---------------------------------------------------
    const M: &str = "/lib/mat";

    check("a third library imports, and builds an identity", vt(M, "ident(2)", "[[1, 0], [0, 1]]"));
    // Elimination goes through fractions to get there, so a denominator that
    // reduces to one is the evidence the arithmetic never left exactness --
    // `-3` rather than `-3/1` is the canonical-form rule doing the checking.
    check(
        "a whole matrix has a whole determinant, and the rendering is the proof",
        vi(M, "det(list(list(1,2,3), list(4,5,6), list(7,8,10)))", -3),
    );
    check("a row swap flips its sign", vi(M, "det(list(list(0,1), list(1,0)))", -1));
    check("the 3x3 Hilbert determinant is exactly 1/2160", vt(M, "det(hilb(3))", "1/2160"));
    // The demonstration. The inverse of a Hilbert matrix is a matrix of whole
    // numbers, and no floating-point library recovers them -- at 3x3 double
    // precision is already out by 1e-12 and single by far more.
    check(
        "and its inverse is whole numbers, which no float recovers",
        vt(M, "inv(hilb(3))", "[[9, -36, 30], [-36, 192, -180], [30, -180, 180]]"),
    );
    check(
        "A times A inverse is exactly the identity, on the matrix built to break that",
        vi(M, "mul(inv(hilb(3)), hilb(3)) == ident(3)", 1),
    );
    // Substituted back rather than compared against a number typed here, which
    // is the `hull` bargain: the check does not depend on the checker having
    // done the arithmetic correctly by hand.
    check(
        "a 4x4 Hilbert solve, verified by putting the answer back in",
        vi(M, "mv(hilb(4), solve(hilb(4), list(1,2,3,4))) == list(1,2,3,4)", 1),
    );
    check(
        "a singular matrix has determinant exactly zero, and solve answers nothing",
        vi(M, "det(list(list(1,2), list(2,4)))", 0)
            && vi(M, "rank(list(list(1,2), list(2,4)))", 1)
            && matches!(
                lib_of(M, "solve(list(list(1,2), list(2,4)), list(1,2))"),
                Some(Value::Nil)
            ),
    );
    check(
        "a shape mismatch answers nothing rather than a product",
        matches!(lib_of(M, "mul(ident(2), ident(3))"), Some(Value::Nil)),
    );
    // **The promise the header makes, demonstrated rather than asserted.**
    // Entries grow during elimination, and the point of refusing to wrap is
    // that a saturated exact answer is a confidently wrong one. So a size that
    // fits must answer, and a size that does not must *fail* -- a library that
    // silently returned a wrapped determinant would look identical from here.
    check(
        "a 5x5 Hilbert determinant fits, and a 12x12 errors rather than wrapping",
        lib_of(M, "det(hilb(5))").is_some() && lib_of(M, "det(hilb(12))").is_none(),
    );
    check(
        "matrix times vector, which is the shape a layer is",
        vt(M, "mv(list(list(1,2), list(3,4)), list(5,6))", "[17, 39]"),
    );
    // `/lib/geom` refused to write `dist` at all, because the length of (1,1)
    // is not a rational. The tower grew a rung for it, and the `~` is the
    // warning the documentation used to have to carry.
    check(
        "a length is approximate and says so, which /lib/geom declined to answer",
        vt(M, "dist(list(0,0), list(1,1))", "~1.41421"),
    );
    check(
        "the pivot test is an ordering, since != calls an approximate zero nonzero",
        vi(M, "nz(real(0))", 0) && vi(M, "real(0) != 0", 1) && vi(M, "nz(real(1))", 1),
    );

    // --- number theory ----------------------------------------------------
    const N: &str = "/lib/num";

    check("gcd and lcm", vi(N, "gcd(1071, 462)", 21) && vi(N, "lcm(4, 6)", 12));
    check(
        "Bezout's coefficients really do reconstruct the gcd",
        vi(N, "egcd(240, 46).g", 2) && vi(N, "egcd(240, 46).x * 240 + egcd(240, 46).y * 46", 2),
    );
    check(
        "a residue is never negative, where % follows the dividend",
        vi(N, "md(-7, 3)", 2) && vi(N, "-7 % 3", -1),
    );
    check(
        "an inverse mod m, and the honest refusal when there is none",
        vi(N, "modinv(3, 11)", 4) && vi(N, "modinv(2, 4)", -1),
    );
    // The whole reason `mulmod` exists, and the second half is what it would
    // have answered without it -- this machine cannot divide a 128-bit
    // integer, so the usual widening is unavailable rather than merely slow.
    check(
        "a modular multiply whose product overflows 64 bits, with no 128-bit divide",
        vi(N, "mulmod(3999999999, 3999999999, 4000000000)", 1)
            && vi(N, "3999999999 * 3999999999 % 4000000000 != 1", 1),
    );
    check(
        "561 is composite, which a Fermat test calls prime",
        vi(N, "isprime(561)", 0) && vi(N, "isprime(2147483647)", 1),
    );
    check(
        "2^61 - 1 is prime, decided through that multiply rather than a coin flip",
        vi(N, "isprime(2305843009213693951)", 1),
    );
    // Past 2^62, where the doubling used to overflow and `mulmod` used to
    // refuse. (m-1)^2 is congruent to 1 for every m, so this needs no prime to
    // be known and lands squarely in the range that was broken: before the
    // subtract-first `addmod` it answered -1, and `isprime` read that as a
    // witness and called primes composite.
    check(
        "a modular square near i64::MAX, where the doubling used to overflow",
        vi(N, "mulmod(8999999999999999999, 8999999999999999999, 9000000000000000000)", 1)
            && vi(N, "modpow(2, 10, 9000000000000000000)", 1024),
    );
    check("factorisation, with repeats", vt(N, "factor(360)", "[2, 2, 2, 3, 3, 5]"));
    check(
        "divisors, totient and the sieve",
        vi(N, "len(divisors(36))", 9) && vi(N, "totient(36)", 12) && vi(N, "len(primes(100))", 25),
    );
    check(
        "the Chinese remainder theorem, including the pair with no solution",
        vi(N, "crt(2, 3, 3, 5)", 8) && vi(N, "crt(1, 2, 2, 4)", -1),
    );
    check(
        "digits, in any base",
        vi(N, "dsum(9875, 10)", 29) && vi(N, "palin(1221, 10)", 1) && vi(N, "palin(9, 2)", 1),
    );

    // --- polynomials ------------------------------------------------------
    const P: &str = "/lib/poly";

    // Not an approximation of the answer: a different polynomial. In whole
    // numbers `x^2 * (1/2)` is `x^2 * 0`.
    check(
        "the integral of x is x^2/2, which whole numbers render as nothing",
        vt(P, "integ(list(0, 1))", "[0, 0, 1/2]"),
    );
    check("and a definite integral is exact", vi(P, "defint(list(0, 0, 1), 0, 3)", 9));
    check("the derivative", vt(P, "deriv(list(5, 3, 2))", "[3, 4]"));
    // The loop terminates because the leading term cancels *exactly*. In
    // floating point it becomes 1e-17, the degree never falls, and the
    // division runs until something else stops it.
    check(
        "long division, exact enough that the degree actually falls",
        vt(P, "pdiv(list(-1, 0, 1), list(-1, 1)).q", "[1, 1]")
            && vt(P, "pdiv(list(-1, 0, 1), list(-1, 1)).r", "[0]"),
    );
    check(
        "and a quotient that only exists in fractions",
        vt(P, "pdiv(list(1, 0, 1), list(0, 2)).q", "[0, 1/2]"),
    );
    check("a polynomial gcd, monic", vt(P, "pgcd(list(-1, 0, 1), list(1, -2, 1))", "[-1, 1]"));
    check(
        "every rational root of (x-1)(x-2)(x-3)",
        vt(P, "roots(list(-6, 11, -6, 1))", "[1, 2, 3]"),
    );
    check(
        "a root that is not whole, by the rational root theorem",
        vt(P, "roots(list(1, -3, 2))", "[1/2, 1]"),
    );
    // Empty is the complete answer rather than a failure to find anything:
    // x^2 - 2 has two real roots and neither is rational.
    check(
        "x^2 - 2 has no rational root, and an empty list says exactly that",
        vt(P, "roots(list(-2, 0, 1))", "[]"),
    );
    check(
        "rational coefficients are cleared first, so the theorem still applies",
        vt(P, "roots(list(rat(1,2), rat(-3,2), 1))", "[1/2, 1]"),
    );
    check(
        "Lagrange interpolation through three points, exactly",
        vt(P, "interp(list(0, 1, 2), list(1, 3, 7))", "[1, 1, 1]"),
    );
    check(
        "evaluating at an approximation carries the tilde out with it",
        vt(P, "ev(list(1, 1, 1), real(2))", "~7"),
    );
    check(
        "a library importing another library, which nothing had done before",
        vi(P, "gcd(12, 18)", 6),
    );

    // --- physics ----------------------------------------------------------
    const H: &str = "/lib/phys";

    check(
        "the speed of light, exact by definition",
        vi(H, "mag(c())", 299792458) && vt(H, "unit(c())", "m/s"),
    );
    // Checkable twice over: against a textbook, and against its own
    // dimensions. The second needs nobody to have read the formula.
    check(
        "kinetic energy comes out in joules, by dimension rather than by name",
        vi(H, "unit(ke(qty(2, \"kg\"), qty(3, \"m/s\"))) == unit(qty(1, \"J\"))", 1)
            && vi(H, "mag(ke(qty(2, \"kg\"), qty(3, \"m/s\")))", 9),
    );
    check(
        "gravitation comes out in newtons, which is not obvious by inspection",
        vi(H, "unit(grav(qty(1, \"kg\"), qty(1, \"kg\"), qty(1, \"m\"))) == unit(qty(1, \"N\"))", 1),
    );
    check(
        "the ideal gas law comes out in pascals",
        vi(H, "unit(gas_p(qty(1, \"mol\"), qty(300, \"K\"), qty(1, \"m^3\"))) == unit(qty(1, \"Pa\"))", 1),
    );
    // The ACPI battery bug in one line: charge over power is not a time, and
    // the dimension says so without anybody having to notice.
    check(
        "amp-seconds over watts is not a time, which is the battery bug by construction",
        vi(H, "unit(qty(2000, \"A*s\") / qty(10, \"W\")) != unit(qty(1, \"s\"))", 1),
    );
    check(
        "adding a mass to a time is refused rather than answered",
        lib_of(H, "qty(1, \"kg\") + qty(1, \"s\")").is_none(),
    );
    check(
        "Ohm's law, and an absolute temperature from a Celsius one",
        vt(H, "ohm_i(qty(12, \"V\"), qty(4, \"Ohm\"))", "3 A") && vt(H, "celsius(25)", "5963/20 K"),
    );
    // Both halves at once: a wrong unit fails and a magnitude that is not a
    // perfect square fails, because the candidate is squared and compared.
    check(
        "the root of a quantity, settled by squaring it back",
        vt(H, "qsqrt(qty(9, \"m^2/s^2\"), \"m/s\")", "3 m/s")
            && matches!(lib_of(H, "qsqrt(qty(2, \"m^2\"), \"m\")"), Some(Value::Nil))
            && matches!(lib_of(H, "qsqrt(qty(9, \"m^2/s^2\"), \"m\")"), Some(Value::Nil)),
    );
    // The range limit, as a failure rather than as a paragraph. `h` needs a
    // denominator of 10^42 and an `i64` stops at about 10^18.
    check(
        "Planck's constant does not fit, and says so rather than rounding",
        lib_of(H, "si(662607015, -42, \"m^2*kg/s\")").is_none()
            && vi(H, "si(299792458, 0, \"m/s\") == c()", 1),
    );

    // --- chemistry --------------------------------------------------------
    const C: &str = "/lib/chem";

    check(
        "the table reads by symbol and by atomic number",
        vt(C, "el(\"Fe\").name", "iron") && vt(C, "byz(26).sym", "Fe"),
    );
    // A molar mass is a sum of terminating decimals, so it is a fraction and
    // the rounding happens once, where somebody asks to read it.
    check(
        "water is exactly 3603/200 g/mol, and 18.015 when read",
        vt(C, "molar(\"H2O\")", "3603/200") && vt(C, "real(molar(\"H2O\"))", "~18.015"),
    );
    check(
        "a parenthesised group, and its multiplier",
        vt(C, "real(molar(\"Ca(OH)2\"))", "~74.092")
            && vt(C, "real(molar(\"H2SO4\"))", "~98.072"),
    );
    // Cobalt against carbon monoxide. There is no way to be helpful about the
    // capital and be right about both, so the case is the answer.
    check(
        "Co is cobalt and CO is carbon monoxide",
        vt(C, "real(molar(\"Co\"))", "~58.933") && vt(C, "real(molar(\"CO\"))", "~28.01"),
    );
    // The refusals, and there are three of them because the table is partial
    // on purpose: an unknown element must take the whole formula with it
    // rather than contributing nothing to the sum.
    check(
        "an element the table does not carry refuses the whole formula",
        vi(C, "molar(\"H2Xx\")", -1) && vi(C, "molar(\"Ca(OH2\")", -1) && vi(C, "molar(\"CaOH)2\")", -1),
    );
    check(
        "atoms are counted through a group, not only at the top level",
        vi(C, "countof(\"Ca(OH)2\", \"O\")", 2) && vi(C, "countof(\"Al2(SO4)3\", \"O\")", 12),
    );
    check(
        "a mass fraction, and grams from moles exactly",
        vi(C, "round(fracmass(\"Fe2O3\", \"Fe\") * 1000)", 699)
            && vt(C, "gram(2, \"H2O\")", "3603/100"),
    );
    // **The claim exactness is for.** Four point oh three two grams of
    // hydrogen and thirty-one point nine nine eight of oxygen are exactly two
    // moles and exactly one, which is exactly what `2 H2 + O2` asks for. In
    // floating point those two ratios differ in the last bit and one of the
    // reagents is declared limiting on a rounding error.
    check(
        "two exactly stoichiometric reagents compare equal, not by a rounding error",
        vi(C, "limiting(rat(4032,1000), \"H2\", 2, rat(31998,1000), \"O2\", 1)", 0)
            && vi(C, "limiting(4, \"H2\", 2, 16, \"O2\", 1)", 1),
    );

    // --- how approximate the approximations are ---------------------------
    //
    // `tensor.rs` says of itself that these are "accurate enough for
    // inference, where a 1e-6 error in a logit changes nothing". That is a
    // statement about a use rather than about digits, and it is now the
    // language's `exp`, `ln`, `sin` and `cos` -- so what it means gets
    // measured and printed rather than quoted.
    //
    // Identities rather than a table of constants: a round trip and a
    // Pythagorean sum are checkable without a reference implementation, which
    // is the only kind of check this machine can make about itself.
    {
        use crate::ai::tensor;

        let mut worst_trip = 0.0f32;
        let mut k = 1;
        while k <= 16 {
            let x = k as f32 * 0.5;
            let back = tensor::lnf(tensor::expf(x));
            let e = ((back - x) / x).abs();
            if e > worst_trip {
                worst_trip = e;
            }
            k += 1;
        }

        let mut worst_pyth = 0.0f32;
        let mut j = 0;
        while j < 32 {
            let x = j as f32 * 0.25;
            let s = tensor::sinf(x);
            let c = tensor::cosf(x);
            let e = (s * s + c * c - 1.0).abs();
            if e > worst_pyth {
                worst_pyth = e;
            }
            j += 1;
        }

        // Hardware, and therefore exact where the answer is representable.
        // Worth separating from the series above, because a reader told "the
        // maths is approximate" would otherwise assume this is too.
        let sqrt_exact = tensor::sqrtf(4.0) == 2.0
            && tensor::sqrtf(144.0) == 12.0
            && tensor::sqrtf(1.0) == 1.0;

        check(
            &alloc::format!(
                "ln(exp(x)) returns x to {:.1e} relative, over 0.5..8",
                worst_trip
            ),
            worst_trip < 1.0e-4,
        );
        // **The header's 1e-6 is about `expf`, not about `sinf`, and this is
        // the first thing to measure the difference.** Round-tripping through
        // exp and ln holds to 2.1e-6; the Pythagorean identity is off by
        // 3.1e-4, three hundred times looser.
        //
        // The arithmetic explains it rather than merely recording it. `sinf`
        // is a Taylor series truncated after x^7, so the first dropped term is
        // x^9/9!, and at the fold boundary x = pi/2 that is 1.5708^9 / 362880
        // = 1.6e-4. Two of those, squared and summed, is the 3e-4 measured.
        //
        // And the tree already knew, without anyone joining it up: the claim
        // at `ai/mod.rs:143` checks `sinf(pi/2)` against a tolerance of
        // **1e-4**, which is the real figure, while the module header beside
        // it says 1e-6. The bound here is 1e-3 so it sits above the measured
        // value with room, and the printed number is the thing to read.
        //
        // Worth knowing beyond this language: `model.rs:971` builds the RoPE
        // tables with these, so the model's positional encoding carries that
        // error too. That is within the "enough for inference" target it was
        // written for, and it is not 1e-6.
        check(
            &alloc::format!(
                "sin^2 + cos^2 is 1 to {:.1e} -- the series, not the 1e-6 the header claims",
                worst_pyth
            ),
            worst_pyth < 1.0e-3,
        );
        check(
            "sqrt is one hardware instruction, so a perfect square is exact",
            sqrt_exact,
        );
    }

    ok
}

/// Programs run end to end, and compared against what they should produce.
///
/// The point of the language is that the machine will write in it, so what
/// matters is not that each piece works but that a whole small program does:
/// define a function, hold a list, loop over it, return early. Each case here
/// is a program, and each is one an application would actually contain.
///
/// Silent, returning only a verdict, because the registry that calls it prints
/// a line per check already.
pub fn selftest() -> bool {
    fn run(src: &str) -> Option<eval::Value> {
        let mut it = eval::Interp::new();
        eval_line(&mut it, src).ok()
    }
    fn int(src: &str, want: i64) -> bool {
        matches!(run(src), Some(eval::Value::Int(v)) if v == want)
    }
    fn text(src: &str, want: &str) -> bool {
        match run(src) {
            Some(v) => v.render() == want,
            None => false,
        }
    }

    // --- the gate ---------------------------------------------------------
    //
    // Checked by name and never by calling. Half this table pokes memory,
    // writes I/O ports or paints over the screen, and a test suite that
    // exercises every row would be a test suite that scribbles on the machine
    // to prove it can.
    {
        use eval::{Caps, Touch, BUILTINS};

        // No two rows may claim the same name. A duplicate means the first
        // silently decides the class, and if they disagree the stricter one is
        // the one that never runs.
        for (i, (n, ..)) in BUILTINS.iter().enumerate() {
            if BUILTINS.iter().skip(i + 1).any(|(m, ..)| m == n) {
                return false;
            }
        }

        // A name absent from the table is refused before dispatch. This is the
        // property the whole inversion rests on: reaching the match at all
        // requires a row, so an arm added without one is dead code rather than
        // an ungated builtin.
        let mut op = eval::Interp::new();
        if eval_line(&mut op, "nosuchbuiltin(1)").is_ok() {
            return false;
        }

        // Everything a stored program may not do is refused for a stored
        // program, and every row is covered rather than a chosen few.
        let mut jailed = eval::Interp::sandboxed("/app/t");
        for (name, touch, lo, _) in BUILTINS {
            if *touch == Touch::Pure || *touch == Touch::Read || *touch == Touch::Write {
                continue;
            }
            let args = alloc::vec!["0"; *lo].join(", ");
            let src = alloc::format!("{}({})", name, args);
            match eval_line(&mut jailed, &src) {
                Err(e) if e.contains("may not") => {}
                // Anything else means it ran, or failed for the wrong reason.
                _ => return false,
            }
        }

        // ...and the ones it may do are not refused by the gate. `read` of a
        // path that does not exist fails on the path, which is the arm
        // answering rather than the gate.
        if eval::available(Caps::Sandbox).len() >= BUILTINS.len() {
            return false;
        }
        if !eval::available(Caps::Operator).contains(&"poke8") {
            return false;
        }
        if eval::available(Caps::Sandbox).contains(&"poke8") {
            return false;
        }

        // Every row that can be exercised without damaging anything has an
        // implementation behind it. This is the other half of the inversion:
        // the gate makes an arm without a row unreachable, and this makes a row
        // without an arm a boot failure rather than something a program
        // discovers at three in the morning.
        //
        // Pure and Read only. The rest write files, poke memory, drive I/O
        // ports or paint over the screen, and a suite that called them to prove
        // they exist would be scribbling on the machine to do it. Those rows
        // are covered by the sandbox check above, which reaches every one.
        //
        // Numbers as arguments, because a builtin wanting a string accepts one
        // (`render` never fails) while a builtin wanting a number rejects a
        // string -- so `0` is the argument that gets furthest into the arm.
        //
        // Captured, because `print`, `println` and `heap` are on this list and
        // do what they say. The boot log is this system's test suite, and a
        // stray "7120 B used of ..." in the middle of it is how a real FAIL
        // three lines later gets skimmed past.
        crate::gfx::console::begin_capture();
        let mut missing = false;
        for (name, touch, lo, _) in BUILTINS {
            if *touch != Touch::Pure && *touch != Touch::Read {
                continue;
            }
            let args = alloc::vec!["0"; *lo].join(", ");
            if let Err(e) = eval_line(&mut op, &alloc::format!("{}({})", name, args)) {
                if e.contains("no implementation") {
                    missing = true;
                }
            }
        }
        let _ = crate::gfx::console::end_capture();
        if missing {
            return false;
        }

        // Arity is enforced from the table, before the arm sees the arguments.
        if eval_line(&mut op, "hex()").is_ok() || eval_line(&mut op, "hex(1, 2)").is_ok() {
            return false;
        }
        // A variadic row accepts none and many.
        if eval_line(&mut op, "list()").is_err() || eval_line(&mut op, "list(1,2,3,4,5)").is_err() {
            return false;
        }
    }

    // The text builtins, on the operations a program actually strings
    // together: split a line, take a field, test it, put it back.
    if !text("join(split(\"a:b:c\", \":\"), \"-\")", "a-b-c") {
        return false;
    }
    if !int("find(\"hello\", \"ll\")", 2) {
        return false;
    }
    if !int("find(\"hello\", \"z\")", -1) {
        return false;
    }
    if !text("substr(\"hello\", 1, 3)", "ell") {
        return false;
    }
    if !text("upper(trim(\"  ok  \"))", "OK") {
        return false;
    }
    if !text("replace(\"a.b.c\", \".\", \"/\")", "a/b/c") {
        return false;
    }
    // An empty separator splits into characters rather than looping forever.
    if !int("len(split(\"abc\", \"\"))", 3) {
        return false;
    }
    // Hex round-trips, which is what a program reading a stored hash needs.
    if !text("hexdec(hexenc(\"hi\"))", "hi") {
        return false;
    }
    if !text("hexenc(\"A\")", "41") {
        return false;
    }
    // Lists: sort numerically rather than by spelling, which is the one a
    // string sort gets wrong and nobody notices until there are ten rows.
    if !text("join(sort(list(10, 9, 100)), \",\")", "9,10,100") {
        return false;
    }
    if !int("index(list(\"a\", \"b\"), \"b\")", 1) {
        return false;
    }
    if !int("len(range(0, 5))", 5) {
        return false;
    }
    // Bounded, so a generated program cannot ask for a billion-element list
    // in a repaint path and take the heap with it.
    if !int("len(range(0, 999999999))", 65_536) {
        return false;
    }
    if !int("len(repeat(\"x\", 999999999))", 65_536) {
        return false;
    }
    if !int("sqrt(144)", 12) {
        return false;
    }

    // --- exact fractions --------------------------------------------------
    //
    // The claim that earns its place is `rat(1,3) * 3 == 1`. A float cannot
    // make that true, and every domain worth adding -- physics, statistics,
    // competition mathematics -- is written in numbers that have to survive
    // being divided and multiplied back. It is also what keeps `differ`'s
    // no-tolerance rule intact: exact values compare exactly.
    if !int("rat(1,3) * 3", 1) {
        return false;
    }
    // Reduced on the way in, so one number has one value. Without this
    // `rat(2,4) == rat(1,2)` is false and two programs computing the same
    // answer disagree about it.
    if !text("rat(2,4)", "1/2") || !int("rat(2,4) == rat(1,2)", 1) {
        return false;
    }
    // A denominator of one is an `Int`, never a `Rat`. `int` here is the
    // check: it matches the variant, so a `3/1` would fail it while
    // rendering identically.
    if !int("rat(3,1)", 3) || !int("rat(6,3)", 2) {
        return false;
    }
    // The sign lives in the numerator, so a negative has one spelling.
    if !text("rat(1,-2)", "-1/2") || !int("rat(-1,2) == rat(1,-2)", 1) {
        return false;
    }
    // **Integer division is untouched.** This is the whole compatibility
    // argument: every core, seeded tool and generated candidate written
    // before fractions existed still means what it meant, because two `Int`s
    // never reach the exact path. Exactness starts at `rat` and propagates.
    if !int("10/3", 3) || !int("7/2", 3) {
        return false;
    }
    // And propagates once it has started.
    if !text("rat(1,3) + rat(1,3)", "2/3") || !text("1 + rat(1,2)", "3/2") {
        return false;
    }
    // Cross-multiplied, so a comparison does not go through a division that
    // would round before the answer.
    if !int("rat(1,3) < rat(1,2)", 1) || !int("rat(2,3) > rat(1,2)", 1) {
        return false;
    }
    // The three roundings disagree, which is exactly why `as_int` refuses to
    // pick one silently. -7/2 is -3.5: down, up, and half away from zero.
    if !int("floor(rat(-7,2))", -4) || !int("ceil(rat(-7,2))", -3)
        || !int("round(rat(-7,2))", -4) || !int("round(rat(7,2))", 4)
    {
        return false;
    }
    if !int("num(rat(3,4))", 3) || !int("den(rat(3,4))", 4) || !int("den(5)", 1) {
        return false;
    }
    // Refusals. A zero denominator is not a number, and an exact answer that
    // does not fit is an error rather than a wrapped one -- a confidently
    // wrong exact value is worse than none, which is the trade exactness
    // makes and the reason the arithmetic is checked.
    if run("rat(1,0)").is_some() {
        return false;
    }
    if run("rat(1,4000000000) * rat(1,4000000000)").is_some() {
        return false;
    }
    // A fraction is not a whole number and will not pretend to be one.
    if run("repeat(\"x\", rat(1,2))").is_some() {
        return false;
    }

    // --- arithmetic that would stop the machine ---------------------------
    //
    // Every one of these is legal to *write* and, unguarded, is either a
    // hardware fault or a silently wrong answer. In a kernel with no process
    // isolation and every IDT vector but `#BP` fatal, the first kind halts the
    // machine -- so a program a model wrote, or a corpus supplied, must not be
    // able to reach one.
    //
    // The guards were already here and had never been checked, which is the
    // same objection `differ` makes about its own suite: a defence nobody has
    // watched refuse anything is indistinguishable from an absent one.
    //
    // `i64::MIN` is built rather than written, because the literal
    // 9223372036854775808 does not fit an `i64` and the lexer would have to
    // read it before the unary minus could apply.
    const MIN: &str = "(0 - 9223372036854775807 - 1)";

    // `idiv` raises `#DE` on `i64::MIN / -1` -- the quotient has no
    // representation. On this machine that is a fatal fault, not an exception
    // somebody catches.
    if !int(&alloc::format!("{} / -1", MIN), i64::MIN) {
        return false;
    }
    if !int(&alloc::format!("{} % -1", MIN), 0) {
        return false;
    }
    // A shift wider than the type is undefined in C and masked by the
    // hardware; either way the answer is not what was written. Masked here,
    // deliberately and consistently, rather than faulting.
    if !int("1 << 64", 1) || !int("1 << 65", 2) || !int("1 << -1", i64::MIN) {
        return false;
    }
    // `abs` of the most negative number is not representable either.
    if !int(&alloc::format!("abs({})", MIN), i64::MAX) {
        return false;
    }
    // Saturating rather than wrapping, so a big power is a big number and
    // never a small negative one.
    //
    // Base 3 rather than 2, and that is the claim working rather than a
    // detail: the exponent is clamped to 62, so `pow(2, 1000)` reaches 2^62 =
    // 4.6e18 and never saturates at all. Written with 2 this would have passed
    // or failed for a reason that has nothing to do with saturation.
    if !int("pow(3, 1000)", i64::MAX) || !int("pow(2, 10)", 1024) {
        return false;
    }
    // Rounding the most negative number must not overflow on the way.
    if !int(&alloc::format!("floor({})", MIN), i64::MIN)
        || !int(&alloc::format!("ceil({})", MIN), i64::MIN)
        || !int(&alloc::format!("round({})", MIN), i64::MIN)
    {
        return false;
    }
    // Division and remainder by zero are errors, not faults.
    if run("1 / 0").is_some() || run("1 % 0").is_some() {
        return false;
    }
    // And the exact path refuses rather than wrapping, which is the one place
    // this language does *not* wrap. An `Int` is a machine word and wrapping
    // is what a hash wants; a `Rat` is a number, and a wrapped numerator is a
    // confidently wrong exact answer.
    if run(&alloc::format!("rat(1, 3) * {}", MIN)).is_some() {
        return false;
    }

    // --- quantities, and the unit error that is now an error --------------
    //
    // `acpi` records what a missing unit cost: "a capacity in mAh over a rate
    // in mW gives a number that looks like a time and is wrong by the
    // battery's voltage". Nothing errors in that failure, nothing is out of
    // range, and the answer is simply about something else. These are that
    // failure caught at the operation.
    if !text("qty(5, \"m\")", "5 m") || !text("qty(rat(1,3), \"m\")", "1/3 m") {
        return false;
    }
    // Multiplying adds exponents and dividing subtracts them, which is the
    // whole of dimensional analysis.
    if !text("qty(10, \"m\") / qty(2, \"s\")", "5 m/s")
        || !text("qty(3, \"m\") * qty(4, \"s\")", "12 m*s")
        || !text("qty(1, \"m\") / qty(1, \"s^2\")", "1 m/s^2")
    {
        return false;
    }
    // **A dimension that cancels leaves a number.** Same rule as a `Rat` never
    // holding a denominator of one: without it `6 m / 2 m` is a quantity that
    // renders as `3` and compares unequal to `3`.
    if !int("qty(6, \"m\") / qty(2, \"m\")", 3) || !int("qty(6,\"m\") / qty(2,\"m\") == 3", 1) {
        return false;
    }
    // Adding metres to seconds is the error this exists for. So is adding a
    // bare number to a quantity -- dimensionless is a dimension, and there is
    // no quantity a plain number may be added to.
    if run("qty(3, \"m\") + qty(4, \"s\")").is_some() {
        return false;
    }
    if run("qty(3, \"m\") + 2").is_some() {
        return false;
    }
    // ...and the same units add fine, which is what makes the refusal a check
    // rather than a blanket.
    if !text("qty(3, \"m\") + qty(4, \"m\")", "7 m") {
        return false;
    }
    // Ordering two different kinds has no answer, so it is refused rather than
    // decided on the magnitudes -- 3 seconds is not less than 4 metres.
    if run("qty(3, \"s\") < qty(4, \"m\")").is_some() {
        return false;
    }
    if !int("qty(3, \"m\") < qty(4, \"m\")", 1) {
        return false;
    }
    // Derived names are a table of things already expressible, so a watt and
    // its base spelling are the same dimension and compare equal.
    if !int("qty(1, \"W\") == qty(1, \"kg*m^2/s^3\")", 1) {
        return false;
    }
    // **The battery bug, as a claim.** Charge over power is not a time, and
    // energy over power is. A unit-free version of this answers a plausible
    // number in both cases and nobody finds out.
    if !text("qty(2, \"J\") / qty(1, \"W\")", "2 s") {
        return false;
    }
    // Charge over power is `s^4*A/m^2*kg` -- not a time, and not anything with
    // a name. Worked out by hand rather than copied from a run, because a
    // claim that records whatever the code printed asserts nothing.
    if !text("qty(2, \"A*s\") / qty(1, \"W\")", "2 s^4*A/m^2*kg") {
        return false;
    }
    // A unit the table does not know is refused rather than ignored: a typo
    // that quietly produced a dimensionless number would defeat the point.
    if run("qty(1, \"furlong\")").is_some() {
        return false;
    }
    // Dropping the unit has to be asked for. Every builtin that wants a plain
    // number refuses a quantity, and names `mag` when it does.
    if !int("mag(qty(7, \"m\"))", 7) || !text("unit(qty(1, \"m/s\"))", "m/s") {
        return false;
    }
    if run("floor(qty(7, \"m\"))").is_some() {
        return false;
    }

    // --- approximation, which says it is one ------------------------------
    //
    // The tower is exact everywhere else, so the only honest way to hold the
    // square root of two is a type that admits what it is. `~` is in the
    // rendering for that reason: a transcript, a ledger line or a forest
    // node's method shows which of its numbers are trustworthy.
    if !text("real(2)", "~2") || !text("pi()", "~3.14159") {
        return false;
    }
    // **Inexactness is opt-in, exactly as exact division was.** `sqrt(2)` is
    // still the integer root and still 1, so every program written before this
    // answers what it answered; the irrational one is asked for by name.
    if !int("sqrt(2)", 1) || !int("sqrt(144)", 12) {
        return false;
    }
    if !text("sqrt(real(2))", "~1.41421") {
        return false;
    }
    // There is no way to *write* one. The lexer has no float -- which is what
    // keeps `.` unambiguously field access -- so an approximate value can only
    // be produced by asking, never typed.
    if run("3.7").is_some() {
        return false;
    }
    // Contagious, because an exact value meeting an approximate one cannot
    // produce an exact answer.
    if !text("real(1) + 1", "~2") || !text("1 + real(1)", "~2") || !text("rat(1,2) * real(2)", "~1") {
        return false;
    }
    // **An approximation is never equal to an exact number, because it is not
    // one.** That reads as strict and is the useful behaviour: the right way
    // to test a float is a tolerance, and making `==` answer true here would
    // let a program compare two approximations and believe the result.
    // Ordering still crosses, so a bound is asked the ordinary way.
    if !int("real(2) == 2", 0) {
        return false;
    }
    if !int("real(2) < 3", 1) || !int("real(2) > 1", 1) {
        return false;
    }
    // And the demonstration that it really is inexact: a root squared does not
    // come back. `1.9999999` is the f32 answer, and it is not 2.
    if !text("sqrt(real(2)) * sqrt(real(2))", "~2") {
        return false;
    }
    if !int("sqrt(real(2)) * sqrt(real(2)) < 2", 1) {
        return false;
    }
    // A dimension must not disappear into a sine. An approximation has no
    // unit, so mixing is refused rather than silently dropping one.
    if run("qty(2, \"m\") * real(3)").is_some() {
        return false;
    }
    // Rounding is the honest exit from the type: it is the operation that
    // turns an inexact number into an exact whole one.
    if !int("floor(real(rat(37,10)))", 3) || !int("ceil(real(rat(37,10)))", 4)
        || !int("round(real(rat(37,10)))", 4)
    {
        return false;
    }
    if !int("round(real(rat(-37,10)))", -4) || !int("floor(real(rat(-37,10)))", -4) {
        return false;
    }
    // And everything that wants a whole number still refuses one, naming the
    // way to ask.
    if run("repeat(\"x\", real(2))").is_some() {
        return false;
    }

    // --- the four operators that had never met the tower ------------------
    //
    // Each of these went through `as_int`, which refuses a fraction, a
    // quantity and an approximation by design -- so the three kinds the tower
    // exists for could not be negated, compared, or raised to a power. The
    // libraries were the evidence: `/lib/geom` carried a hand-written `absv`
    // with `0 - x` inside it, and `/lib/prob` a hand-written selection sort,
    // both working around the language rather than using it. A gap a library
    // papers over is a gap that survives.
    if !text("-rat(1,2)", "-1/2") || !text("-qty(5, \"m\")", "-5 m") {
        return false;
    }
    if !text("-real(2)", "~-2") || !int("-3", -3) {
        return false;
    }
    if !text("abs(rat(-1,2))", "1/2") || !text("abs(qty(-5, \"m\"))", "5 m") {
        return false;
    }
    // `min` answers the value rather than a whole number, so a third really is
    // a third. It reads the same ordering `<` does, which is the point: a
    // `min` that disagreed with `<` would be two answers to one question.
    if !text("min(rat(1,2), rat(1,3))", "1/3") || !text("max(rat(1,2), rat(1,3))", "1/2") {
        return false;
    }
    if !text("clamp(rat(5,2), 0, 2)", "2") || !text("min(qty(3, \"m\"), qty(4, \"m\"))", "3 m") {
        return false;
    }
    // Comparing two quantities of different kinds has no answer, so `min`
    // refuses it exactly as `<` does rather than deciding on the magnitudes.
    if run("min(qty(3, \"m\"), qty(4, \"s\"))").is_some() {
        return false;
    }
    // A whole base with a non-negative exponent is untouched, saturation and
    // the clamp at 62 included -- the `pow(3, 1000)` claim above still reads
    // `i64::MAX`, so nothing written before this moved.
    //
    // **A negative exponent used to answer 1, for every base.** Not a wrong
    // answer to the question -- an answer to no question. It is the reciprocal
    // now, which is exact for a fraction and inverts a dimension with it.
    if !text("pow(2, -1)", "1/2") || !text("pow(rat(1,2), 3)", "1/8") {
        return false;
    }
    if !text("pow(qty(2, \"m\"), 3)", "8 m^3") || !text("pow(qty(2, \"s\"), -1)", "1/2 1/s") {
        return false;
    }
    if !text("pow(qty(3, \"m\"), 0)", "1") || !text("pow(real(2), -2)", "~0.25") {
        return false;
    }
    // Bounded, and **refused rather than clamped**. Clamping is what the old
    // implementation did when it turned every negative exponent into 1: it
    // answers a different question, confidently. The whole-number path keeps
    // its clamp and only because changing it would move what programs written
    // before this answered.
    if run("pow(rat(1,2), 100000)").is_some() || run("pow(2, -100000)").is_some() {
        return false;
    }
    // `sort` tested "is this a number" with `as_int`, so a list of fractions
    // fell through to comparing *renderings* -- where "19/2" sorts before "9"
    // and the list comes back wrong while looking sorted.
    if !text("get(sort(list(rat(19,2), 9)), 0)", "9") {
        return false;
    }
    if !text("get(sort(list(rat(2,3), rat(1,2))), 0)", "1/2") {
        return false;
    }

    // --- records ----------------------------------------------------------
    //
    // A declaration, a constructor, a field read, a field write, and the
    // rendering all of those are read through.
    if !int("rec Host { name, port } h = Host(\"a\", 80) h.port", 80) {
        return false;
    }
    if !text("rec Host { name, port } h = Host(\"a\", 80) h.name", "a") {
        return false;
    }
    if run("rec Host { name, port } h = Host(\"a\", 80) h.nope = 1").is_some() {
        return false;
    }
    if !int("rec Host { name, port } h = Host(\"a\", 80) h.port = 443 h.port", 443) {
        return false;
    }
    if !text("rec P { x, y } P(1, 2)", "P{x: 1, y: 2}") {
        return false;
    }
    // Values, not references: a copy taken before the write keeps the old
    // field. This is the property that means nothing has to be said about
    // aliasing, so it is worth a test rather than a comment.
    if !int("rec P { x } a = P(1) b = a a.x = 9 b.x", 1) {
        return false;
    }
    // Records nest, and reading through two levels works.
    if !int("rec In { v } rec Out { i } o = Out(In(7)) o.i.v", 7) {
        return false;
    }
    // Constructor arity comes from the declaration.
    if run("rec P { x, y } P(1)").is_some() || run("rec P { x, y } P(1, 2, 3)").is_some() {
        return false;
    }
    // A field that was never declared is not readable.
    if run("rec P { x } P(1).y").is_some() {
        return false;
    }
    // Reading a field off something that is not a record is a mistake in the
    // program, not a Nil to carry forward.
    if run("(1).x").is_some() || run("\"s\".x").is_some() {
        return false;
    }
    // A record may not take a builtin's name, because it would install a
    // constructor and leave `len(x)` ambiguous to a reader.
    if run("rec len { a }").is_some() {
        return false;
    }
    // Two fields with one name is a declaration nobody can index.
    if run("rec P { x, x }").is_some() {
        return false;
    }

    // --- types --------------------------------------------------------------
    //
    // Optional. A function that says nothing behaves as it always did, which
    // is what every application already written depends on.
    if !int("fn f(a) { return a } f(1)", 1) {
        return false;
    }
    if !int("fn f(a: int): int { return a + 1 } f(1)", 2) {
        return false;
    }
    // The mistake is caught at the call, where the caller is, and not
    // wherever the value happened to be used.
    if run("fn f(a: int) { return a } f(\"x\")").is_some() {
        return false;
    }
    // A return type is checked too, which is the half that catches a function
    // falling off its end when it was supposed to answer something.
    if run("fn f(): int { return \"x\" } f()").is_some() {
        return false;
    }
    if run("fn f(): int { } f()").is_some() {
        return false;
    }
    // `any` is explicit as well as default.
    if !text("fn f(a: any): any { return a } f(\"x\")", "x") {
        return false;
    }
    // Record types are usable as annotations, and are checked by name.
    if !int("rec P { x } fn f(p: P): int { return p.x } f(P(4))", 4) {
        return false;
    }
    if run("rec P { x } rec Q { x } fn f(p: P) { return 1 } f(Q(1))").is_some() {
        return false;
    }
    // Field annotations hold at construction and at assignment, or the
    // annotation would be a comment.
    if run("rec P { x: int } P(\"s\")").is_some() {
        return false;
    }
    if run("rec P { x: int } p = P(1) p.x = \"s\"").is_some() {
        return false;
    }
    // A list is a type like any other.
    if !int("fn f(l: list): int { return len(l) } f(list(1, 2))", 2) {
        return false;
    }

    // --- the shapes the kernel hands back -----------------------------------
    //
    // By declaration rather than by calling, for the reason the gate check
    // gives: these read live hardware and a boot suite should not depend on
    // what the machine happens to have in it. What is worth pinning here is
    // that every shape is well formed and that nothing in the table collides
    // with a builtin -- both of which are how a record gets silently
    // unreachable.
    {
        use eval::KERNEL_RECS;
        for (i, (name, fields)) in KERNEL_RECS.iter().enumerate() {
            if fields.is_empty() {
                return false;
            }
            // A record whose name is a builtin could never be constructed by
            // name, because the builtin is found first.
            if eval::is_builtin(name) {
                return false;
            }
            if KERNEL_RECS.iter().skip(i + 1).any(|(m, _)| m == name) {
                return false;
            }
            // No duplicate field names: `.x` would answer whichever came
            // first, and the other would be unreachable.
            for (j, (f, _)) in fields.iter().enumerate() {
                if fields.iter().skip(j + 1).any(|(g, _)| g == f) {
                    return false;
                }
            }
        }
        // Registered in every interpreter, so an annotation names something
        // real rather than a record the program forgot to declare.
        let mut op3 = eval::Interp::new();
        if eval_line(&mut op3, "rec Device { x }").is_ok() {
            return false;
        }
        // And reachable in every interpreter, which is the half that stopped
        // being a `BTreeMap` lookup. `Interp::new` no longer copies
        // `KERNEL_RECS` into the program's own table, so a kernel shape is
        // found by a second search rather than by having been allocated 49
        // strings at construction. These three claims are the ones that would
        // have gone quiet if that second search were missed: the arity, the
        // field names, and the declared type of a field on assignment. Every
        // other record claim above exercises only a program's own `rec`, and
        // all of them passed with the kernel half of the lookup unwritten.
        if !text("t = Tcp(\"open\", \"\") t.state", "open") {
            return false;
        }
        if run("Tcp(\"open\")").is_some() {
            return false;
        }
        if run("t = Tcp(\"open\", \"\") t.state = 1").is_some() {
            return false;
        }
        // `ls` answers a list now, not newline-joined text. `None` is allowed
        // and is the usual case here: the boot selftests run before
        // `sysbox::init`, so there is no namespace to list yet. What this
        // pins is that when it does answer, it does not answer a string.
        match run("ls(\"/\")") {
            Some(eval::Value::List(_)) | None => {}
            _ => return false,
        }
    }

    // --- use ----------------------------------------------------------------
    //
    // The file half needs a namespace and is driven under QEMU. What is
    // checkable here is the refusal, which is the part that matters: a
    // sandboxed program's imports are confined so that what it depends on can
    // be found in one place.
    {
        let mut jailed = eval::Interp::sandboxed("/app/t");
        match eval_line(&mut jailed, "use \"/ai/godel/HEAD\"") {
            Err(e) if e.contains("may only use") => {}
            _ => return false,
        }
        // Its own subtree is allowed to get as far as looking for the file.
        match eval_line(&mut jailed, "use \"/app/t/helper\"") {
            Err(e) if e.contains("no such program") => {}
            _ => return false,
        }
        // An operator is not confined, and gets the same honest miss.
        let mut op2 = eval::Interp::new();
        match eval_line(&mut op2, "use \"/nowhere/at/all\"") {
            Err(e) if e.contains("no such program") => {}
            _ => return false,
        }
    }

    // Functions: definition, call, arguments, and a value coming back out.
    if !int("fn add(a, b) { return a + b } add(2, 3)", 5) {
        return false;
    }
    // Recursion, which is the shape of anything that walks a structure.
    if !int("fn f(n) { if (n < 2) { return n } return f(n - 1) + f(n - 2) } f(10)", 55) {
        return false;
    }
    // A return inside a loop leaves the function, not just the loop. This is
    // the one that a sentinel-based return gets wrong if a block forgets to
    // check it.
    if !int("fn first_big(n) { i = 0 while (i < 100) { if (i > n) { return i } i = i + 1 } return -1 } first_big(7)", 8) {
        return false;
    }
    // Falling off the end yields nothing rather than the last expression,
    // which is what a procedure called for its effect should say.
    if !text("fn noisy() { 42 } noisy()", "") {
        return false;
    }

    // Lists: build, measure, index, extend, replace.
    if !text("list(1, 2, 3)", "[1, 2, 3]") {
        return false;
    }
    if !int("len(list(1, 2, 3))", 3) {
        return false;
    }
    if !int("get(list(4, 5, 6), 1)", 5) {
        return false;
    }
    if !text("push(list(1), 2)", "[1, 2]") {
        return false;
    }
    if !text("set(list(1, 2), 0, 9)", "[9, 2]") {
        return false;
    }
    // Past the end is nothing, not a fault: a program walking a list it did
    // not build should be able to ask.
    if !text("get(list(1), 5)", "") {
        return false;
    }
    // Values and not references. If `push` mutated, `xs` would have grown too,
    // and two names would disagree about one list -- the whole reason this
    // language does not have references.
    if !text("xs = list(1) ys = push(xs, 2) xs", "[1]") {
        return false;
    }

    // Scope: a parameter shadows a global without destroying it, and an
    // assignment to a name that already exists outside updates it there.
    if !int("n = 1 fn shadow(n) { n = 99 return n } shadow(5) n", 1) {
        return false;
    }
    if !int("g = 1 fn bump() { g = g + 1 } bump() bump() g", 3) {
        return false;
    }

    // A function cannot see, or write, a caller's locals.
    //
    // Both of these passed for as long as the scope walk went all the way down
    // the stack, and neither was ever asserted -- the two claims above check a
    // parameter shadowing a global and a function updating a global, which
    // hold under either rule. `call_user` pushes the callee's frame on top of
    // the caller's without popping it, so walking innermost-first let a callee
    // resolve a free name against whichever caller above it happened to use
    // that name: the binding depended on the dynamic call chain, which no
    // reader of the callee could work out.
    //
    // The read case answered 42 and the write case answered 99 at the prompt
    // before this was narrowed.
    if run("fn inner() { return x } fn outer() { x = 42 return inner() } outer()").is_some() {
        return false;
    }
    if !int("fn poke() { y = 99 return 0 } fn host() { y = 1 poke() return y } host()", 1) {
        return false;
    }

    // A whole program of the shape an application has: state in a list, a
    // function over it, a loop driving them.
    if !int(
        "fn total(xs) { s = 0 i = 0 while (i < len(xs)) { s = s + get(xs, i) i = i + 1 } return s } \
         items = list() items = push(items, 10) items = push(items, 20) total(items)",
        30,
    ) {
        return false;
    }

    // A program spread over real lines, which is what a stored one is.
    //
    // Every case above is a single-line string, and that is exactly how the
    // language shipped for months looking fine while `run` could not execute
    // any file containing a function: `sysbox::cmd_run` fed the blob to
    // `eval_line` one line at a time, and `fn total(xs) {` on its own is not a
    // statement. The lexer has always treated a newline as whitespace, so the
    // defect was never here -- but nothing here could have caught it either.
    // It can now.
    if !int(
        "fn total(xs) {\n    \
             s = 0\n    \
             i = 0\n    \
             while (i < len(xs)) {\n        \
                 s = s + get(xs, i)\n        \
                 i = i + 1\n    \
             }\n    \
             return s\n\
         }\n\
         total(list(1, 2, 3, 4))",
        10,
    ) {
        return false;
    }
    // Comments and blank lines between statements, which any authored file has.
    if !int("// a program\n\nfn one() {\n  return 1\n}\n\n// call it\none()", 1) {
        return false;
    }

    // --- capabilities ---------------------------------------------------
    //
    // The operator keeps everything and a stored program does not. Each of
    // these is a way an application that was written by something other than a
    // person could reach past its own subtree, and each is refused rather than
    // merely discouraged: a check that produces a warning is a check that gets
    // ignored by whatever is generating the program.
    fn boxed(src: &str) -> Option<eval::Value> {
        let mut it = eval::Interp::sandboxed("/app/demo");
        eval_line(&mut it, src).ok()
    }
    // Raw memory and the I/O ports. A hallucinated port write on real hardware
    // is not something the machine recovers from.
    if boxed("poke64(4096, 1)").is_some() || boxed("peek32(0)").is_some() {
        return false;
    }
    if boxed("outb(112, 0)").is_some() || boxed("inb(112)").is_some() {
        return false;
    }
    // Drawing, which goes straight at the framebuffer with no window in the
    // way, so a sandboxed program could paint over the whole desktop.
    if boxed("rect(0, 0, 10, 10, 1)").is_some() || boxed("pixel(0, 0, 1)").is_some() {
        return false;
    }
    // ...and the operator still has all of it.
    if run("hex(255)").is_none() {
        return false;
    }
    // Writes outside the program's own subtree, including by the route every
    // jail is defeated through. The check runs on the *resolved* path, so
    // `..` is spent before the comparison rather than after it.
    if boxed("write(\"/ai/godel/HEAD\", \"x\")").is_some() {
        return false;
    }
    if boxed("write(\"/app/demo/../../ai/godel/HEAD\", \"x\")").is_some() {
        return false;
    }
    // A neighbour whose name merely begins the same way is outside the jail:
    // `/app/demo-evil` is not under `/app/demo`, and a prefix test alone would
    // have said it was.
    if boxed("write(\"/app/demo-evil/x\", \"y\")").is_some() {
        return false;
    }
    // Applets that change something. The flag consulted is the one
    // `harness::Trust::ReadOnly` filters the model's grammar with, so "safe to
    // call" has a single definition in this tree rather than two that drift.
    if boxed("applet(\"write /ai/godel/HEAD x\")").is_some() {
        return false;
    }
    if boxed("applet(\"rm /ai/tools/hello.l\")").is_some() {
        return false;
    }
    // `run` is classified mutating whatever the program text says, so a
    // sandboxed program cannot launder its way out through another one.
    if boxed("applet(\"run /ai/tools/hello.l\")").is_some() {
        return false;
    }
    if boxed("applet(\"nosuchapplet\")").is_some() {
        return false;
    }
    // Everything a sandbox is *allowed* to do needs a namespace, and this runs
    // before `sysbox::init`. Those are proved by the seeded application
    // instead, which writes its items under its own subtree through this same
    // gate every time a row is added -- an app that could not write would not
    // work at all, which is a louder failure than an assertion here.

    // A lowered budget stops a loop the full one would let run.
    //
    // The desktop calls an application's row function on every repaint, so a
    // generated loop that runs for a second is a window manager that feels
    // broken for reasons nobody can attribute. Twenty million steps is the
    // operator's budget, where a long loop is visible and can be stopped.
    {
        let mut it = eval::Interp::new().with_step_budget(1000);
        if eval_line(&mut it, "i = 0 while (i < 100000) { i = i + 1 } i").is_ok() {
            return false;
        }
        // ...and something short still finishes inside it.
        let mut it = eval::Interp::new().with_step_budget(1000);
        if !matches!(eval_line(&mut it, "i = 0 while (i < 10) { i = i + 1 } i"),
                     Ok(eval::Value::Int(10)))
        {
            return false;
        }
        // The budget cannot be raised past the default by asking.
        let mut it = eval::Interp::new().with_step_budget(u64::MAX);
        if eval_line(&mut it, "i = 0 while (i < 30000000) { i = i + 1 } i").is_ok() {
            return false;
        }
    }

    // Runaway recursion is refused rather than being allowed to eat the
    // kernel stack, which has no guard page and would triple fault.
    if run("fn boom(n) { return boom(n + 1) } boom(0)").is_some() {
        return false;
    }
    // The same hazard at parse time, which the call guard above does not
    // cover: the parser is recursive-descent, so nesting is stack depth, and
    // a deeply nested expression or block overflows before a single step
    // runs. Measured on this kernel: fine at 200 nested parens, a silent
    // triple fault before 300. These are past the cap and must come back as
    // an error rather than as a reboot -- built into strings here so the
    // source of this file is not itself a wall of brackets.
    //
    // Every mechanism that can build depth is here, because they overflow by
    // two different routes and a fix for one is not a fix for the other. Nested
    // parens and nested blocks overflow the *parser*; a flat operator chain, a
    // unary run and an assignment run build a tall *tree* that overflows later,
    // in eval or in drop. All must come back as an error, and this selftest
    // runs at boot -- so if any of them still faulted, the machine would not
    // reach a prompt to report it.
    {
        let n = 400;
        let cases = [
            alloc::format!("{}1{}", "(".repeat(n), ")".repeat(n)),   // parser
            alloc::format!("fn f() {{ {}1 {} }}", "if (1) { ".repeat(n), "}".repeat(n)),
            alloc::format!("1{}", "+1".repeat(n)),                    // flat tree
            alloc::format!("{}1", "-".repeat(n)),                    // unary run
            alloc::format!("{}1", "a=".repeat(n)),                   // assignment run
        ];
        for c in cases {
            if run(&c).is_some() {
                return false;
            }
        }
        // And the cap does not reject a program anybody would actually write.
        // Ten deep, and a chain of ten, are far past what real code has and
        // well under the bound.
        if !int("((((((((((7))))))))))", 7) || !int("1+1+1+1+1+1+1+1+1+1", 10) {
            return false;
        }
    }
    // Malformed input is an error, not a panic. This is also the gate that
    // makes generated programs safe to run at all.
    run("fn (").is_none() && run("return return").is_none()
}
