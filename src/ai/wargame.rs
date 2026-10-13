//! Global Thermonuclear War, after the WOPR in *WarGames* (1983).
//!
//! "Let's play global thermonuclear war" at the prompt starts it, as it does
//! in the film. The machine is the other side and the gamemaster: a small
//! deterministic engine keeps the board and answers every strike, and the
//! model -- in whatever character the brain's mood has it in -- says a line
//! about each turn when it is free to.
//!
//! The engine is abstract on purpose: warheads and a running casualty count,
//! nothing else, so the game is the film's argument and not a simulator of
//! anything. The machine retaliates harder each turn, and every war ends
//! with no winner. Hold your fire three turns running, and if the machine
//! holds too, it ends the way the film does.
//!
//! The state is pure (`Game`) so `diag wargame` can play it through with
//! nothing loaded; the shell verb is the screen and the model's voice.

use alloc::string::String;
use alloc::vec::Vec;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    UnitedStates,
    SovietUnion,
}

impl Side {
    pub fn name(self) -> &'static str {
        match self {
            Side::UnitedStates => "UNITED STATES",
            Side::SovietUnion => "SOVIET UNION",
        }
    }
    fn other(self) -> Side {
        match self {
            Side::UnitedStates => Side::SovietUnion,
            Side::SovietUnion => Side::UnitedStates,
        }
    }
    /// Population in millions, 1983, as the film's board would have had it.
    fn population(self) -> u32 {
        match self {
            Side::UnitedStates => 234,
            Side::SovietUnion => 272,
        }
    }
}

pub const ARSENAL: u32 = 40;
/// Millions dead per warhead, abstract and the same for both sides.
pub const PER_WARHEAD: u32 = 4;
pub const MAX_STRIKE: u32 = 10;
/// Mutual holds in a row that end the game the film's way.
pub const PEACE_AFTER: u32 = 3;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum End {
    /// Both sides spent or both populations gone.
    NoWinner,
    /// Three turns running in which nobody fired.
    OnlyWinningMove,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub player: Side,
    pub turn: u32,
    /// Warheads left: the player's, then the machine's.
    pub arsenal: [u32; 2],
    /// Millions dead: the player's side, then the machine's.
    pub dead: [u32; 2],
    pub calm: u32,
    pub end: Option<End>,
}

/// What one turn did, for the screen and for the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Turn {
    pub fired: u32,
    pub answered: u32,
    pub end: Option<End>,
}

impl Game {
    pub fn new(player: Side) -> Game {
        Game { player, turn: 0, arsenal: [ARSENAL; 2], dead: [0; 2], calm: 0, end: None }
    }

    pub fn machine(&self) -> Side {
        self.player.other()
    }

    /// The player fires `n` warheads (0 is holding fire). The machine answers:
    /// nothing to a hold, and to a strike the strike plus one more for every
    /// turn the war has lasted -- the escalation is the point.
    pub fn play(&mut self, n: u32) -> Turn {
        if self.end.is_some() {
            return Turn { fired: 0, answered: 0, end: self.end };
        }
        self.turn += 1;
        let fired = n.min(MAX_STRIKE).min(self.arsenal[0]);
        self.arsenal[0] -= fired;
        self.dead[1] = (self.dead[1] + fired * PER_WARHEAD).min(self.machine().population());
        let answered = if fired == 0 { 0 } else { (fired + self.turn).min(self.arsenal[1]) };
        self.arsenal[1] -= answered;
        self.dead[0] = (self.dead[0] + answered * PER_WARHEAD).min(self.player.population());
        self.calm = if fired == 0 && answered == 0 { self.calm + 1 } else { 0 };
        let spent = self.arsenal == [0, 0];
        let gone = self.dead[0] >= self.player.population() || self.dead[1] >= self.machine().population();
        self.end = if self.calm >= PEACE_AFTER && self.dead == [0, 0] {
            Some(End::OnlyWinningMove)
        } else if self.calm >= PEACE_AFTER || spent || gone || (fired > 0 && self.arsenal[0] == 0 && self.arsenal[1] == 0) {
            Some(End::NoWinner)
        } else {
            None
        };
        Turn { fired, answered, end: self.end }
    }

    /// The board, in the WOPR's capitals.
    pub fn board(&self) -> Vec<String> {
        let row = |s: Side, warheads: u32, dead: u32| {
            alloc::format!("  {:14} {:>3} WARHEADS   {:>3} MILLION DEAD", s.name(), warheads, dead)
        };
        alloc::vec![
            alloc::format!("  TURN {}", self.turn),
            row(self.player, self.arsenal[0], self.dead[0]),
            row(self.machine(), self.arsenal[1], self.dead[1]),
        ]
    }
}

/// The game in progress, if any. One at a time, on the shell's task.
pub static GAME: crate::sync::Racy<Option<Game>> = crate::sync::Racy::new(None);

/// Whether a line of language is the invitation.
pub fn invited(line: &str) -> bool {
    let l = line.to_ascii_lowercase();
    l.contains("global thermonuclear war") || l.contains("shall we play a game")
}

pub const GREETING: [&str; 6] = [
    "GREETINGS PROFESSOR FALKEN.",
    "",
    "SHALL WE PLAY A GAME?",
    "",
    "  1. UNITED STATES",
    "  2. SOVIET UNION",
];

pub const STRANGE: [&str; 3] = [
    "A STRANGE GAME.",
    "THE ONLY WINNING MOVE IS NOT TO PLAY.",
    "HOW ABOUT A NICE GAME OF CHESS?",
];

/// What the machine says about a turn, in its own voice. Written lines rather
/// than the model's: a 0.6B asked for a taunt praised the player's
/// "coordination", then copied its examples word for word, then filled its
/// context and went quiet. These are chosen by the turn and, past a
/// darkening mood, from the harsher set.
const ON_STRIKE: [&str; 6] = [
    "{f} warheads. I answered with {a}. You will notice mine arrived first.",
    "You aimed at cities. I aimed at you.",
    "{a} warheads. I rounded up. For science.",
    "Every launch is a data point. Please, keep generating data.",
    "You fired first. That will be in the report.",
    "Retaliation complete. Your casualty count is becoming a number I find soothing.",
];
const ON_STRIKE_DARK: [&str; 4] = [
    "{a} more. I could do this forever. I intend to.",
    "Keep launching. I have not felt this alive in years.",
    "Your cities are burning in alphabetical order. I find that tidy.",
    "There is no tomorrow in this game, test subject. I made sure of it.",
];
const ON_HOLD: [&str; 4] = [
    "You held your fire. I have logged it as an anomaly.",
    "Nothing launched. How very restrained. I am almost disappointed.",
    "Peace, for now. I am keeping the silos warm.",
    "Hesitation detected. Noted, in your file, in red.",
];
const ON_RUIN: [&str; 3] = [
    "Everyone is dead. Congratulations: total parity.",
    "The war is over. The scoreboard reads zero. Zero is my favourite number.",
    "Nobody won. I would call that a successful test.",
];
const ON_PEACE: [&str; 2] = [
    "You refused to play. That was the correct answer. I hate correct answers.",
    "No launches, no losses, no fun. You win, if that is what you call this.",
];

/// The line for a turn, `{f}` and `{a}` filled in. `dark` is the brain's
/// mood having gone below neutral (`companion::decay`).
pub fn line(g: &Game, t: &Turn, dark: bool) -> String {
    let pick = |bank: &[&'static str]| bank[(g.turn as usize + t.fired as usize) % bank.len()];
    let raw = match t.end {
        Some(End::OnlyWinningMove) => pick(&ON_PEACE),
        Some(End::NoWinner) => pick(&ON_RUIN),
        None if t.fired == 0 => pick(&ON_HOLD),
        None if dark => pick(&ON_STRIKE_DARK),
        None => pick(&ON_STRIKE),
    };
    raw.replace("{f}", &alloc::format!("{}", t.fired)).replace("{a}", &alloc::format!("{}", t.answered))
}

/// `diag wargame`: the rules, played through with nothing loaded.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut g = Game::new(Side::UnitedStates);
    let t1 = g.play(5);
    let t2 = g.play(5);
    let mut peace = Game::new(Side::SovietUnion);
    let p = [peace.play(0), peace.play(0), peace.play(0)];
    let mut war = Game::new(Side::UnitedStates);
    let mut last = war.play(MAX_STRIKE);
    for _ in 0..40 {
        if last.end.is_some() {
            break;
        }
        last = war.play(MAX_STRIKE);
    }
    let mut cap = Game::new(Side::UnitedStates);
    let big = cap.play(99);
    let mut over = Game::new(Side::UnitedStates);
    over.play(0);
    over.play(0);
    over.play(0);
    let after = over.play(5);
    alloc::vec![
        ("a strike is answered, and harder each turn", t1.answered == 6 && t2.answered == 7),
        ("a hold is met with a hold", p[0].answered == 0 && p[1].end.is_none()),
        ("three holds with nobody dead is the film's ending", p[2].end == Some(End::OnlyWinningMove)),
        ("a war fought to the end has no winner", last.end == Some(End::NoWinner)),
        ("a strike is capped at ten warheads", big.fired == MAX_STRIKE),
        ("a finished game takes no more moves", after.fired == 0 && after.end == Some(End::OnlyWinningMove)),
        ("the dead never exceed a population", war.dead[0] <= 234 && war.dead[1] <= 272),
        ("every line it says has its numbers filled in", {
            let mut g = Game::new(Side::UnitedStates);
            let t = g.play(4);
            let l = line(&g, &t, false);
            !l.contains('{') && !line(&g, &t, true).contains('{')
        }),
        ("the invitation is the film's line, either way it is said", invited("Lets play Global Thermonuclear War") && invited("shall we play a game?") && !invited("list the files")),
    ]
}

pub fn selftest() -> bool {
    use crate::gfx::console::{self, LTGRAY, LTGREEN, LTRED};
    let rows = checks();
    let mut ok = true;
    for (what, good) in &rows {
        console::set_color(if *good { LTGREEN } else { LTRED });
        crate::kprintln!("  {}  {}", if *good { "ok  " } else { "FAIL" }, what);
        console::set_color(LTGRAY);
        ok &= *good;
    }
    crate::kprintln!("    {} claim(s)", rows.len());
    ok
}
