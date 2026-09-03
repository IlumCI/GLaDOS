//! What the player has, and what walking over something does to it.
//!
//! Ported from room4doom's `gameplay/src/player.rs` and the `touch_special`
//! half of `gameplay/src/thing/interact.rs`, cut to the part that has
//! somewhere to go: health, armour, ammo, keys and which weapons are owned.
//! Firing them is `shoot.rs`; drawing one is not written yet.
//!
//! ### The table is keyed by sprite name, and that is id's own choice
//!
//! A pickup is not identified by its doomednum here but by the four-character
//! name of the sprite it is wearing, which is what DOOM does and looks like a
//! mistake until the reason lands: a *dropped* weapon and a placed one are
//! different objects with different doomednums and the same sprite, and the
//! rule "walking over `SHOT` gives you a shotgun" is true of both. The name is
//! the identity of the pickup; the doomednum is the identity of the placement.
//!
//! It also means this table needs nothing from the map. `Obj` knows its state,
//! the state knows its sprite, and a WAD that adds a new thing wearing `MEDI`
//! is a medikit without anybody being told.
//!
//! ### What is deliberately absent
//!
//! No powerups. Invulnerability, berserk, invisibility, the radiation suit,
//! the computer map and light amplification are all *timers* on a player, and
//! a timer that ticks down while nothing reads it is worse than no timer: it
//! looks implemented. They arrive with the status bar that shows them.
//!
//! No `dropped` flag either. It changes how much ammo a weapon comes with, and
//! only a monster drops anything -- nothing on this machine can die yet, so
//! every pickup is a placed one and the flag would be a constant `false`
//! dressed as a parameter.

use super::info;

/// What a medikit will not take you past. Bonuses go to twice this, which is
/// the whole difference between them.
pub const MAX_HEALTH: i32 = 100;
/// What a bonus will not take you past.
pub const MAX_BONUS: i32 = 200;

/// How much one clip of each kind holds, and the most that can be carried.
///
/// Order is DOOM's: clip, shell, cell, missile. Which is *not* the order the
/// weapons come in, and reordering it to match would silently rewrite every
/// pickup in the game -- a box of rockets would hand over 200 bullets.
pub const CLIP: [u32; 4] = [10, 4, 20, 1];
pub const MAX_AMMO: [u32; 4] = [200, 50, 300, 50];

/// The six keys. Three colours in two forms, and a door accepts either form of
/// its colour.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Card {
    Blue,
    Yellow,
    Red,
    BlueSkull,
    YellowSkull,
    RedSkull,
}

/// A door's colour, which is a *pair* of keys rather than one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Colour {
    Blue,
    Yellow,
    Red,
}

/// The four kinds of ammunition.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Ammo {
    Clip,
    Shell,
    Cell,
    Missile,
}

/// The nine weapons, in the order the number keys select them.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Weapon {
    Fist,
    Pistol,
    Shotgun,
    Chaingun,
    Missile,
    Plasma,
    Bfg,
    Chainsaw,
    SuperShotgun,
}

impl Weapon {
    /// What it eats, if anything.
    pub fn ammo(self) -> Option<Ammo> {
        match self {
            Weapon::Fist | Weapon::Chainsaw => None,
            Weapon::Pistol | Weapon::Chaingun => Some(Ammo::Clip),
            Weapon::Shotgun | Weapon::SuperShotgun => Some(Ammo::Shell),
            Weapon::Plasma | Weapon::Bfg => Some(Ammo::Cell),
            Weapon::Missile => Some(Ammo::Missile),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Weapon::Fist => "fist",
            Weapon::Pistol => "pistol",
            Weapon::Shotgun => "shotgun",
            Weapon::Chaingun => "chaingun",
            Weapon::Missile => "rocket launcher",
            Weapon::Plasma => "plasma rifle",
            Weapon::Bfg => "BFG",
            Weapon::Chainsaw => "chainsaw",
            Weapon::SuperShotgun => "super shotgun",
        }
    }
}

/// Everything the player is carrying.
pub struct Status {
    pub health: i32,
    pub armour: i32,
    /// 1 for the green jacket, 2 for the blue. It decides how much damage the
    /// armour takes rather than how much of it there is.
    pub armour_type: i32,
    pub ammo: [u32; 4],
    pub max_ammo: [u32; 4],
    pub cards: [bool; 6],
    pub owned: [bool; 9],
    pub weapon: Weapon,
    pub backpack: bool,
    /// How many things have been picked up. A counter rather than a log,
    /// because what a headless run needs to report is that a pickup happened.
    pub picked: usize,
}

impl Default for Status {
    fn default() -> Status {
        Status::new()
    }
}

impl Status {
    /// How DOOM starts you: full health, no armour, a pistol and fifty
    /// bullets.
    pub fn new() -> Status {
        let mut owned = [false; 9];
        owned[Weapon::Fist as usize] = true;
        owned[Weapon::Pistol as usize] = true;
        Status {
            health: MAX_HEALTH,
            armour: 0,
            armour_type: 0,
            ammo: [50, 0, 0, 0],
            max_ammo: MAX_AMMO,
            cards: [false; 6],
            owned,
            weapon: Weapon::Pistol,
            backpack: false,
            picked: 0,
        }
    }

    /// Whether a door of this colour will open. Either form of the colour
    /// does, which is why a door asks for a `Colour` and not a `Card`.
    pub fn opens(&self, c: Colour) -> bool {
        match c {
            Colour::Blue => self.cards[Card::Blue as usize] || self.cards[Card::BlueSkull as usize],
            Colour::Yellow => {
                self.cards[Card::Yellow as usize] || self.cards[Card::YellowSkull as usize]
            }
            Colour::Red => self.cards[Card::Red as usize] || self.cards[Card::RedSkull as usize],
        }
    }

    /// How many keys are held, for a caller that wants to say so.
    pub fn keys(&self) -> usize {
        self.cards.iter().filter(|c| **c).count()
    }

    /// A medikit or stimpack. Refuses at full health, which is what makes a
    /// medikit stay on the floor until it is needed.
    pub fn give_body(&mut self, n: i32) -> bool {
        if self.health >= MAX_HEALTH {
            return false;
        }
        self.health = (self.health + n).min(MAX_HEALTH);
        true
    }

    /// A jacket. Refuses one no better than what is already worn -- and note
    /// it *replaces* rather than adds, so picking up a green jacket at 150
    /// blue armour is refused rather than taking you down to 100.
    pub fn give_armour(&mut self, kind: i32) -> bool {
        let hits = kind * 100;
        if self.armour >= hits {
            return false;
        }
        self.armour_type = kind;
        self.armour = hits;
        true
    }

    pub fn give_key(&mut self, card: Card) {
        self.cards[card as usize] = true;
    }

    /// `clips` of ammunition, or half a clip when it is zero -- which is what
    /// a dropped clip gives and is the only reason the zero case exists.
    ///
    /// Refuses when already at the maximum, so a box of shells left on the
    /// floor is a box of shells you can come back for.
    pub fn give_ammo(&mut self, a: Ammo, clips: u32) -> bool {
        let i = a as usize;
        if self.ammo[i] >= self.max_ammo[i] {
            return false;
        }
        let n = if clips != 0 { clips * CLIP[i] } else { CLIP[i] / 2 };
        self.ammo[i] = (self.ammo[i] + n).min(self.max_ammo[i]);
        true
    }

    /// A weapon, and the ammunition it comes with.
    ///
    /// Taken even when the weapon is already owned, because the ammunition is
    /// still worth having -- which is why this is an `||` and not an early
    /// return.
    pub fn give_weapon(&mut self, w: Weapon) -> bool {
        let got_ammo = match w.ammo() {
            Some(a) => self.give_ammo(a, 2),
            None => false,
        };
        let got_weapon = !self.owned[w as usize];
        if got_weapon {
            self.owned[w as usize] = true;
            self.weapon = w;
        }
        got_ammo || got_weapon
    }

    /// Walking over a thing wearing this sprite. True means it was taken and
    /// should come off the level.
    ///
    /// Everything that can refuse does. A refused pickup is not a failure --
    /// it is a medikit you do not need yet, and it has to stay where it is.
    pub fn touch(&mut self, sprite: &str) -> bool {
        let took = match sprite {
            // Armour.
            "ARM1" => self.give_armour(1),
            "ARM2" => self.give_armour(2),

            // Bonuses, which are the two things that go past 100.
            "BON1" => {
                self.health = (self.health + 1).min(MAX_BONUS);
                true
            }
            "BON2" => {
                self.armour = (self.armour + 1).min(MAX_BONUS);
                if self.armour_type == 0 {
                    self.armour_type = 1;
                }
                true
            }
            "SOUL" => {
                self.health = (self.health + 100).min(MAX_BONUS);
                true
            }
            "MEGA" => {
                self.health = MAX_BONUS;
                self.give_armour(2);
                true
            }

            // Keys. Always taken, whether or not one is already held: a
            // second blue key is still removed from the floor.
            "BKEY" => {
                self.give_key(Card::Blue);
                true
            }
            "YKEY" => {
                self.give_key(Card::Yellow);
                true
            }
            "RKEY" => {
                self.give_key(Card::Red);
                true
            }
            "BSKU" => {
                self.give_key(Card::BlueSkull);
                true
            }
            "YSKU" => {
                self.give_key(Card::YellowSkull);
                true
            }
            "RSKU" => {
                self.give_key(Card::RedSkull);
                true
            }

            // Health.
            "STIM" => self.give_body(10),
            "MEDI" => self.give_body(25),

            // Ammunition.
            "CLIP" => self.give_ammo(Ammo::Clip, 1),
            "AMMO" => self.give_ammo(Ammo::Clip, 5),
            "SHEL" => self.give_ammo(Ammo::Shell, 1),
            "SBOX" => self.give_ammo(Ammo::Shell, 5),
            "CELL" => self.give_ammo(Ammo::Cell, 1),
            "CELP" => self.give_ammo(Ammo::Cell, 5),
            "ROCK" => self.give_ammo(Ammo::Missile, 1),
            "BROK" => self.give_ammo(Ammo::Missile, 5),
            "BPAK" => {
                if !self.backpack {
                    for m in self.max_ammo.iter_mut() {
                        *m *= 2;
                    }
                    self.backpack = true;
                }
                // One clip of everything, and the backpack is taken whether or
                // not any of it fit.
                for i in 0..4 {
                    let a = match i {
                        0 => Ammo::Clip,
                        1 => Ammo::Shell,
                        2 => Ammo::Cell,
                        _ => Ammo::Missile,
                    };
                    self.give_ammo(a, 1);
                }
                true
            }

            // Weapons.
            "SHOT" => self.give_weapon(Weapon::Shotgun),
            "SGN2" => self.give_weapon(Weapon::SuperShotgun),
            "MGUN" => self.give_weapon(Weapon::Chaingun),
            "LAUN" => self.give_weapon(Weapon::Missile),
            "PLAS" => self.give_weapon(Weapon::Plasma),
            "BFUG" => self.give_weapon(Weapon::Bfg),
            "CSAW" => self.give_weapon(Weapon::Chainsaw),

            // The powerups, which are recognised and declined rather than
            // ignored: leaving them out of the match entirely would make a
            // radiation suit indistinguishable from a decoration, and the day
            // powerups arrive the compiler would not point here.
            "PINV" | "PSTR" | "PINS" | "SUIT" | "PMAP" | "PVIS" => false,

            _ => false,
        };
        if took {
            self.picked += 1;
        }
        took
    }
}

/// Which colour a manual door demands, if it demands one.
///
/// 26/32 blue, 27/34 yellow, 28/33 red. The pairs are the ordinary door and
/// the one that opens and stays open, which differ in everything except the
/// key they want.
pub fn locked(special: u16) -> Option<Colour> {
    match special {
        26 | 32 => Some(Colour::Blue),
        27 | 34 => Some(Colour::Yellow),
        28 | 33 => Some(Colour::Red),
        _ => None,
    }
}

/// What `diag doom` asks of the inventory.
pub fn checks() -> alloc::vec::Vec<(&'static str, bool)> {
    let mut out: alloc::vec::Vec<(&'static str, bool)> = alloc::vec::Vec::new();

    // A medikit refuses at full health and a bonus does not. That difference
    // is the whole of why there are two ceilings, and getting it wrong reads
    // as items that vanish for nothing.
    let mut s = Status::new();
    let full = !s.touch("MEDI");
    s.health = 90;
    let hurt = s.touch("MEDI") && s.health == MAX_HEALTH;
    let mut b = Status::new();
    let bonus = b.touch("BON1") && b.health == MAX_HEALTH + 1;
    out.push((
        "a medikit stops at 100 and a bonus does not",
        full && hurt && bonus,
    ));

    // A soulsphere goes to 200 and no further.
    let mut soul = Status::new();
    soul.touch("SOUL");
    out.push(("a soulsphere stops at 200", soul.health == MAX_BONUS));

    // Ammunition is clips, not rounds. A box of bullets is fifty and a box of
    // rockets is five, and the two tables being in different orders is
    // exactly how that goes wrong.
    let mut a = Status::new();
    let box_bullets = a.touch("AMMO") && a.ammo[Ammo::Clip as usize] == 100;
    let rockets = a.touch("BROK") && a.ammo[Ammo::Missile as usize] == 5;
    out.push(("a box of bullets is 50 and a box of rockets is 5", box_bullets && rockets));

    // Ammunition stops at the maximum, and a backpack raises it.
    let mut cap = Status::new();
    cap.ammo[Ammo::Clip as usize] = MAX_AMMO[Ammo::Clip as usize];
    let refused = !cap.touch("CLIP");
    cap.touch("BPAK");
    let raised = cap.max_ammo[Ammo::Clip as usize] == MAX_AMMO[Ammo::Clip as usize] * 2;
    out.push(("a full pocket refuses ammunition until a backpack", refused && raised));

    // A weapon is taken for its ammunition even when it is already owned,
    // which is why a shotgun on the floor is worth walking over twice.
    let mut w = Status::new();
    let first = w.touch("SHOT") && w.owned[Weapon::Shotgun as usize];
    let again = w.touch("SHOT");
    out.push(("a second shotgun is taken for the shells", first && again));

    // Either form of a colour opens its door, and neither opens another's.
    let mut k = Status::new();
    k.give_key(Card::BlueSkull);
    out.push((
        "a skull key opens the card's door",
        k.opens(Colour::Blue) && !k.opens(Colour::Red) && !k.opens(Colour::Yellow),
    ));

    // The specials a locked door wears, in both of their forms.
    out.push((
        "a locked door names its colour",
        matches!(locked(26), Some(Colour::Blue))
            && matches!(locked(32), Some(Colour::Blue))
            && matches!(locked(27), Some(Colour::Yellow))
            && matches!(locked(28), Some(Colour::Red))
            && locked(1).is_none(),
    ));

    // Every sprite the table names must be one the WAD could actually carry.
    // A typo here is a pickup that never fires, and nothing would ever say so.
    let named = [
        "ARM1", "ARM2", "BON1", "BON2", "SOUL", "MEGA", "BKEY", "YKEY", "RKEY", "BSKU", "YSKU",
        "RSKU", "STIM", "MEDI", "CLIP", "AMMO", "SHEL", "SBOX", "CELL", "CELP", "ROCK", "BROK",
        "BPAK", "SHOT", "SGN2", "MGUN", "LAUN", "PLAS", "BFUG", "CSAW", "PINV", "PSTR", "PINS",
        "SUIT", "PMAP", "PVIS",
    ];
    out.push((
        "every pickup names a sprite the game has",
        named.iter().all(|n| info::SPRITES.contains(n)),
    ));

    out
}
