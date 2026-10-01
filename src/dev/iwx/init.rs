//! What the part needs told before it will answer a question about itself.
//!
//! **`NVM_GET_INFO` is not the first command, and sending it first was a defect
//! in this driver.** `iwx_run_init_mvm_ucode` reads the NVM only after a
//! four-step handshake, and each step is a thing firmware is waiting to be told:
//!
//! 1. the **platform-NVM** doorbell, on AX210 and only when the ALIVE
//!    notification carried a SKU id;
//! 2. `INIT_EXTENDED_CFG_CMD` with the `NVM` flag -- upstream's comment calls it
//!    "init config command to mark that we are sending NVM access commands";
//! 3. `NVM_ACCESS_COMPLETE`, which says there are no more of them;
//! 4. and a wait for `INIT_COMPLETE_NOTIF`, which is firmware saying it has
//!    finished with what it was told.
//!
//! Only then is the NVM readable. Asking before it is asking a part that has not
//! been told the question is coming.
//!
//! ### The platform NVM is owed even when there is no file
//!
//! This was recorded here as "no PNVM is needed for this part", which was half
//! right and the wrong half was load-bearing. There is no
//! `iwlwifi-so-a0-hr-b0.pnvm` in the firmware tree -- only the Gale Force
//! variants have one -- and the *handshake* happens anyway: upstream's own
//! comment is "if we don't have a platform NVM file simply ask firmware to
//! proceed without it", and it rings the doorbell and waits for the completion
//! either way. A driver that skipped it on the strength of the missing file would
//! wait forever at step 4 for a firmware still expecting step 1.
//!
//! What genuinely decides it is the **SKU id out of the ALIVE notification**: all
//! three words zero and the step is skipped entirely. That is why `Alive` carries
//! it.
//!
//! ### Provenance
//!
//! As the rest of `iwx`: numbers from OpenBSD's `iwx(4)`, Intel's dual BSD/GPLv2
//! headers underneath, BSD arm, `NOTICE.md`.

use alloc::string::String;
use alloc::vec::Vec;

use super::alive::{Alive, Buffers, Rx};
use super::cmd::{self, Queue, LONG_GROUP, REGULATORY_AND_NVM_GROUP};
use super::ctxt::Rings;

/// The group the configuration commands live in.
pub const SYSTEM_GROUP: u8 = 0x2;

pub const INIT_EXTENDED_CFG_CMD: u8 = 0x03;
pub const NVM_ACCESS_COMPLETE: u8 = 0x00;
/// Firmware's answer to the pair of them. In the legacy group, so `expect` has to
/// accept the long group beside it.
pub const INIT_COMPLETE_NOTIF: u8 = 0x04;
/// And to the platform-NVM doorbell, in the regulatory group.
pub const PNVM_INIT_COMPLETE: u8 = 0xfe;

/// "I am about to send NVM access commands." Bit 1 and not bit 0 -- the flag word
/// has other bits and `INIT_NVM` is the second of them, so a driver reaching for
/// the obvious 1 sets something else.
pub const INIT_NVM: u32 = 1 << 1;

/// Where the doorbell that asks firmware to proceed without a platform NVM lives,
/// in upper-MAC peripheral space.
pub const UREG_DOORBELL_TO_ISR6: u32 = 0xa0_5c04;
pub const DOORBELL_PNVM: u32 = 1 << 20;

/// One step of the handshake.
///
/// A table for `POWER_UP` and `KICK`'s reason: the ordering is the thing that is
/// unverifiable without the part and fatal when wrong, and as data a suite can
/// assert it on any machine. Here it is *more* than ordering -- the first version
/// of this driver had no steps at all and went straight to the question.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Step {
    /// Ring the doorbell that says "proceed without a platform NVM", under the
    /// MAC access lock, then wait for the completion. Skipped when the ALIVE
    /// notification carried no SKU id.
    Pnvm,
    /// Send a command with a four-byte payload and wait for nothing.
    Tell { group: u8, code: u8, flags: u32 },
    /// Wait for a notification.
    Await { group: u8, code: u8 },
}

/// The sequence, in order.
pub const HANDSHAKE: &[Step] = &[
    Step::Pnvm,
    Step::Tell { group: SYSTEM_GROUP, code: INIT_EXTENDED_CFG_CMD, flags: INIT_NVM },
    // Sent with a zero payload: the structure is one reserved word, and a command
    // with nothing to say still has to say four bytes of it.
    Step::Tell { group: REGULATORY_AND_NVM_GROUP, code: NVM_ACCESS_COMPLETE, flags: 0 },
    // **Both tells before either wait.** Upstream sends the pair and then waits
    // once; waiting after the first would wait for a completion firmware has not
    // been given the second half of.
    Step::Await { group: 0, code: INIT_COMPLETE_NOTIF },
];

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Fault {
    /// The MAC access lock was not granted, so the doorbell would have been
    /// dropped without a word.
    NoLock,
    /// A step's command or wait failed. Carries the index, so the transcript names
    /// which of the four rather than that one of them did.
    At(usize, cmd::CmdError),
}

impl Fault {
    pub fn why(&self) -> String {
        match self {
            Fault::NoLock => String::from(
                "the MAC access lock was not granted, so the platform-NVM doorbell would have been dropped",
            ),
            Fault::At(i, e) => match HANDSHAKE.get(*i) {
                Some(Step::Pnvm) => alloc::format!("the platform-NVM handshake did not complete: {}", e.why()),
                Some(Step::Tell { group, code, .. }) => alloc::format!(
                    "step {} (group {:#04x} code {:#04x}) was not accepted: {}",
                    i, group, code, e.why()
                ),
                Some(Step::Await { code, .. }) => {
                    alloc::format!("firmware never sent notification {:#04x}: {}", code, e.why())
                }
                None => alloc::format!("step {}: {}", i, e.why()),
            },
        }
    }
}

/// Run the handshake, so the part is ready to be asked about itself.
///
/// # Safety
/// `bar0` must be a mapped aperture for a part whose firmware is alive.
pub unsafe fn handshake(
    bar0: u64,
    rings: &mut Rings,
    bufs: &Buffers,
    rx: &mut Rx,
    q: &mut Queue,
    alive: &Alive,
    ms: u32,
) -> Result<(), Fault> {
    for (i, step) in HANDSHAKE.iter().enumerate() {
        match *step {
            Step::Pnvm => {
                // The one step that is conditional, and the condition is the SKU
                // id rather than whether a file exists.
                if !alive.wants_pnvm() {
                    continue;
                }
                if !super::gen3::lock(bar0) {
                    return Err(Fault::NoLock);
                }
                super::gen3::prph_write(bar0, super::gen3::umac_prph(UREG_DOORBELL_TO_ISR6), DOORBELL_PNVM);
                super::gen3::unlock(bar0);
                cmd::expect(bar0, rings, bufs, rx, REGULATORY_AND_NVM_GROUP, PNVM_INIT_COMPLETE, ms)
                    .map_err(|e| Fault::At(i, e))?;
            }
            Step::Tell { group, code, flags } => {
                // No reply is waited for here: the completion for the pair comes
                // once, at the last step.
                q.send(bar0, rings, group, code, 0, &flags.to_le_bytes())
                    .map_err(|e| Fault::At(i, e))?;
            }
            Step::Await { group, code } => {
                cmd::expect(bar0, rings, bufs, rx, group, code, ms).map_err(|e| Fault::At(i, e))?;
            }
        }
    }
    Ok(())
}

/// Claims. No radio, and no register written.
pub fn checks() -> Vec<(&'static str, bool)> {
    let mut out: Vec<(&'static str, bool)> = Vec::new();
    let mut ok = |c: bool, w: &'static str| out.push((w, c));

    let idx = |pred: fn(&Step) -> bool| HANDSHAKE.iter().position(pred);
    let pnvm = idx(|s| matches!(s, Step::Pnvm));
    let cfg = idx(|s| matches!(s, Step::Tell { code, .. } if *code == INIT_EXTENDED_CFG_CMD));
    let done = idx(|s| matches!(s, Step::Tell { code, .. } if *code == NVM_ACCESS_COMPLETE));
    let wait = idx(|s| matches!(s, Step::Await { code, .. } if *code == INIT_COMPLETE_NOTIF));

    // **The sequence exists at all**, which is the defect this table fixes: the
    // first version of the NVM path sent `NVM_GET_INFO` with none of this in front
    // of it.
    ok(HANDSHAKE.len() == 4, "the handshake is four steps");
    ok(pnvm == Some(0), "the platform-NVM doorbell comes first");
    ok(pnvm < cfg, "then the configuration command");
    ok(cfg < done, "then the one that says there are no more NVM accesses");
    // Both tells before either wait. Waiting after the first would wait for a
    // completion firmware has not been given the second half of.
    ok(done < wait, "and the wait is last, after both commands rather than between them");
    ok(
        HANDSHAKE.iter().filter(|s| matches!(s, Step::Await { .. })).count() == 1,
        "there is one wait and not one per command",
    );

    // The flag. Bit 1, not bit 0 -- the word has other bits and reaching for the
    // obvious 1 sets something else.
    ok(INIT_NVM == 2, "the NVM flag is bit one");
    ok(INIT_NVM != 1, "which is not the bit a reader would guess");
    ok(
        HANDSHAKE.iter().any(|s| matches!(s, Step::Tell { code, flags, .. } if *code == INIT_EXTENDED_CFG_CMD && *flags == INIT_NVM)),
        "and the configuration command carries it",
    );
    ok(
        HANDSHAKE.iter().any(|s| matches!(s, Step::Tell { code, flags, .. } if *code == NVM_ACCESS_COMPLETE && *flags == 0)),
        "where the completion command carries a zero word, being one reserved field",
    );

    // The groups, which are three different ones and easy to transpose.
    ok(SYSTEM_GROUP == 0x2, "the configuration command is in the system group");
    ok(REGULATORY_AND_NVM_GROUP == 0xc, "the NVM ones in the regulatory group");
    ok(
        HANDSHAKE.iter().any(|s| matches!(s, Step::Await { group: 0, .. })),
        "and the init-complete notification is in the legacy group",
    );
    ok(SYSTEM_GROUP != REGULATORY_AND_NVM_GROUP, "so the two commands do not share a group");
    // `expect` accepts the long group where zero is asked for, because `rx_pkt`
    // normalises one to the other. Asserted here because the alternative is a wait
    // that never ends on a packet that did arrive.
    ok(LONG_GROUP == 1, "the long group is 1, which a legacy notification can arrive as");

    // The doorbell.
    ok(UREG_DOORBELL_TO_ISR6 == 0xa0_5c04, "the doorbell is at 0xa05c04");
    ok(DOORBELL_PNVM == 1 << 20, "and the platform-NVM bit is 20");
    ok(
        super::gen3::umac_prph(UREG_DOORBELL_TO_ISR6) == 0xd0_5c04,
        "reached through upper-MAC space, like the boot doorbell",
    );
    // A different doorbell from the one that starts the firmware, and 0x40 apart
    // -- close enough to transpose and both legal.
    ok(
        UREG_DOORBELL_TO_ISR6 != 0xa0_5c44,
        "and it is not the boot doorbell, which is sixty-four bytes further on",
    );

    // The conditional step, whose condition is the SKU id and not a file.
    let mut a = super::alive::Alive {
        version: 5, status: super::alive::ALIVE_STATUS_OK, flags: 0,
        ucode_major: 0, ucode_minor: 0, umac_major: 0, umac_minor: 0,
        lmac_error_table: [0; 2], umac_error_table: 0, log_event_table: 0,
        sku_id: [0; 3],
    };
    ok(!a.wants_pnvm(), "a part with no SKU id owes no platform-NVM handshake");
    a.sku_id = [1, 0, 0];
    ok(a.wants_pnvm(), "and one word of it is enough to owe one");

    out
}
