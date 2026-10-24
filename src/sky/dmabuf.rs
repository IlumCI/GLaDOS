//! `zwp_linux_dmabuf_v1`, as far as naming the GPU.
//!
//! **This compositor takes no dma-buf, and still has to speak the protocol.**
//! Mesa's EGL on Wayland finds the GPU to render with by asking the
//! compositor: version 4 of this interface sends a *default feedback* whose
//! `main_device` is a `dev_t`, and EGL opens that device's render node. With
//! no such global it decides there is no GPU at all and drops to software,
//! which for a GL program means llvmpipe however good the Vulkan driver
//! underneath is. So the global exists to say "the render node is
//! `/dev/dri/renderD128`", and Zink on that node then presents through Vulkan
//! WSI -- which, with `MESA_VK_WSI_DEBUG=sw`, blits into ordinary `wl_shm`
//! buffers this compositor already takes.
//!
//! Buffers made from dma-bufs are refused, through the protocol's own
//! `failed` event rather than an error: a client that tries is told no and
//! can fall back. Reading pixels out of a GPU buffer on the CPU is the
//! compositor work this does not do yet.
//!
//! The feedback is the formats and the device, and nothing else: one tranche,
//! ARGB8888 and XRGB8888, linear only.

use super::client::{done, Client};
use super::wire::{Error, Message, Writer};
use alloc::vec::Vec;

pub const DMABUF: &str = "zwp_linux_dmabuf_v1";
pub const FEEDBACK: &str = "zwp_linux_dmabuf_feedback_v1";
pub const PARAMS: &str = "zwp_linux_buffer_params_v1";

const ARGB8888: u32 = 0x3432_5241; // 'AR24'
const XRGB8888: u32 = 0x3432_5258; // 'XR24'
const MOD_LINEAR: u64 = 0;

pub fn handles(iface: &str) -> bool {
    matches!(iface, DMABUF | FEEDBACK | PARAMS)
}

/// Whether there is a GPU to name. Without one the global would send a client
/// to a render node that is not there.
pub fn offered() -> bool {
    crate::linux::drm::present().is_some()
}

/// The format table: 16 bytes per entry, a format, four bytes of padding, and
/// a modifier. Sent as a descriptor because the protocol says so -- a client
/// maps it -- and indexed by the tranche that follows.
fn table() -> Vec<u8> {
    let mut t = Vec::new();
    for f in [ARGB8888, XRGB8888] {
        t.extend_from_slice(&f.to_le_bytes());
        t.extend_from_slice(&0u32.to_le_bytes());
        t.extend_from_slice(&MOD_LINEAR.to_le_bytes());
    }
    t
}

impl Client {
    fn delete(&mut self, id: u32) -> Result<(), Error> {
        self.objects.destroy(id)?;
        let mut w = Writer::new(super::object::DISPLAY, 1);
        w.uint(id);
        self.post(w);
        Ok(())
    }

    /// Everything a feedback object says, in the order the protocol fixes:
    /// the table, the main device, one tranche, done.
    fn send_feedback(&mut self, id: u32) {
        let t = table();
        let Some(at) = crate::linux::syscall::alloc_pages(t.len()) else { return };
        unsafe { core::ptr::copy_nonoverlapping(t.as_ptr(), at as *mut u8, t.len()) };
        let fd = crate::linux::fs::Fd::Memfd(alloc::rc::Rc::new(core::cell::RefCell::new(
            crate::linux::fs::Memfd { name: alloc::string::String::from("dmabuf-formats"), at, len: t.len(), cursor: 0, maps: 0 },
        )));
        self.out_fds.push(fd);
        let mut w = Writer::new(id, 1); // format_table(fd, size)
        w.uint(t.len() as u32);
        self.post(w);
        // A `dev_t` is eight bytes for every libc this compositor serves,
        // i386 glibc included, and Mesa checks the array's length against it.
        let dev = crate::linux::drm::RDEV.to_le_bytes();
        let mut m = Writer::new(id, 2); // main_device
        m.array(&dev);
        self.post(m);
        let mut td = Writer::new(id, 4); // tranche_target_device
        td.array(&dev);
        self.post(td);
        let mut tf = Writer::new(id, 5); // tranche_formats: u16 indices
        tf.array(&[0, 0, 1, 0]);
        self.post(tf);
        let mut fl = Writer::new(id, 6); // tranche_flags
        fl.uint(0);
        self.post(fl);
        self.post(Writer::new(id, 3)); // tranche_done
        self.post(Writer::new(id, 0)); // done
    }

    /// What binding the global says. Below version 4 the formats go out as
    /// events; from 4 they are only in the feedback.
    pub(super) fn dmabuf_bound(&mut self, id: u32, iface: &str, version: u32) {
        if iface != DMABUF || version >= 4 {
            return;
        }
        for f in [ARGB8888, XRGB8888] {
            if version >= 3 {
                let mut w = Writer::new(id, 1); // modifier(format, hi, lo)
                w.uint(f).uint((MOD_LINEAR >> 32) as u32).uint(MOD_LINEAR as u32);
                self.post(w);
            } else {
                let mut w = Writer::new(id, 0); // format
                w.uint(f);
                self.post(w);
            }
        }
    }

    pub(super) fn dmabuf_request(&mut self, iface: &'static str, version: u32, msg: &Message) -> Result<(), Error> {
        let mut r = msg.reader();
        match (iface, msg.opcode) {
            (DMABUF, 0) | (FEEDBACK, 0) | (PARAMS, 0) => {
                done(&r)?;
                self.delete(msg.object)
            }
            (DMABUF, 1) => {
                let id = r.new_id()?;
                done(&r)?;
                self.objects.create(id, PARAMS, version)
            }
            (DMABUF, 2) | (DMABUF, 3) => {
                let id = r.new_id()?;
                if msg.opcode == 3 {
                    let _surface = r.object()?;
                }
                done(&r)?;
                self.objects.create(id, FEEDBACK, version)?;
                self.send_feedback(id);
                Ok(())
            }
            // add(fd, plane, offset, stride, modifier_hi, modifier_lo): the
            // descriptor is taken off the queue and dropped, since nothing
            // will be made of it.
            (PARAMS, 1) => {
                let _fd = r.fd(&mut self.fds)?;
                for _ in 0..5 {
                    r.uint()?;
                }
                done(&r)?;
                Ok(())
            }
            // create, and create_immed: refused with `failed`, which a client
            // is required to handle.
            (PARAMS, 2) | (PARAMS, 3) => {
                if msg.opcode == 3 {
                    let _buffer = r.new_id()?;
                }
                for _ in 0..4 {
                    r.uint()?;
                }
                done(&r)?;
                self.post(Writer::new(msg.object, 1));
                Ok(())
            }
            _ => Err(Error::NoMethod),
        }
    }
}

/// The table and the order of the feedback, with no client attached.
pub fn checks() -> Vec<(&'static str, bool)> {
    let t = table();
    alloc::vec![
        ("the format table is 16 bytes an entry, format first and modifier last", t.len() == 32 && t[0..4] == ARGB8888.to_le_bytes() && t[24..32] == MOD_LINEAR.to_le_bytes()),
        ("and the two entries are the formats wl_shm takes", u32::from_le_bytes([t[16], t[17], t[18], t[19]]) == XRGB8888),
    ]
}
