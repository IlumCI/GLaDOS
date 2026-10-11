//! `GLADOSV1`, the voice model file: a table of named tensors over
//! 64-byte-aligned data, written by `tools/kokoro.py`.
//!
//! Named, where the language model's checkpoint is indexed by arithmetic,
//! because this one is five hundred and fifty tensors in no regular shape:
//! a reader that walked them by position would be one miscount away from
//! loading every tensor after the mistake as perfectly valid garbage. The
//! names cost 20 KB. A tensor is looked up once at load, with its shape
//! checked, and a wrong shape is a refusal naming the tensor.
//!
//! Everything is a view into the blob -- read into a LoaderData pool before
//! ExitBootServices, the way the checkpoint is -- so nothing is copied and
//! the 170 MB costs no heap. The f32 views are cast, which is sound because
//! the writer aligns every tensor to 64 bytes from a data start that is
//! itself 64-aligned, and the pool allocation is page-aligned; `parse`
//! checks the base alignment rather than trusting it.

use crate::ai::weights::Mat;
use alloc::vec::Vec;

const MAGIC: &[u8; 8] = b"GLADOSV1";
const ALIGN: usize = 64;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    Q8,
    Bytes,
}

pub struct Entry {
    pub name: &'static str,
    pub dtype: Dtype,
    pub dims: [usize; 4],
    pub nd: usize,
    pub data: &'static [u8],
}

impl Entry {
    pub fn rows(&self) -> usize {
        self.dims[0]
    }
    pub fn cols(&self) -> usize {
        self.dims[1..self.nd].iter().product::<usize>().max(1)
    }
    pub fn len(&self) -> usize {
        self.dims[..self.nd].iter().product()
    }
}

pub struct VoiceFile {
    entries: Vec<Entry>,
}

fn u16_at(b: &[u8], o: usize) -> Option<u16> {
    Some(u16::from_le_bytes([*b.get(o)?, *b.get(o + 1)?]))
}
fn u32_at(b: &[u8], o: usize) -> Option<u32> {
    Some(u32::from_le_bytes([*b.get(o)?, *b.get(o + 1)?, *b.get(o + 2)?, *b.get(o + 3)?]))
}
fn u64_at(b: &[u8], o: usize) -> Option<u64> {
    let lo = u32_at(b, o)? as u64;
    let hi = u32_at(b, o + 4)? as u64;
    Some(lo | (hi << 32))
}

impl VoiceFile {
    pub fn parse(b: &'static [u8]) -> Result<VoiceFile, &'static str> {
        if b.len() < 16 || &b[..8] != MAGIC {
            return Err("not a GLADOSV1 file");
        }
        if (b.as_ptr() as usize) % 8 != 0 {
            return Err("the voice blob is not 8-byte aligned, so its f32 tensors cannot be viewed in place");
        }
        if u32_at(b, 8) != Some(1) {
            return Err("GLADOSV1 version is not 1");
        }
        let count = u32_at(b, 12).ok_or("short header")? as usize;
        let mut p = 16;
        let mut raw = Vec::with_capacity(count);
        for _ in 0..count {
            let nl = u16_at(b, p).ok_or("short table")? as usize;
            p += 2;
            let name = core::str::from_utf8(b.get(p..p + nl).ok_or("short table")?).map_err(|_| "a tensor name is not utf-8")?;
            p += nl;
            let dtype = match b.get(p).copied() {
                Some(0) => Dtype::F32,
                Some(1) => Dtype::Q8,
                Some(2) => Dtype::Bytes,
                _ => return Err("unknown tensor dtype"),
            };
            let nd = *b.get(p + 1).ok_or("short table")? as usize;
            p += 2;
            if nd > 4 {
                return Err("a tensor has more than four dims");
            }
            let mut dims = [1usize; 4];
            for d in dims.iter_mut().take(nd) {
                *d = u32_at(b, p).ok_or("short table")? as usize;
                p += 4;
            }
            let off = u64_at(b, p).ok_or("short table")? as usize;
            let n = u64_at(b, p + 8).ok_or("short table")? as usize;
            p += 16;
            raw.push((name, dtype, dims, nd, off, n));
        }
        let data_start = (p + ALIGN - 1) / ALIGN * ALIGN;
        let mut entries = Vec::with_capacity(count);
        for (name, dtype, dims, nd, off, n) in raw {
            let start = data_start.checked_add(off).ok_or("tensor offset overflows")?;
            let end = start.checked_add(n).ok_or("tensor length overflows")?;
            let data = b.get(start..end).ok_or("a tensor runs past the end of the file")?;
            let want = match dtype {
                Dtype::F32 => dims[..nd].iter().product::<usize>() * 4,
                Dtype::Q8 => dims[..nd].iter().product::<usize>() + dims[0] * 4,
                Dtype::Bytes => n,
            };
            if want != n {
                return Err("a tensor's byte length disagrees with its dims");
            }
            entries.push(Entry { name, dtype, dims, nd, data });
        }
        Ok(VoiceFile { entries })
    }

    pub fn get(&self, name: &str) -> Option<&Entry> {
        self.entries.iter().find(|e| e.name == name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// An f32 tensor as a slice, viewed in place.
    pub fn f32(&self, name: &str) -> Result<&'static [f32], &'static str> {
        let e = self.get(name).ok_or("missing tensor")?;
        if e.dtype != Dtype::F32 {
            return Err("not an f32 tensor");
        }
        Ok(view_f32(e.data))
    }

    /// An f32 tensor whose shape is asserted; the error names nothing, so
    /// callers wrap it with the tensor's name.
    pub fn f32_shaped(&self, name: &str, dims: &[usize]) -> Result<&'static [f32], &'static str> {
        let e = self.get(name).ok_or("missing tensor")?;
        if e.dtype != Dtype::F32 || e.nd != dims.len() || e.dims[..e.nd] != *dims {
            return Err("wrong shape or dtype");
        }
        Ok(view_f32(e.data))
    }

    /// A matrix as the kernel's `Mat`: int8 or f32 as the file has it,
    /// `rows` by `cols` (every dim after the first folded into cols).
    pub fn mat(&self, name: &str, rows: usize, cols: usize) -> Result<Mat<'static>, &'static str> {
        let e = self.get(name).ok_or("missing tensor")?;
        if e.rows() != rows || e.cols() != cols {
            return Err("wrong shape");
        }
        match e.dtype {
            Dtype::F32 => Ok(Mat::F32 { data: view_f32(e.data), rows, cols }),
            Dtype::Q8 => {
                let (q, s) = e.data.split_at(rows * cols);
                let q = unsafe { core::slice::from_raw_parts(q.as_ptr() as *const i8, q.len()) };
                Ok(Mat::Q8 { data: q, scales: s, rows, cols })
            }
            Dtype::Bytes => Err("a byte blob is not a matrix"),
        }
    }

    pub fn bytes(&self, name: &str) -> Option<&'static [u8]> {
        let e = self.get(name)?;
        (e.dtype == Dtype::Bytes).then_some(e.data)
    }
}

fn view_f32(b: &'static [u8]) -> &'static [f32] {
    debug_assert_eq!((b.as_ptr() as usize) % 4, 0);
    unsafe { core::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
}

pub fn checks() -> Vec<(&'static str, bool)> {
    // A file written by hand, the way the writer writes one: two tensors,
    // one q8, one blob, and the data start rounded up to 64.
    fn build() -> Vec<u8> {
        let mut t = Vec::new();
        t.extend_from_slice(MAGIC);
        t.extend_from_slice(&1u32.to_le_bytes());
        t.extend_from_slice(&3u32.to_le_bytes());
        let mut data: Vec<u8> = Vec::new();
        let entry = |t: &mut Vec<u8>, name: &str, dtype: u8, dims: &[u32], bytes: &[u8], data: &mut Vec<u8>| {
            while data.len() % ALIGN != 0 {
                data.push(0);
            }
            let off = data.len();
            data.extend_from_slice(bytes);
            t.extend_from_slice(&(name.len() as u16).to_le_bytes());
            t.extend_from_slice(name.as_bytes());
            t.push(dtype);
            t.push(dims.len() as u8);
            for d in dims {
                t.extend_from_slice(&d.to_le_bytes());
            }
            t.extend_from_slice(&(off as u64).to_le_bytes());
            t.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
        };
        let a: Vec<u8> = [1.5f32, -2.0, 0.25, 8.0, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        entry(&mut t, "a", 0, &[2, 3], &a, &mut data);
        // q8 2x2: rows [127, -127] scale 0.5 -> [63.5, -63.5]; [0, 64] scale 2 -> [0, 128]
        let mut q = alloc::vec![127u8, 129, 0, 64];
        q.extend_from_slice(&0.5f32.to_le_bytes());
        q.extend_from_slice(&2.0f32.to_le_bytes());
        entry(&mut t, "q", 1, &[2, 2], &q, &mut data);
        entry(&mut t, "blob", 2, &[5], b"hello", &mut data);
        while t.len() % ALIGN != 0 {
            t.push(0);
        }
        t.extend_from_slice(&data);
        t
    }
    let file: &'static [u8] = {
        // 64-aligned by allocating a Vec<u64> and leaking it -- the claims
        // are about parsing, and a misaligned test blob would fail the
        // alignment refusal before it reached them.
        let v = build();
        let mut aligned: Vec<u64> = alloc::vec![0; (v.len() + 7) / 8];
        unsafe { core::ptr::copy_nonoverlapping(v.as_ptr(), aligned.as_mut_ptr() as *mut u8, v.len()) };
        let len = v.len();
        let leaked: &'static mut [u64] = aligned.leak();
        unsafe { core::slice::from_raw_parts(leaked.as_ptr() as *const u8, len) }
    };
    let parsed = VoiceFile::parse(file);
    let ok = parsed.is_ok();
    let f = parsed.ok();
    let a = f.as_ref().and_then(|f| f.f32_shaped("a", &[2, 3]).ok());
    let q = f.as_ref().and_then(|f| f.mat("q", 2, 2).ok());
    let mut out = [0f32; 2];
    if let Some(m) = &q {
        m.matvec(&mut out, &[1.0, 1.0]);
    }
    let mut bad = file.to_vec();
    bad[8] = 2;
    let bad: &'static [u8] = Vec::leak(bad);
    alloc::vec![
        ("a hand-written GLADOSV1 parses", ok),
        ("its f32 tensor reads in place with the written values", a.map(|a| a[0] == 1.5 && a[3] == 8.0 && a[5] == 1.0).unwrap_or(false)),
        ("its q8 tensor dequantises through the kernel's own Mat", q.is_some() && (out[0] - 0.0).abs() < 1e-5 && (out[1] - 128.0).abs() < 1e-5),
        ("its byte blob is handed back whole", f.as_ref().and_then(|f| f.bytes("blob")) == Some(b"hello")),
        ("a wrong shape is refused rather than reshaped", f.as_ref().map(|f| f.f32_shaped("a", &[3, 2]).is_err()).unwrap_or(false)),
        ("a missing name is refused", f.as_ref().map(|f| f.f32("nope").is_err()).unwrap_or(false)),
        ("a future version is refused", VoiceFile::parse(bad).is_err()),
    ]
}
