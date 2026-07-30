//! Physical memory and address translation.

pub mod frame;
pub mod heap;
pub mod paging;

pub const PAGE_SIZE: u64 = 4096;
pub const LARGE_PAGE_SIZE: u64 = 2 * 1024 * 1024;
pub const GIB: u64 = 1024 * 1024 * 1024;
