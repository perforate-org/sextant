//! The memory seam: the one place a slab's bytes enter or leave.

use crate::error::GrowFailed;
use alloc::rc::Rc;
use alloc::vec::Vec;
use core::cell::RefCell;

/// Bytes per page; [`Memory::grow`] counts in pages.
pub const PAGE_SIZE: u64 = 65_536;

/// A growable, byte-addressed backend.
///
/// The shape mirrors `ic_stable_structures::Memory` (shared `&self`, interior mutability,
/// page-granular growth); with the `ic-stable-structures` feature every memory of that crate is
/// one.
pub trait Memory {
    /// Current size in pages.
    fn size(&self) -> u64;

    /// Grows by `pages` pages and returns the previous size in pages.
    ///
    /// # Errors
    ///
    /// [`GrowFailed`] when the backend cannot provide the pages; the size is then unchanged.
    fn grow(&self, pages: u64) -> Result<u64, GrowFailed>;

    /// Copies `buf.len()` bytes starting at `offset` into `buf`. The range lies within the
    /// current size.
    fn read(&self, offset: u64, buf: &mut [u8]);

    /// Copies `buf` to `offset`. The range lies within the current size.
    fn write(&self, offset: u64, buf: &[u8]);
}

/// Every ic-stable-structures memory is a sextant-index memory: `DefaultMemoryImpl`, a
/// `VirtualMemory` from a `MemoryManager`, a `VectorMemory`. Its `grow` reports failure as −1,
/// which becomes [`GrowFailed`].
#[cfg(feature = "ic-stable-structures")]
impl<M: ic_stable_structures::Memory> Memory for M {
    fn size(&self) -> u64 {
        ic_stable_structures::Memory::size(self)
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        let previous = ic_stable_structures::Memory::grow(self, pages);
        u64::try_from(previous)
            .map_err(|_| GrowFailed::new(ic_stable_structures::Memory::size(self), pages))
    }

    fn read(&self, offset: u64, buf: &mut [u8]) {
        ic_stable_structures::Memory::read(self, offset, buf);
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        ic_stable_structures::Memory::write(self, offset, buf);
    }
}

/// A heap-backed [`Memory`] for tests and in-process use.
///
/// Clones share the bytes, so a slab can be reopened from a clone. Out-of-range reads yield zeros
/// and out-of-range writes are dropped (the trait's callers never make them).
#[derive(Clone, Debug, Default)]
pub struct VecMemory(Rc<RefCell<Vec<u8>>>);

impl VecMemory {
    /// An empty memory of zero pages.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Memory for VecMemory {
    fn size(&self) -> u64 {
        u64::try_from(self.0.borrow().len())
            .unwrap_or(u64::MAX)
            .checked_div(PAGE_SIZE)
            .unwrap_or(0)
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        let previous = self.size();
        let len = previous
            .checked_add(pages)
            .and_then(|total| total.checked_mul(PAGE_SIZE))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| GrowFailed::new(previous, pages))?;
        self.0.borrow_mut().resize(len, 0);
        Ok(previous)
    }

    fn read(&self, offset: u64, buf: &mut [u8]) {
        let bytes = self.0.borrow();
        match window(offset, buf.len()).and_then(|range| bytes.get(range)) {
            Some(source) => buf.copy_from_slice(source),
            None => buf.fill(0),
        }
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        let mut bytes = self.0.borrow_mut();
        if let Some(target) = window(offset, buf.len()).and_then(|range| bytes.get_mut(range)) {
            target.copy_from_slice(buf);
        }
    }
}

fn window(offset: u64, len: usize) -> Option<core::ops::Range<usize>> {
    let start = usize::try_from(offset).ok()?;
    Some(start..start.checked_add(len)?)
}
