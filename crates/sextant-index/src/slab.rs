//! The row slab: a header and pages appended at its tail, in one memory.
//!
//! The slab keeps no page directory: whoever appends a page records the offset it returns. It
//! guarantees only that pages are appended at `occupied_tail` and that the tail is persisted
//! last, so a crash between writing a page and rewriting the header leaves a valid, smaller slab.

use crate::error::SlabError;
use crate::header::{SLAB_HEADER_SIZE, SlabHeader};
use crate::memory::{Memory, PAGE_SIZE};

/// The header's length as a memory offset.
const HEADER_END: u64 = 32;

/// A row slab over `memory`.
#[derive(Debug)]
pub struct Slab<'a, M> {
    memory: &'a M,
    occupied_tail: u64,
}

impl<'a, M: Memory> Slab<'a, M> {
    /// Opens the slab in `memory`, or formats one there when the memory is empty or its header
    /// bytes are all zero (a region grown but never written).
    ///
    /// # Errors
    ///
    /// [`SlabError`] for an invalid header, a tail outside the memory, or a memory that cannot
    /// grow to hold a fresh header.
    pub fn open_or_init(memory: &'a M) -> Result<Self, SlabError> {
        let mut bytes = [0; SLAB_HEADER_SIZE];
        let memory_len = memory.size().saturating_mul(PAGE_SIZE);
        if memory_len >= HEADER_END {
            memory.read(0, &mut bytes);
        }
        if bytes.iter().all(|&byte| byte == 0) {
            write_at(memory, 0, &SlabHeader::new(HEADER_END, 0).to_bytes())?;
            return Ok(Self {
                memory,
                occupied_tail: HEADER_END,
            });
        }
        let tail = SlabHeader::from_bytes(&bytes)?.occupied_tail();
        if !(HEADER_END..=memory_len).contains(&tail) {
            return Err(SlabError::TailOutOfBounds { tail, memory_len });
        }
        Ok(Self {
            memory,
            occupied_tail: tail,
        })
    }

    /// The slab's first unused byte: the end of the last page.
    #[must_use]
    pub const fn occupied_tail(&self) -> u64 {
        self.occupied_tail
    }

    /// The slab's current header.
    #[must_use]
    pub const fn header(&self) -> SlabHeader {
        SlabHeader::new(self.occupied_tail, 0)
    }

    /// Appends `page` at the tail and returns its offset. The page is written first and the
    /// header last; the tail moves only once both are written.
    ///
    /// # Errors
    ///
    /// [`SlabError`] when the memory cannot grow or the tail would overflow.
    pub fn append_page(&mut self, page: &[u8]) -> Result<u64, SlabError> {
        let offset = self.occupied_tail;
        let end = u64::try_from(page.len())
            .ok()
            .and_then(|len| offset.checked_add(len))
            .ok_or(SlabError::TailOverflow)?;
        write_at(self.memory, offset, page)?;
        write_at(self.memory, 0, &SlabHeader::new(end, 0).to_bytes())?;
        self.occupied_tail = end;
        Ok(offset)
    }
}

/// Writes `bytes` at `offset`, growing the memory first when it is too small.
fn write_at<M: Memory>(memory: &M, offset: u64, bytes: &[u8]) -> Result<(), SlabError> {
    let end = u64::try_from(bytes.len())
        .ok()
        .and_then(|len| offset.checked_add(len))
        .ok_or(SlabError::TailOverflow)?;
    let size = memory.size().saturating_mul(PAGE_SIZE);
    if let Some(missing) = end.checked_sub(size).filter(|&missing| missing > 0) {
        let pages = missing.div_ceil(PAGE_SIZE);
        let _previous = memory.grow(pages)?;
    }
    memory.write(offset, bytes);
    Ok(())
}
