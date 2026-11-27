//! Page geometry: where each table of a page lies.
//!
//! ```text
//! [PageHeader] [run_table × run_capacity] [row_meta × capacity] (pad to 16)
//! [vector_bytes × capacity] [code_bytes × capacity]
//! ```
//!
//! The vector table starts on a 16-byte boundary and its stride is a multiple of 16, so every
//! row's vector is 16-byte aligned; the code table, present when the header's `code_stride` is
//! non-zero, follows it.

use crate::error::HeaderError;
use crate::header::{PAGE_HEADER_SIZE, PageHeader};
use crate::run::RunEntry;
use core::ops::Range;

/// The byte geometry of one page, derived from its header with checked arithmetic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageLayout {
    run_table_end: usize,
    meta_stride: usize,
    row_stride: usize,
    vector_bytes_offset: usize,
    code_offset: usize,
    code_stride: usize,
    capacity: u32,
    page_len: usize,
}

impl PageLayout {
    /// The geometry of a page with `header`.
    ///
    /// # Errors
    ///
    /// [`HeaderError::SpanOverflow`] when the page does not fit the address space.
    pub fn new(header: &PageHeader) -> Result<Self, HeaderError> {
        let size = |value: u32| usize::try_from(value).map_err(|_| HeaderError::SpanOverflow);
        let capacity = size(header.capacity())?;
        let meta_stride = size(header.meta_stride().bytes())?;
        let row_stride = size(header.row_stride())?;
        let code_stride = size(header.code_stride())?;
        let table = |stride: usize| {
            capacity
                .checked_mul(stride)
                .ok_or(HeaderError::SpanOverflow)
        };
        let run_table_end = size(header.run_capacity())?
            .checked_mul(RunEntry::SIZE)
            .and_then(|len| len.checked_add(PAGE_HEADER_SIZE))
            .ok_or(HeaderError::SpanOverflow)?;
        let vector_bytes_offset = run_table_end
            .checked_add(table(meta_stride)?)
            .and_then(|end| end.checked_next_multiple_of(16))
            .ok_or(HeaderError::SpanOverflow)?;
        let code_offset = vector_bytes_offset
            .checked_add(table(row_stride)?)
            .ok_or(HeaderError::SpanOverflow)?;
        let page_len = code_offset
            .checked_add(table(code_stride)?)
            .ok_or(HeaderError::SpanOverflow)?;
        Ok(Self {
            run_table_end,
            meta_stride,
            row_stride,
            vector_bytes_offset,
            code_offset,
            code_stride,
            capacity: header.capacity(),
            page_len,
        })
    }

    /// The page's length in bytes.
    #[must_use]
    pub const fn page_len(&self) -> usize {
        self.page_len
    }

    /// Where the vector table starts (a multiple of 16).
    #[must_use]
    pub const fn vector_bytes_offset(&self) -> usize {
        self.vector_bytes_offset
    }

    /// Vector bytes per row.
    #[must_use]
    pub const fn vector_stride(&self) -> usize {
        self.row_stride
    }

    /// Meta bytes per row.
    #[must_use]
    pub const fn meta_stride(&self) -> usize {
        self.meta_stride
    }

    /// Code-segment bytes per row (0 without a code table).
    #[must_use]
    pub const fn code_stride(&self) -> usize {
        self.code_stride
    }

    /// The run table.
    #[must_use]
    pub const fn run_table_range(&self) -> Range<usize> {
        PAGE_HEADER_SIZE..self.run_table_end
    }

    /// The row-meta table, with its padding.
    #[must_use]
    pub const fn row_meta_range(&self) -> Range<usize> {
        self.run_table_end..self.vector_bytes_offset
    }

    /// Row `row`'s meta entry; `None` past the capacity.
    #[must_use]
    pub fn row_meta_range_at(&self, row: u32) -> Option<Range<usize>> {
        self.entry(self.run_table_end, self.meta_stride, row)
    }

    /// Row `row`'s vector bytes; `None` past the capacity.
    #[must_use]
    pub fn vector_range_at(&self, row: u32) -> Option<Range<usize>> {
        self.entry(self.vector_bytes_offset, self.row_stride, row)
    }

    /// Row `row`'s code segment (empty without a code table); `None` past the capacity.
    #[must_use]
    pub fn code_range_at(&self, row: u32) -> Option<Range<usize>> {
        self.entry(self.code_offset, self.code_stride, row)
    }

    fn entry(&self, table: usize, stride: usize, row: u32) -> Option<Range<usize>> {
        if row >= self.capacity {
            return None;
        }
        let start = usize::try_from(row)
            .ok()?
            .checked_mul(stride)?
            .checked_add(table)?;
        Some(start..start.checked_add(stride)?)
    }

    /// The largest capacity whose page fits `max_page_bytes` with this geometry, or `None` when
    /// not even one row fits or the geometry is invalid. Page length grows strictly with the
    /// capacity, so a binary search over [`Self::new`] finds it exactly.
    #[must_use]
    pub fn max_capacity_for(
        max_page_bytes: usize,
        row_stride: u32,
        meta_stride: crate::MetaStride,
        run_capacity: u32,
        code_stride: u32,
    ) -> Option<u32> {
        let fits = |capacity: u32| {
            PageHeader::with_code_stride(
                capacity,
                row_stride,
                meta_stride,
                run_capacity,
                code_stride,
            )
            .and_then(|header| Self::new(&header))
            .is_ok_and(|layout| layout.page_len() <= max_page_bytes)
        };
        if !fits(1) {
            return None;
        }
        // Every row costs at least its strides, which bounds the search.
        let per_row = u64::from(meta_stride.bytes())
            .saturating_add(u64::from(row_stride))
            .saturating_add(u64::from(code_stride));
        let budget = u64::try_from(max_page_bytes).unwrap_or(u64::MAX);
        let upper = budget.checked_div(per_row).unwrap_or(0);
        let (mut lo, mut hi) = (1u32, u32::try_from(upper).unwrap_or(u32::MAX).max(1));
        // Invariant: `fits(lo)`; the answer lies in `lo..=hi`.
        while lo < hi {
            let mid = lo.midpoint(hi).saturating_add(1).min(hi);
            if fits(mid) {
                lo = mid;
            } else {
                hi = mid.saturating_sub(1);
            }
        }
        Some(lo)
    }
}

/// Proofs over every input (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::*;
    use crate::MetaStride;

    /// Every row's tables of a valid page lie inside it, the vectors 16-byte aligned. Bounded
    /// to small geometries: symbolic 64-bit products are beyond the solver, and the proptests
    /// cover the large ones.
    #[kani::proof]
    fn rows_lie_inside_their_page() {
        let stride = match kani::any::<u8>() % 3 {
            0 => MetaStride::Bare,
            1 => MetaStride::Aux4,
            _ => MetaStride::Aux8,
        };
        let small = |bound: u8| u32::from(kani::any::<u8>() % bound);
        let Ok(header) =
            PageHeader::with_code_stride(small(9), small(5) * 16, stride, small(5), small(5) * 4)
        else {
            return;
        };
        let Ok(layout) = PageLayout::new(&header) else {
            return;
        };
        let row: u32 = kani::any();
        kani::assume(row < header.capacity());
        let vector = layout.vector_range_at(row).unwrap();
        assert!(vector.start % 16 == 0);
        assert!(layout.row_meta_range_at(row).unwrap().end <= layout.vector_bytes_offset());
        assert!(vector.end <= layout.code_range_at(row).unwrap().start);
        assert!(layout.code_range_at(row).unwrap().end <= layout.page_len());
    }
}
