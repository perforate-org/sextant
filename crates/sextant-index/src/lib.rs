//! The physical layer of a vector index: a page format for fixed-stride rows on paged memory,
//! and distance kernels over the rows.
//!
//! A [`Slab`] is a header and pages appended at its tail. A page is self-describing: its
//! [`PageHeader`] fixes the row capacity and strides, and [`PageLayout`] derives where its run
//! table, row metas, vectors and optional per-row codes lie. Vectors start 16-byte aligned. A
//! row's meta holds its [`RowPayload`] (a 30-bit id and a tombstone bit) and opaque aux bytes;
//! the run table records the shard rows appended together belong to.
//!
//! The kernels ([`F32Row`], [`I8Row`], [`U8Row`]) compute squared L2 and dot products, with early
//! exits for pruned search, using wasm SIMD where it is enabled and giving bitwise the same
//! results everywhere.
//!
//! The crate knows storage facts only: strides and aux widths arrive as parameters, and
//! partitions, centroids and search policy belong to the layer above. Library code never panics;
//! every header is validated on decode. The crate is `no_std` + `alloc`.
//!
//! ```
//! use sextant_index::{
//!     F32Row, MetaStride, PAGE_HEADER_SIZE, PageHeader, PageLayout, RowMeta, RowPayload, Slab,
//!     VecMemory,
//! };
//!
//! // A page of 2 rows of 4 f32s (16 bytes), bare metas and one run.
//! let header = PageHeader::new(2, 16, MetaStride::Bare, 1)?;
//! let layout = PageLayout::new(&header)?;
//! let mut page = vec![0; layout.page_len()];
//! page[..PAGE_HEADER_SIZE].copy_from_slice(&header.to_bytes());
//! for (row, values) in [[1.0f32, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]].iter().enumerate() {
//!     let row = u32::try_from(row)?;
//!     let id = RowPayload::new(10 + row).ok_or("id out of range")?;
//!     let meta = layout.row_meta_range_at(row).ok_or("no such row")?;
//!     RowMeta::new(id, [0; 8]).write_into(&mut page[meta], MetaStride::Bare)?;
//!     let vector = layout.vector_range_at(row).ok_or("no such row")?;
//!     let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
//!     page[vector].copy_from_slice(&bytes);
//! }
//!
//! // Pages append at the slab's tail; the offset locates the page later.
//! let memory = VecMemory::new();
//! let mut slab = Slab::open_or_init(&memory)?;
//! let offset = slab.append_page(&page)?;
//! assert_eq!(offset, 32);
//!
//! // Search reads a row's vector bytes and scores them against the query.
//! let row = &page[layout.vector_range_at(1).ok_or("no such row")?];
//! assert_eq!(F32Row::new(row).l2_squared(&[0.0, 1.0, 0.0, 0.0]), 1.0);
//! # Ok::<(), Box<dyn core::error::Error>>(())
//! ```

#![no_std]

extern crate alloc;

// Layers, each using only those before it: `error`, then `run` and `header`, then `layout`, then
// `payload`, `memory` and `slab`. `kernel` stands alone.
mod error;
mod header;
mod kernel;
mod layout;
mod memory;
mod payload;
mod run;
mod slab;

pub use error::{GrowFailed, HeaderError, PayloadError, SlabError};
pub use header::{
    FORMAT_VERSION, MAGIC_PAGE, MAGIC_SLAB, MAX_RUNS, MetaStride, PAGE_HEADER_SIZE, PageHeader,
    SLAB_HEADER_SIZE, SlabHeader,
};
pub use kernel::{DotBound, F32Row, I8Row, U8Row, popcount_xnor_words};
pub use layout::PageLayout;
pub use memory::{Memory, PAGE_SIZE, VecMemory};
pub use payload::{MAX_ROW_ID, RowMeta, RowPayload};
pub use run::{RunEntry, read_run, write_run};
pub use slab::Slab;
