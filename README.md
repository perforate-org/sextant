# sextant-index

The physical layer of a vector index: a page format for fixed-stride rows on paged memory, and
distance kernels over the rows. It keeps three promises:

1. **Self-describing pages.** A slab is a 32-byte header and pages appended at its tail. Each
   page's header fixes its row capacity and strides, so pages of one slab can differ, and
   `PageLayout` derives every table's place with checked arithmetic. Vectors start 16-byte
   aligned; an optional per-row code table (RaBitQ codes, say) follows them.
2. **One answer everywhere.** The kernels (`F32Row`, `I8Row`, `U8Row`) compute squared L2 and
   dot products, with early exits for pruned search. They use wasm SIMD where `simd128` is
   enabled and four scalar lanes elsewhere, summing in the same order, so a distance is bitwise
   the same on every target.
3. **No panics.** Headers are validated on decode, ranges past a page's capacity are `None`, and
   a short row or query is scored over the shorter of the two.

The crate knows storage facts only. Strides and aux widths arrive as parameters; partitions,
centroids and search policy belong to the layer above. `Memory` abstracts the paged memory, with
an in-process `VecMemory` and, behind the `ic-stable-structures` feature, every
`ic_stable_structures::Memory`. The crate is `no_std` + `alloc`.

## Use

```rust
use sextant_index::{
    F32Row, MetaStride, PAGE_HEADER_SIZE, PageHeader, PageLayout, RowMeta, RowPayload, Slab,
    VecMemory,
};

// A page of 2 rows of 4 f32s (16 bytes), bare metas and one run.
let header = PageHeader::new(2, 16, MetaStride::Bare, 1)?;
let layout = PageLayout::new(&header)?;
let mut page = vec![0; layout.page_len()];
page[..PAGE_HEADER_SIZE].copy_from_slice(&header.to_bytes());
for (row, values) in [[1.0f32, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]].iter().enumerate() {
    let row = u32::try_from(row)?;
    let id = RowPayload::new(10 + row).ok_or("id out of range")?;
    let meta = layout.row_meta_range_at(row).ok_or("no such row")?;
    RowMeta::new(id, [0; 8]).write_into(&mut page[meta], MetaStride::Bare)?;
    let vector = layout.vector_range_at(row).ok_or("no such row")?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    page[vector].copy_from_slice(&bytes);
}

// Pages append at the slab's tail; the offset locates the page later.
let memory = VecMemory::new();
let mut slab = Slab::open_or_init(&memory)?;
let offset = slab.append_page(&page)?;
assert_eq!(offset, 32);

// Search reads a row's vector bytes and scores them against the query.
let row = &page[layout.vector_range_at(1).ok_or("no such row")?];
assert_eq!(F32Row::new(row).l2_squared(&[0.0, 1.0, 0.0, 0.0]), 1.0);
```

## Where things are written down

The code is the one source of truth. Each module's documentation (`cargo doc
--document-private-items`) draws its byte layout and gives the reasons for its choices;
`crates/sextant-index/tests/public-api.txt` pins the public API, and `tests/page.rs` pins the
header bytes.

## Development

`just gate` runs fmt, clippy, tests, docs, deslop-please and qlty; `just kani` the proofs and
`just mutants` mutation testing.

## Origin and license

Extracted from [Gleaph](https://github.com/perforate-org/gleaph)'s vector canister. The page
and slab layout is unchanged, so slabs written by either read with the other. Kernel results
match Gleaph's wasm SIMD path bit for bit; its old scalar path summed in another order, so
host results can differ from it in the last bits.

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
