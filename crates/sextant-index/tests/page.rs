//! The page format: header bytes pinned, validation, geometry, row metas, the run table and the
//! slab over a memory.

#![expect(
    clippy::arithmetic_side_effects,
    reason = "test helpers over small known inputs"
)]

use proptest::prelude::*;
use sextant_index::{
    GrowFailed, HeaderError, MAX_ROW_ID, MAX_RUNS, Memory, MetaStride, PAGE_SIZE, PageHeader,
    PageLayout, PayloadError, RowMeta, RowPayload, RunEntry, Slab, SlabError, SlabHeader,
    VecMemory, read_run, write_run,
};

#[test]
fn header_bytes_are_pinned() {
    let mut header = PageHeader::with_code_stride(3, 32, MetaStride::Aux4, 2, 12).unwrap();
    header.set_run_count(1).unwrap();
    let bytes = header.to_bytes();
    let expected: Vec<u8> = [
        &b"VPG\x01"[..],
        &3u32.to_le_bytes(),
        &32u32.to_le_bytes(),
        &8u32.to_le_bytes(),
        &2u32.to_le_bytes(),
        &1u32.to_le_bytes(),
        &12u32.to_le_bytes(),
        &[0; 4],
    ]
    .concat();
    assert_eq!(bytes[..], expected[..]);
    assert_eq!(PageHeader::from_bytes(&bytes), Ok(header));
    let slab = SlabHeader::new(4096, 7);
    let bytes = slab.to_bytes();
    assert_eq!(bytes[..4], *b"VSL\x01");
    assert_eq!(bytes[4..12], 4096u64.to_le_bytes());
    assert_eq!(bytes[12..16], 7u32.to_le_bytes());
    assert_eq!(SlabHeader::from_bytes(&bytes), Ok(slab));
    assert_eq!((slab.occupied_tail(), slab.flags()), (4096, 7));
}

#[test]
fn headers_fail_closed() {
    let good = PageHeader::new(4, 16, MetaStride::Bare, 1)
        .unwrap()
        .to_bytes();
    let mut ascii = good;
    ascii[3] = b'1';
    assert_eq!(
        PageHeader::from_bytes(&ascii),
        Err(HeaderError::UnsupportedVersion { found: b'1' })
    );
    assert_eq!(
        PageHeader::from_bytes(&SlabHeader::new(32, 0).to_bytes()),
        Err(HeaderError::BadMagic {
            expected: *b"VPG",
            found: *b"VSL"
        })
    );
    let mut stride = good;
    stride[12..16].copy_from_slice(&6u32.to_le_bytes());
    assert_eq!(
        PageHeader::from_bytes(&stride),
        Err(HeaderError::InvalidMetaStride { found: 6 })
    );
    let mut runs = good;
    runs[20..24].copy_from_slice(&1u32.to_le_bytes());
    assert!(PageHeader::from_bytes(&runs).is_ok_and(|full| full.run_count() == 1));
    runs[20..24].copy_from_slice(&2u32.to_le_bytes());
    assert_eq!(
        PageHeader::from_bytes(&runs),
        Err(HeaderError::RunCountExceedsCapacity {
            run_count: 2,
            run_capacity: 1
        })
    );
    let bare = MetaStride::Bare;
    assert_eq!(
        PageHeader::new(0, 16, bare, 1),
        Err(HeaderError::ZeroCapacity)
    );
    assert_eq!(
        PageHeader::new(1, 24, bare, 1),
        Err(HeaderError::InvalidRowStride { found: 24 })
    );
    assert_eq!(
        PageHeader::new(1, 0, bare, 1),
        Err(HeaderError::InvalidRowStride { found: 0 })
    );
    assert_eq!(
        PageHeader::new(1, 16, bare, 0),
        Err(HeaderError::InvalidRunCapacity { found: 0 })
    );
    assert_eq!(
        PageHeader::new(1, 16, bare, MAX_RUNS + 1),
        Err(HeaderError::InvalidRunCapacity { found: 65 })
    );
    assert!(PageHeader::new(1, 16, bare, MAX_RUNS).is_ok());
    assert_eq!(
        PageHeader::with_code_stride(1, 16, bare, 1, 6),
        Err(HeaderError::InvalidCodeStride { found: 6 })
    );
    let mut header = PageHeader::new(1, 16, bare, 2).unwrap();
    assert_eq!(
        header.set_run_count(3),
        Err(HeaderError::RunCountExceedsCapacity {
            run_count: 3,
            run_capacity: 2
        })
    );
    assert!(header.set_run_count(2).is_ok());
    assert_eq!(header.run_count(), 2);
    for error in [HeaderError::ZeroCapacity, HeaderError::SpanOverflow] {
        assert_ne!(error.to_string(), "");
    }
}

#[test]
fn layout_places_every_table() {
    let header = PageHeader::with_code_stride(3, 32, MetaStride::Aux8, 2, 12).unwrap();
    let layout = PageLayout::new(&header).unwrap();
    assert_eq!(layout.run_table_range(), 32..48);
    assert_eq!(layout.row_meta_range(), 48..96);
    assert_eq!(layout.row_meta_range_at(1), Some(60..72));
    assert_eq!(layout.vector_bytes_offset(), 96);
    assert_eq!(layout.vector_range_at(2), Some(160..192));
    assert_eq!(layout.code_range_at(0), Some(192..204));
    assert_eq!(layout.page_len(), 228);
    assert_eq!(layout.vector_range_at(3), None);
    assert_eq!(
        (
            layout.vector_stride(),
            layout.meta_stride(),
            layout.code_stride()
        ),
        (32, 12, 12)
    );
    assert_eq!(
        PageLayout::max_capacity_for(228, 32, MetaStride::Aux8, 2, 12),
        Some(3)
    );
    assert_eq!(
        PageLayout::max_capacity_for(48, 32, MetaStride::Aux8, 2, 12),
        None
    );
}

#[test]
fn row_metas_and_payloads() {
    let payload = RowPayload::new(MAX_ROW_ID).unwrap();
    assert_eq!(RowPayload::new(MAX_ROW_ID + 1), None);
    assert!(!payload.is_tombstone());
    let dead = payload.tombstoned();
    assert!(dead.is_tombstone());
    assert_eq!(dead.id(), MAX_ROW_ID);
    assert_eq!(dead.tombstoned(), dead);
    assert_eq!(
        RowPayload::from_raw(1 << 30),
        Err(PayloadError::ReservedBitSet { raw: 1 << 30 })
    );
    let meta = RowMeta::new(dead, [1, 2, 3, 4, 5, 6, 7, 8]);
    for (stride, aux) in [
        (MetaStride::Bare, 0),
        (MetaStride::Aux4, 4),
        (MetaStride::Aux8, 8),
    ] {
        assert_eq!(MetaStride::for_aux(aux), Some(stride));
        assert_eq!(stride.aux_bytes(), aux);
        assert_eq!(MetaStride::from_bytes(stride.bytes()), Ok(stride));
        let mut out = vec![0; aux + 4];
        meta.write_into(&mut out, stride).unwrap();
        assert_eq!(out[..4], dead.raw().to_le_bytes());
        let back = RowMeta::from_bytes(&out, stride).unwrap();
        assert_eq!(back.payload(), dead);
        assert_eq!(back.aux(stride), &meta.aux(stride)[..aux]);
        assert_eq!(
            RowMeta::from_bytes(&out[1..], stride),
            Err(PayloadError::BadSliceLength {
                expected: aux + 4,
                found: aux + 3
            })
        );
        assert_eq!(
            meta.write_into(&mut [0; 13], stride),
            Err(PayloadError::BadSliceLength {
                expected: aux + 4,
                found: 13
            })
        );
    }
    assert_eq!(MetaStride::for_aux(2), None);
    assert_eq!(
        PayloadError::BadSliceLength {
            expected: 8,
            found: 3
        }
        .to_string(),
        "row meta of 3 bytes; the stride is 8"
    );
    assert_eq!(
        PayloadError::ReservedBitSet { raw: 1 << 30 }.to_string(),
        "row payload 0x40000000 sets the reserved bit"
    );
}

#[test]
fn run_table_reads_and_writes() {
    let mut table = [0u8; 20];
    let entry = RunEntry::new(7, 3);
    assert_eq!(write_run(&mut table, 1, entry), Some(()));
    assert_eq!(write_run(&mut table, 2, entry), None);
    assert_eq!(read_run(&table, 1), Some(entry));
    assert_eq!(read_run(&table, 0), Some(RunEntry::new(0, 0)));
    assert_eq!(read_run(&table, 2), None);
    assert_eq!((entry.shard_id(), entry.run_len()), (7, 3));
    assert_eq!(RunEntry::from_bytes(entry.to_bytes()), entry);
}

#[test]
fn slab_appends_and_reopens() {
    let memory = VecMemory::new();
    let mut slab = Slab::open_or_init(&memory).unwrap();
    assert_eq!(slab.occupied_tail(), 32);
    let page = vec![9u8; 100_000];
    assert_eq!(slab.append_page(&page).unwrap(), 32);
    assert_eq!(slab.append_page(&[1, 2]).unwrap(), 100_032);
    assert_eq!(slab.header().occupied_tail(), 100_034);
    assert_eq!(memory.size(), 2);
    let reopened = Slab::open_or_init(&memory).unwrap();
    assert_eq!(reopened.occupied_tail(), 100_034);
    let mut bytes = [0; 2];
    memory.read(100_032, &mut bytes);
    assert_eq!(bytes, [1, 2]);
    // A grown but never written region is formatted.
    let fresh = VecMemory::new();
    assert_eq!(fresh.grow(1), Ok(0));
    assert_eq!(Slab::open_or_init(&fresh).unwrap().occupied_tail(), 32);
}

/// A memory that never grows.
struct Fixed(VecMemory);

impl Memory for Fixed {
    fn size(&self) -> u64 {
        self.0.size()
    }

    fn grow(&self, pages: u64) -> Result<u64, GrowFailed> {
        Err(GrowFailed::new(self.0.size(), pages))
    }

    fn read(&self, offset: u64, buf: &mut [u8]) {
        self.0.read(offset, buf);
    }

    fn write(&self, offset: u64, buf: &[u8]) {
        self.0.write(offset, buf);
    }
}

#[test]
fn slab_grows_only_past_the_end() {
    let inner = VecMemory::new();
    let _ = inner.grow(1).unwrap();
    let memory = Fixed(inner);
    let mut slab = Slab::open_or_init(&memory).unwrap();
    let fits = usize::try_from(PAGE_SIZE - 32).unwrap();
    assert_eq!(slab.append_page(&vec![1; fits]).unwrap(), 32);
    assert_eq!(
        slab.append_page(&[1]),
        Err(SlabError::Grow(GrowFailed::new(1, 1)))
    );
}

#[test]
fn slab_headers_fail_closed() {
    let memory = VecMemory::new();
    let _ = memory.grow(1).unwrap();
    memory.write(0, &SlabHeader::new(PAGE_SIZE + 1, 0).to_bytes());
    assert_eq!(
        Slab::open_or_init(&memory).unwrap_err(),
        SlabError::TailOutOfBounds {
            tail: PAGE_SIZE + 1,
            memory_len: PAGE_SIZE
        }
    );
    memory.write(0, &SlabHeader::new(31, 0).to_bytes());
    assert!(matches!(
        Slab::open_or_init(&memory),
        Err(SlabError::TailOutOfBounds { tail: 31, .. })
    ));
    memory.write(0, b"VSL1");
    assert!(matches!(
        Slab::open_or_init(&memory),
        Err(SlabError::Header(HeaderError::UnsupportedVersion {
            found: b'1'
        }))
    ));
    let error = SlabError::from(GrowFailed::new(1, 2));
    assert_eq!(
        error.to_string(),
        "memory could not grow by 2 pages from 1 pages"
    );
    assert_eq!(GrowFailed::new(3, 2).current_pages(), 3);
    assert_eq!(GrowFailed::new(3, 2).requested_pages(), 2);
}

proptest! {
    #[test]
    fn headers_decode_totally(bytes in proptest::array::uniform32(any::<u8>())) {
        if let Ok(header) = PageHeader::from_bytes(&bytes) {
            if let Ok(layout) = PageLayout::new(&header) {
                prop_assert_eq!(layout.vector_bytes_offset() % 16, 0);
            }
            prop_assert_eq!(&header.to_bytes()[..28], &bytes[..28]);
        }
        let _ = SlabHeader::from_bytes(&bytes);
    }

    #[test]
    fn max_capacity_is_the_largest_fit(
        budget in 0usize..200_000,
        row in 1u32..64,
        aux in prop_oneof![Just(0usize), Just(4), Just(8)],
        runs in 1u32..=MAX_RUNS,
        code in 0u32..16,
    ) {
        let stride = MetaStride::for_aux(aux).unwrap();
        let (row, code) = (row * 16, code * 4);
        let fits = |capacity: u32| {
            PageHeader::with_code_stride(capacity, row, stride, runs, code)
                .ok()
                .and_then(|header| PageLayout::new(&header).ok())
                .is_some_and(|layout| layout.page_len() <= budget)
        };
        match PageLayout::max_capacity_for(budget, row, stride, runs, code) {
            Some(capacity) => {
                prop_assert!(fits(capacity));
                prop_assert!(!fits(capacity + 1));
            }
            None => prop_assert!(!fits(1)),
        }
    }
}
