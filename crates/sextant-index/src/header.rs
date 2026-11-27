//! Slab and page headers: fixed-width, validated on every decode.
//!
//! ```text
//! slab header (32 B): magic "VSL" | version u8 | occupied_tail u64 | flags u32 | reserved [u8; 16]
//! page header (32 B): magic "VPG" | version u8 | capacity u32 | row_stride u32 | meta_stride u32
//!                     | run_capacity u32 | run_count u32 | code_stride u32 | zero [u8; 4]
//! ```
//!
//! All integers are little-endian. The version byte is binary (`1`); the retired ASCII layout
//! (`VSL1`, `VPG1`) carries `0x31` there and is rejected.

use crate::error::HeaderError;

/// The slab header's magic.
pub const MAGIC_SLAB: [u8; 3] = *b"VSL";
/// The page header's magic.
pub const MAGIC_PAGE: [u8; 3] = *b"VPG";
/// The format version both headers carry.
pub const FORMAT_VERSION: u8 = 1;
/// Runs a page can record at most.
pub const MAX_RUNS: u32 = 64;
/// Bytes of a slab header.
pub const SLAB_HEADER_SIZE: usize = 32;
/// Bytes of a page header.
pub const PAGE_HEADER_SIZE: usize = 32;

/// Code strides are multiples of this (the code tier's `f16` aux lanes stay aligned).
const CODE_STRIDE_ALIGN: u32 = 4;

/// The bytes of a row's meta entry: the 4-byte payload and 0, 4 or 8 opaque aux bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MetaStride {
    /// The payload alone.
    Bare,
    /// The payload and 4 aux bytes.
    Aux4,
    /// The payload and 8 aux bytes.
    Aux8,
}

impl MetaStride {
    /// The stride whose entry holds `aux_bytes` aux bytes (0, 4 or 8).
    #[must_use]
    pub const fn for_aux(aux_bytes: usize) -> Option<Self> {
        match aux_bytes {
            0 => Some(Self::Bare),
            4 => Some(Self::Aux4),
            8 => Some(Self::Aux8),
            _ => None,
        }
    }

    /// The stride stored as `bytes`.
    ///
    /// # Errors
    ///
    /// [`HeaderError::InvalidMetaStride`] for anything but 4, 8 or 12.
    pub const fn from_bytes(bytes: u32) -> Result<Self, HeaderError> {
        match bytes {
            4 => Ok(Self::Bare),
            8 => Ok(Self::Aux4),
            12 => Ok(Self::Aux8),
            found => Err(HeaderError::InvalidMetaStride { found }),
        }
    }

    /// Bytes of an entry.
    #[must_use]
    pub const fn bytes(self) -> u32 {
        match self {
            Self::Bare => 4,
            Self::Aux4 => 8,
            Self::Aux8 => 12,
        }
    }

    /// Aux bytes of an entry.
    #[must_use]
    pub const fn aux_bytes(self) -> usize {
        match self {
            Self::Bare => 0,
            Self::Aux4 => 4,
            Self::Aux8 => 8,
        }
    }
}

/// The header at offset 0 of a slab: where its first unused byte is.
///
/// Pages are appended at `occupied_tail` and the header is rewritten last, so a crash between
/// the two leaves a valid, smaller slab.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlabHeader {
    occupied_tail: u64,
    flags: u32,
    reserved: [u8; 16],
}

impl SlabHeader {
    /// A header whose first unused byte is `occupied_tail`.
    #[must_use]
    pub const fn new(occupied_tail: u64, flags: u32) -> Self {
        Self {
            occupied_tail,
            flags,
            reserved: [0; 16],
        }
    }

    /// The slab's first unused byte.
    #[must_use]
    pub const fn occupied_tail(&self) -> u64 {
        self.occupied_tail
    }

    /// Format flags (reserved for layout switches).
    #[must_use]
    pub const fn flags(&self) -> u32 {
        self.flags
    }

    /// The header's bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; SLAB_HEADER_SIZE] {
        let mut out = [0; SLAB_HEADER_SIZE];
        let mut fields = Writer(&mut out);
        fields.put(&MAGIC_SLAB);
        fields.put(&[FORMAT_VERSION]);
        fields.put(&self.occupied_tail.to_le_bytes());
        fields.put(&self.flags.to_le_bytes());
        fields.put(&self.reserved);
        out
    }

    /// Decodes and validates a header.
    ///
    /// # Errors
    ///
    /// [`HeaderError::BadMagic`] or [`HeaderError::UnsupportedVersion`].
    pub fn from_bytes(bytes: &[u8; SLAB_HEADER_SIZE]) -> Result<Self, HeaderError> {
        let mut fields = Reader::open(bytes, MAGIC_SLAB)?;
        Ok(Self {
            occupied_tail: u64::from_le_bytes(fields.take()),
            flags: fields.u32(),
            reserved: fields.take(),
        })
    }
}

/// The header at offset 0 of every page: its geometry and how many runs it records.
///
/// A header checks its fields; whether the page they describe fits this platform's address
/// space is [`crate::PageLayout::new`]'s question, since the same header can fit a 64-bit
/// address space and overflow a 32-bit one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageHeader {
    capacity: u32,
    row_stride: u32,
    meta_stride: MetaStride,
    run_capacity: u32,
    run_count: u32,
    code_stride: u32,
}

impl PageHeader {
    /// A page of `capacity` rows of `row_stride` vector bytes and `meta_stride` meta bytes, with
    /// room for `run_capacity` runs and no code segment.
    ///
    /// # Errors
    ///
    /// [`HeaderError`] for invalid geometry.
    pub fn new(
        capacity: u32,
        row_stride: u32,
        meta_stride: MetaStride,
        run_capacity: u32,
    ) -> Result<Self, HeaderError> {
        Self::with_code_stride(capacity, row_stride, meta_stride, run_capacity, 0)
    }

    /// [`Self::new`] with `code_stride` bytes of code segment per row (0 for none).
    ///
    /// # Errors
    ///
    /// [`HeaderError`] for invalid geometry.
    pub fn with_code_stride(
        capacity: u32,
        row_stride: u32,
        meta_stride: MetaStride,
        run_capacity: u32,
        code_stride: u32,
    ) -> Result<Self, HeaderError> {
        let header = Self {
            capacity,
            row_stride,
            meta_stride,
            run_capacity,
            run_count: 0,
            code_stride,
        };
        header.validate()?;
        Ok(header)
    }

    /// Row slots in the page.
    #[must_use]
    pub const fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Vector bytes per row (a multiple of 16).
    #[must_use]
    pub const fn row_stride(&self) -> u32 {
        self.row_stride
    }

    /// Meta bytes per row.
    #[must_use]
    pub const fn meta_stride(&self) -> MetaStride {
        self.meta_stride
    }

    /// Runs the page can record.
    #[must_use]
    pub const fn run_capacity(&self) -> u32 {
        self.run_capacity
    }

    /// Runs the page records.
    #[must_use]
    pub const fn run_count(&self) -> u32 {
        self.run_count
    }

    /// Code-segment bytes per row (0 when the page has no code table). Pages describe
    /// themselves, so pages of one slab may differ here.
    #[must_use]
    pub const fn code_stride(&self) -> u32 {
        self.code_stride
    }

    /// Records `run_count` runs.
    ///
    /// # Errors
    ///
    /// [`HeaderError::RunCountExceedsCapacity`] past the run capacity.
    pub const fn set_run_count(&mut self, run_count: u32) -> Result<(), HeaderError> {
        if run_count > self.run_capacity {
            return Err(HeaderError::RunCountExceedsCapacity {
                run_count,
                run_capacity: self.run_capacity,
            });
        }
        self.run_count = run_count;
        Ok(())
    }

    const fn validate(&self) -> Result<(), HeaderError> {
        if self.capacity == 0 {
            return Err(HeaderError::ZeroCapacity);
        }
        if self.row_stride == 0 || !self.row_stride.is_multiple_of(16) {
            return Err(HeaderError::InvalidRowStride {
                found: self.row_stride,
            });
        }
        if self.run_capacity == 0 || self.run_capacity > MAX_RUNS {
            return Err(HeaderError::InvalidRunCapacity {
                found: self.run_capacity,
            });
        }
        if !self.code_stride.is_multiple_of(CODE_STRIDE_ALIGN) {
            return Err(HeaderError::InvalidCodeStride {
                found: self.code_stride,
            });
        }
        if self.run_count > self.run_capacity {
            return Err(HeaderError::RunCountExceedsCapacity {
                run_count: self.run_count,
                run_capacity: self.run_capacity,
            });
        }
        Ok(())
    }

    /// The header's bytes.
    #[must_use]
    pub fn to_bytes(&self) -> [u8; PAGE_HEADER_SIZE] {
        let mut out = [0; PAGE_HEADER_SIZE];
        let mut fields = Writer(&mut out);
        fields.put(&MAGIC_PAGE);
        fields.put(&[FORMAT_VERSION]);
        for word in [
            self.capacity,
            self.row_stride,
            self.meta_stride.bytes(),
            self.run_capacity,
            self.run_count,
            self.code_stride,
        ] {
            fields.put(&word.to_le_bytes());
        }
        out
    }

    /// Decodes and validates a header.
    ///
    /// # Errors
    ///
    /// [`HeaderError`] for a wrong magic or version, or invalid geometry.
    pub fn from_bytes(bytes: &[u8; PAGE_HEADER_SIZE]) -> Result<Self, HeaderError> {
        let mut fields = Reader::open(bytes, MAGIC_PAGE)?;
        let capacity = fields.u32();
        let row_stride = fields.u32();
        let meta_stride = MetaStride::from_bytes(fields.u32())?;
        let header = Self {
            capacity,
            row_stride,
            meta_stride,
            run_capacity: fields.u32(),
            run_count: fields.u32(),
            code_stride: fields.u32(),
        };
        header.validate()?;
        Ok(header)
    }
}

/// Reads fixed-width fields front to back; past the end it reads zeros (never reached over the
/// fixed-size headers).
struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    /// Checks the magic and the version and returns a reader past them.
    fn open(bytes: &'a [u8], magic: [u8; 3]) -> Result<Self, HeaderError> {
        let mut fields = Self(bytes);
        let found: [u8; 3] = fields.take();
        if found != magic {
            return Err(HeaderError::BadMagic {
                expected: magic,
                found,
            });
        }
        let [version] = fields.take();
        if version != FORMAT_VERSION {
            return Err(HeaderError::UnsupportedVersion { found: version });
        }
        Ok(fields)
    }

    const fn take<const N: usize>(&mut self) -> [u8; N] {
        match self.0.split_first_chunk::<N>() {
            Some((field, rest)) => {
                self.0 = rest;
                *field
            }
            None => [0; N],
        }
    }

    const fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take())
    }
}

/// Writes fields front to back; past the end it writes nothing (never reached over the
/// fixed-size headers).
struct Writer<'a>(&'a mut [u8]);

impl Writer<'_> {
    fn put(&mut self, field: &[u8]) {
        let rest = core::mem::take(&mut self.0);
        if let Some((head, tail)) = rest.split_at_mut_checked(field.len()) {
            head.copy_from_slice(field);
            self.0 = tail;
        }
    }
}

/// Proofs over every input (`cargo kani`).
#[cfg(kani)]
mod proofs {
    use super::*;

    /// Decoding any 32 bytes is total, and a decoded page header encodes back to them.
    #[kani::proof]
    fn page_header_decoding_is_total() {
        let bytes: [u8; PAGE_HEADER_SIZE] = kani::any();
        kani::assume(bytes[28..] == [0; 4]);
        if let Ok(header) = PageHeader::from_bytes(&bytes) {
            assert!(header.to_bytes() == bytes);
        }
    }

    /// Slab headers round-trip.
    #[kani::proof]
    fn slab_header_round_trips() {
        let header = SlabHeader::new(kani::any(), kani::any());
        assert!(SlabHeader::from_bytes(&header.to_bytes()) == Ok(header));
    }
}
