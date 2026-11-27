//! Errors: what can be wrong with a header, a row's meta, or the memory a slab lives in.

use core::fmt;

/// Why a slab or page header is invalid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum HeaderError {
    /// The magic is not the header's.
    BadMagic {
        /// The magic the header kind carries.
        expected: [u8; 3],
        /// The magic found.
        found: [u8; 3],
    },
    /// A format version this reader does not speak.
    UnsupportedVersion {
        /// The version byte found.
        found: u8,
    },
    /// A meta stride other than 4, 8 or 12.
    InvalidMetaStride {
        /// The stride found.
        found: u32,
    },
    /// A row stride that is zero or not a multiple of 16.
    InvalidRowStride {
        /// The stride found.
        found: u32,
    },
    /// A page of no rows.
    ZeroCapacity,
    /// A run capacity that is zero or above [`crate::MAX_RUNS`].
    InvalidRunCapacity {
        /// The capacity found.
        found: u32,
    },
    /// A code stride that is neither 0 nor a multiple of 4.
    InvalidCodeStride {
        /// The stride found.
        found: u32,
    },
    /// More runs than the run capacity.
    RunCountExceedsCapacity {
        /// The run count.
        run_count: u32,
        /// The run capacity.
        run_capacity: u32,
    },
    /// The page's byte span overflows the address space.
    SpanOverflow,
}

impl fmt::Display for HeaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic { expected, found } => {
                write!(f, "header magic {found:?} is not {expected:?}")
            }
            Self::UnsupportedVersion { found } => {
                write!(f, "header format version {found} unsupported")
            }
            Self::InvalidMetaStride { found } => {
                write!(f, "meta stride {found} is not 4, 8 or 12")
            }
            Self::InvalidRowStride { found } => {
                write!(f, "row stride {found} is not a positive multiple of 16")
            }
            Self::ZeroCapacity => f.write_str("page capacity is zero"),
            Self::InvalidRunCapacity { found } => {
                write!(f, "run capacity {found} is outside 1..=64")
            }
            Self::InvalidCodeStride { found } => {
                write!(f, "code stride {found} is neither 0 nor a multiple of 4")
            }
            Self::RunCountExceedsCapacity {
                run_count,
                run_capacity,
            } => write!(f, "{run_count} runs exceed the capacity of {run_capacity}"),
            Self::SpanOverflow => f.write_str("page span overflows the address space"),
        }
    }
}

impl core::error::Error for HeaderError {}

/// Why a row's meta cannot be read or written.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PayloadError {
    /// The byte span is not the meta stride long.
    BadSliceLength {
        /// The stride's length.
        expected: usize,
        /// The span's length.
        found: usize,
    },
    /// The stored payload sets the reserved bit.
    ReservedBitSet {
        /// The raw payload word.
        raw: u32,
    },
}

impl fmt::Display for PayloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadSliceLength { expected, found } => {
                write!(f, "row meta of {found} bytes; the stride is {expected}")
            }
            Self::ReservedBitSet { raw } => {
                write!(f, "row payload {raw:#010x} sets the reserved bit")
            }
        }
    }
}

impl core::error::Error for PayloadError {}

/// A memory could not grow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrowFailed {
    current_pages: u64,
    requested_pages: u64,
}

impl GrowFailed {
    /// A failed request for `requested_pages` more pages on a memory of `current_pages`.
    #[must_use]
    pub const fn new(current_pages: u64, requested_pages: u64) -> Self {
        Self {
            current_pages,
            requested_pages,
        }
    }

    /// The memory's size, in pages, when growth failed.
    #[must_use]
    pub const fn current_pages(&self) -> u64 {
        self.current_pages
    }

    /// The pages the request asked for.
    #[must_use]
    pub const fn requested_pages(&self) -> u64 {
        self.requested_pages
    }
}

impl fmt::Display for GrowFailed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "memory could not grow by {} pages from {} pages",
            self.requested_pages, self.current_pages
        )
    }
}

impl core::error::Error for GrowFailed {}

/// Why a slab cannot be opened or appended to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SlabError {
    /// The slab header is invalid.
    Header(HeaderError),
    /// The header's tail lies before the header's end or past the memory.
    TailOutOfBounds {
        /// The tail the header records.
        tail: u64,
        /// The memory's size in bytes.
        memory_len: u64,
    },
    /// The memory could not grow to hold the write.
    Grow(GrowFailed),
    /// The write would end past `u64::MAX`.
    TailOverflow,
}

impl fmt::Display for SlabError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Header(error) => write!(f, "slab header: {error}"),
            Self::TailOutOfBounds { tail, memory_len } => write!(
                f,
                "slab tail {tail} out of bounds for a memory of {memory_len} bytes"
            ),
            Self::Grow(error) => error.fmt(f),
            Self::TailOverflow => f.write_str("slab append overflows the address space"),
        }
    }
}

impl core::error::Error for SlabError {}

impl From<HeaderError> for SlabError {
    fn from(error: HeaderError) -> Self {
        Self::Header(error)
    }
}

impl From<GrowFailed> for SlabError {
    fn from(error: GrowFailed) -> Self {
        Self::Grow(error)
    }
}
