//! A row's identity and meta entry.
//!
//! A row stores a 30-bit id local to its run's shard (the shard is recorded once per run). Bit 30
//! is reserved and bit 31 marks a deleted row, so a scan skips dead rows from the payload alone.

use crate::error::PayloadError;
use crate::header::MetaStride;

/// The largest row id: ids have 30 bits.
pub const MAX_ROW_ID: u32 = ID_MASK;

const ID_MASK: u32 = (1 << 30) - 1;
const RESERVED_BIT: u32 = 1 << 30;
const TOMBSTONE_BIT: u32 = 1 << 31;

/// A row's 30-bit id and its tombstone bit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct RowPayload(u32);

impl RowPayload {
    /// A live row with `id`, or `None` past [`MAX_ROW_ID`].
    #[must_use]
    pub const fn new(id: u32) -> Option<Self> {
        if id > ID_MASK { None } else { Some(Self(id)) }
    }

    /// The payload stored as `raw`.
    ///
    /// # Errors
    ///
    /// [`PayloadError::ReservedBitSet`] when bit 30 is set.
    pub const fn from_raw(raw: u32) -> Result<Self, PayloadError> {
        if raw & RESERVED_BIT != 0 {
            return Err(PayloadError::ReservedBitSet { raw });
        }
        Ok(Self(raw))
    }

    /// The row's id.
    #[must_use]
    pub const fn id(self) -> u32 {
        self.0 & ID_MASK
    }

    /// Whether the row is deleted.
    #[must_use]
    pub const fn is_tombstone(self) -> bool {
        self.0 & TOMBSTONE_BIT != 0
    }

    /// The same row, deleted.
    #[must_use]
    pub const fn tombstoned(self) -> Self {
        Self(self.0 | TOMBSTONE_BIT)
    }

    /// The stored word.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// A row's meta entry: its payload and up to 8 aux bytes, opaque to this crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RowMeta {
    payload: RowPayload,
    aux: [u8; 8],
}

impl RowMeta {
    /// An entry of `payload` and `aux` (only the stride's aux bytes are stored).
    #[must_use]
    pub const fn new(payload: RowPayload, aux: [u8; 8]) -> Self {
        Self { payload, aux }
    }

    /// The row's payload.
    #[must_use]
    pub const fn payload(&self) -> RowPayload {
        self.payload
    }

    /// The aux bytes a `stride` entry stores.
    #[must_use]
    pub fn aux(&self, stride: MetaStride) -> &[u8] {
        self.aux.get(..stride.aux_bytes()).unwrap_or_default()
    }

    /// Writes the entry into `out`: the payload, then the stride's aux bytes.
    ///
    /// # Errors
    ///
    /// [`PayloadError::BadSliceLength`] unless `out` is the stride's length.
    pub fn write_into(&self, out: &mut [u8], stride: MetaStride) -> Result<(), PayloadError> {
        let (payload, aux) = split(out, stride)?;
        payload.copy_from_slice(&self.payload.raw().to_le_bytes());
        aux.copy_from_slice(self.aux(stride));
        Ok(())
    }

    /// Reads an entry of `stride` from `bytes`.
    ///
    /// # Errors
    ///
    /// [`PayloadError`] unless `bytes` is the stride's length and the payload's reserved bit is
    /// clear.
    pub fn from_bytes(bytes: &[u8], stride: MetaStride) -> Result<Self, PayloadError> {
        let expected = meta_len(stride);
        let (&raw, rest) = bytes
            .split_first_chunk::<4>()
            .filter(|_| bytes.len() == expected)
            .ok_or(PayloadError::BadSliceLength {
                expected,
                found: bytes.len(),
            })?;
        let payload = RowPayload::from_raw(u32::from_le_bytes(raw))?;
        let mut aux = [0; 8];
        if let Some(head) = aux.get_mut(..rest.len()) {
            head.copy_from_slice(rest);
        }
        Ok(Self { payload, aux })
    }
}

const fn meta_len(stride: MetaStride) -> usize {
    stride.aux_bytes().saturating_add(4)
}

/// `out` as the payload word and the aux bytes of a `stride` entry.
fn split(out: &mut [u8], stride: MetaStride) -> Result<(&mut [u8; 4], &mut [u8]), PayloadError> {
    let expected = meta_len(stride);
    let found = out.len();
    if found != expected {
        return Err(PayloadError::BadSliceLength { expected, found });
    }
    out.split_first_chunk_mut::<4>()
        .ok_or(PayloadError::BadSliceLength { expected, found })
}
