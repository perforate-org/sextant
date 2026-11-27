//! Stored row encodings, as blocks of lanes for the kernels and as single components for the
//! tail past the last whole block.

#![expect(
    clippy::float_arithmetic,
    reason = "dequantization is floating-point by definition"
)]

use super::lanes::Lanes;

/// How the two accumulators of a full kernel are added up.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Total {
    /// Every lane of the first, then every lane of the second, left to right.
    Sequential,
    /// The first's lane sum plus the second's.
    Grouped,
}

impl Total {
    pub(super) fn combine(self, first: Lanes, second: Lanes) -> f32 {
        match self {
            Self::Sequential => second
                .to_array()
                .into_iter()
                .fold(first.sum(), |sum, lane| sum + lane),
            Self::Grouped => first.sum() + second.sum(),
        }
    }
}

/// A row encoding whose blocks are `N` lanes (`4 × N` components).
pub(super) trait Encoding<const N: usize>: Copy {
    /// How a full kernel adds up its accumulators.
    const TOTAL: Total;

    /// Components the row stores.
    fn dims(&self) -> usize;

    /// Block `block`, dequantized; `None` past the row.
    fn lanes(&self, block: usize) -> Option<[Lanes; N]>;

    /// Component `index`, dequantized; `None` past the row.
    fn component(&self, index: usize) -> Option<f32>;
}

/// Little-endian `f32` components, four bytes each.
#[derive(Clone, Copy, Debug)]
pub(super) struct F32<'a>(&'a [u8]);

impl<'a> F32<'a> {
    pub(super) const fn new(bytes: &'a [u8]) -> Self {
        Self(bytes)
    }
}

impl Encoding<1> for F32<'_> {
    const TOTAL: Total = Total::Sequential;

    fn dims(&self) -> usize {
        self.0.as_chunks::<4>().0.len()
    }

    fn lanes(&self, block: usize) -> Option<[Lanes; 1]> {
        let chunk = self.0.as_chunks::<16>().0.get(block)?;
        Some([Lanes::from_le_bytes(*chunk)])
    }

    fn component(&self, index: usize) -> Option<f32> {
        self.0
            .as_chunks::<4>()
            .0
            .get(index)
            .map(|&bytes| f32::from_le_bytes(bytes))
    }
}

/// Signed bytes scaled by `k`: component `b as i8 × k`.
#[derive(Clone, Copy, Debug)]
pub(super) struct I8<'a> {
    bytes: &'a [u8],
    k: f32,
}

impl<'a> I8<'a> {
    /// A row quantized against `scale`: a byte of magnitude 127 stands for `scale`.
    pub(super) fn new(bytes: &'a [u8], scale: f32) -> Self {
        Self {
            bytes,
            k: scale / 127.0,
        }
    }
}

impl Encoding<4> for I8<'_> {
    const TOTAL: Total = Total::Grouped;

    fn dims(&self) -> usize {
        self.bytes.len()
    }

    fn lanes(&self, block: usize) -> Option<[Lanes; 4]> {
        let chunk = self.bytes.as_chunks::<16>().0.get(block)?;
        let k = Lanes::splat(self.k);
        Some(Lanes::widen_i8(*chunk).map(|lanes| lanes.mul(k)))
    }

    fn component(&self, index: usize) -> Option<f32> {
        let byte = *self.bytes.get(index)?;
        Some(f32::from(i8::from_le_bytes([byte])) * self.k)
    }
}

/// Unsigned bytes offset by 128 and scaled by `k`: component `(b − 128) × k`. The lanes compute
/// `b × k − 128 × k`, which rounds differently from the single components; both are kept as the
/// format was measured.
#[derive(Clone, Copy, Debug)]
pub(super) struct U8<'a> {
    bytes: &'a [u8],
    k: f32,
}

impl<'a> U8<'a> {
    /// A row quantized against `scale`: a byte of magnitude 127 stands for `scale`.
    pub(super) fn new(bytes: &'a [u8], scale: f32) -> Self {
        Self {
            bytes,
            k: scale / 127.0,
        }
    }
}

impl Encoding<4> for U8<'_> {
    const TOTAL: Total = Total::Grouped;

    fn dims(&self) -> usize {
        self.bytes.len()
    }

    fn lanes(&self, block: usize) -> Option<[Lanes; 4]> {
        let chunk = self.bytes.as_chunks::<16>().0.get(block)?;
        let k = Lanes::splat(self.k);
        let offset = Lanes::splat(128.0 * self.k);
        Some(Lanes::widen_u8(*chunk).map(|lanes| lanes.mul(k).sub(offset)))
    }

    fn component(&self, index: usize) -> Option<f32> {
        let byte = *self.bytes.get(index)?;
        Some((f32::from(byte) - 128.0) * self.k)
    }
}
