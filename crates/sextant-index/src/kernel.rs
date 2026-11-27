//! Distance kernels over stored rows.
//!
//! A row is a byte span in one of three encodings: [`F32Row`] (little-endian `f32`s), [`I8Row`]
//! (signed bytes against a per-row scale) and [`U8Row`] (unsigned bytes offset by 128). Each
//! offers squared L2, dot product, and early-exit forms of both that stop as soon as the row
//! cannot beat a threshold.
//!
//! A kernel reads the components the row and the query both have, in blocks (4 components of an
//! `f32` row, 16 of a quantized one) over two alternating four-lane accumulators, then the tail
//! one component at a time. The lane type is a wasm `v128` where `simd128` is enabled and an
//! array elsewhere; both run the same operations in the same order, so a distance is bitwise the
//! same on a canister and on the host that tests it.

#![expect(
    clippy::float_arithmetic,
    reason = "distance kernels are floating-point by definition"
)]

mod lanes;
mod row;

use core::cmp::Ordering;
use lanes::Lanes;
use row::{Encoding, F32, I8, Total, U8};

/// The Cauchy-Schwarz bound that lets a dot product stop early.
///
/// `suffix_norms[j]` is the norm of the query's components from `j` on (`query.len() + 1`
/// entries, the last 0), and `max_norm` bounds the norm of any suffix of the row. Then the
/// components still to come add at most `suffix_norms[j] × max_norm`, and a row whose partial
/// dot plus that stays below `threshold` cannot reach it. A missing entry disables the check
/// there; a non-finite `threshold` never triggers it.
#[derive(Clone, Copy, Debug)]
pub struct DotBound<'a> {
    suffix_norms: &'a [f32],
    max_norm: f32,
    threshold: f32,
}

impl<'a> DotBound<'a> {
    /// The bound for rows of norm at most `max_norm`.
    #[must_use]
    pub const fn new(suffix_norms: &'a [f32], max_norm: f32, threshold: f32) -> Self {
        Self {
            suffix_norms,
            max_norm,
            threshold,
        }
    }

    /// The bound for unit-normalized rows.
    #[must_use]
    pub const fn unit(suffix_norms: &'a [f32], threshold: f32) -> Self {
        Self::new(suffix_norms, 1.0, threshold)
    }

    /// Whether a partial dot over the first `consumed` components can still reach the threshold.
    fn reachable(&self, partial: f32, consumed: usize) -> bool {
        self.suffix_norms
            .get(consumed)
            // Not `>=`: a NaN bound must not stop the scan (the finiteness check catches it).
            .is_none_or(|&suffix| {
                (partial + suffix * self.max_norm).partial_cmp(&self.threshold)
                    != Some(Ordering::Less)
            })
    }
}

/// A row of little-endian `f32` components.
#[derive(Clone, Copy, Debug)]
pub struct F32Row<'a>(F32<'a>);

impl<'a> F32Row<'a> {
    /// The row stored in `bytes`; a trailing partial component is ignored.
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self(F32::new(bytes))
    }

    /// The dot product with `query` and the row's squared norm, in one pass (cosine scoring of
    /// unnormalized rows).
    #[must_use]
    pub fn dot_and_norm2(self, query: &[f32]) -> (f32, f32) {
        let dims = self.0.dims().min(query.len());
        let blocks = whole_blocks::<1>(dims);
        let (mut dot, mut norm) = (Lanes::zero(), Lanes::zero());
        for block in 0..blocks {
            if let (Some([v]), Some([q])) = (self.0.lanes(block), query_lanes::<1>(query, block)) {
                dot = dot.add(v.mul(q));
                norm = norm.add(v.mul(v));
            }
        }
        tail::<1, _>(self.0, query, blocks, dims)
            .fold((dot.sum(), norm.sum()), |(dot, norm), (v, q)| {
                (dot + v * q, norm + v * v)
            })
    }
}

/// A row of signed bytes against a per-row scale: component `b × scale / 127`.
#[derive(Clone, Copy, Debug)]
pub struct I8Row<'a>(I8<'a>);

impl<'a> I8Row<'a> {
    /// The row stored in `bytes`, quantized against `scale`. A zero scale is the zero vector.
    #[must_use]
    pub fn new(bytes: &'a [u8], scale: f32) -> Self {
        Self(I8::new(bytes, scale))
    }
}

/// A row of unsigned bytes offset by 128 against a per-row scale: component
/// `(b − 128) × scale / 127`.
#[derive(Clone, Copy, Debug)]
pub struct U8Row<'a>(U8<'a>);

impl<'a> U8Row<'a> {
    /// The row stored in `bytes`, quantized against `scale`. A zero scale is the zero vector.
    #[must_use]
    pub fn new(bytes: &'a [u8], scale: f32) -> Self {
        Self(U8::new(bytes, scale))
    }
}

/// The four kernels of a row type, over its encoding with `N`-lane blocks.
macro_rules! kernels {
    ($row:ident, $lanes:literal) => {
        impl $row<'_> {
            /// Squared L2 distance `Σ (v − q)²` over the components the row and `query` share.
            #[must_use]
            pub fn l2_squared(self, query: &[f32]) -> f32 {
                full::<$lanes, _>(self.0, query, Metric::L2Squared)
            }

            /// [`Self::l2_squared`], or `None` as soon as the partial distance exceeds
            /// `threshold` (partial sums only grow, so the exit is exact) or when the distance
            /// is not finite. A non-finite `threshold` never triggers the exit.
            #[must_use]
            pub fn l2_squared_within(self, query: &[f32], threshold: f32) -> Option<f32> {
                l2_within::<$lanes, _>(self.0, query, threshold)
            }

            /// Dot product `Σ v × q` over the components the row and `query` share.
            #[must_use]
            pub fn dot(self, query: &[f32]) -> f32 {
                full::<$lanes, _>(self.0, query, Metric::Dot)
            }

            /// [`Self::dot`], or `None` as soon as `bound` shows it cannot reach its threshold
            /// (checked about every eighth of the row) or when the dot is not finite.
            #[must_use]
            pub fn dot_reaching(self, query: &[f32], bound: DotBound<'_>) -> Option<f32> {
                dot_reaching::<$lanes, _>(self.0, query, bound)
            }
        }
    };
}

kernels!(F32Row, 1);
kernels!(I8Row, 4);
kernels!(U8Row, 4);

/// Matching bits of two code spans: `XNOR` and popcount over whole little-endian `u64` words.
/// Spans of unequal length are compared over the shorter; a partial trailing word is ignored.
#[must_use]
pub fn popcount_xnor_words(a: &[u8], b: &[u8]) -> u32 {
    a.as_chunks::<8>()
        .0
        .iter()
        .zip(b.as_chunks::<8>().0)
        .fold(0u32, |matched, (&a, &b)| {
            let same = !(u64::from_le_bytes(a) ^ u64::from_le_bytes(b));
            matched.saturating_add(same.count_ones())
        })
}

/// What a full kernel sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Metric {
    L2Squared,
    Dot,
}

impl Metric {
    fn term(self, v: Lanes, q: Lanes) -> Lanes {
        match self {
            Self::L2Squared => {
                let d = v.sub(q);
                d.mul(d)
            }
            Self::Dot => v.mul(q),
        }
    }

    fn scalar(self, v: f32, q: f32) -> f32 {
        match self {
            Self::L2Squared => {
                let d = v - q;
                d * d
            }
            Self::Dot => v * q,
        }
    }

    /// The contribution of one block: its lanes' terms added right to left.
    fn block<const N: usize>(self, v: [Lanes; N], q: [Lanes; N]) -> Option<Lanes> {
        v.into_iter()
            .zip(q)
            .map(|(v, q)| self.term(v, q))
            .rev()
            .reduce(|right, term| term.add(right))
    }
}

/// Components per block of `N` lanes.
const fn block_dims<const N: usize>() -> usize {
    N.saturating_mul(4)
}

/// Whole blocks of `N` lanes in `dims` components.
fn whole_blocks<const N: usize>(dims: usize) -> usize {
    dims.checked_div(block_dims::<N>()).unwrap_or(0)
}

/// Block `block` of `query` as `N` lanes.
fn query_lanes<const N: usize>(query: &[f32], block: usize) -> Option<[Lanes; N]> {
    let start = block.checked_mul(block_dims::<N>())?;
    let span = query.get(start..start.checked_add(block_dims::<N>())?)?;
    let lanes: &[[f32; 4]; N] = span.as_chunks::<4>().0.try_into().ok()?;
    Some(lanes.map(Lanes::from_array))
}

/// The `(row, query)` component pairs past the first `blocks` whole blocks, up to `dims`.
fn tail<'q, const N: usize, E: Encoding<N> + 'q>(
    row: E,
    query: &'q [f32],
    blocks: usize,
    dims: usize,
) -> impl Iterator<Item = (f32, f32)> + 'q {
    let start = blocks.saturating_mul(block_dims::<N>());
    (start..dims).filter_map(move |index| Some((row.component(index)?, *query.get(index)?)))
}

/// The contribution of block `block`.
fn block<const N: usize, E: Encoding<N>>(
    row: E,
    query: &[f32],
    block: usize,
    metric: Metric,
) -> Option<Lanes> {
    metric.block(row.lanes(block)?, query_lanes::<N>(query, block)?)
}

/// Sums blocks `0..blocks` into two alternating accumulators, calling `check` with them after
/// each one; stops with `None` when `check` does.
fn accumulate<const N: usize, E: Encoding<N>>(
    row: E,
    query: &[f32],
    blocks: usize,
    metric: Metric,
    mut check: impl FnMut(usize, Lanes, Lanes) -> bool,
) -> Option<(Lanes, Lanes)> {
    let (mut even, mut odd) = (Lanes::zero(), Lanes::zero());
    for index in 0..blocks {
        if let Some(contribution) = block::<N, E>(row, query, index, metric) {
            if index.is_multiple_of(2) {
                even = even.add(contribution);
            } else {
                odd = odd.add(contribution);
            }
        }
        if !check(index.saturating_add(1), even, odd) {
            return None;
        }
    }
    Some((even, odd))
}

fn full<const N: usize, E: Encoding<N>>(row: E, query: &[f32], metric: Metric) -> f32 {
    let dims = row.dims().min(query.len());
    let blocks = whole_blocks::<N>(dims);
    let total = accumulate::<N, E>(row, query, blocks, metric, |_, _, _| true)
        .map_or(0.0, |(even, odd)| E::TOTAL.combine(even, odd));
    tail::<N, E>(row, query, blocks, dims).fold(total, |sum, (v, q)| sum + metric.scalar(v, q))
}

fn l2_within<const N: usize, E: Encoding<N>>(row: E, query: &[f32], threshold: f32) -> Option<f32> {
    let metric = Metric::L2Squared;
    let dims = row.dims().min(query.len());
    let blocks = whole_blocks::<N>(dims);
    let mut sum = 0.0;
    for index in 0..blocks {
        sum += block::<N, E>(row, query, index, metric).map_or(0.0, Lanes::sum);
        if sum > threshold {
            return None;
        }
    }
    for (v, q) in tail::<N, E>(row, query, blocks, dims) {
        sum += metric.scalar(v, q);
        if sum > threshold {
            return None;
        }
    }
    // A non-finite component makes the sum NaN or infinite; NaN never exceeds the threshold, so
    // it is caught here.
    sum.is_finite().then_some(sum)
}

fn dot_reaching<const N: usize, E: Encoding<N>>(
    row: E,
    query: &[f32],
    bound: DotBound<'_>,
) -> Option<f32> {
    let metric = Metric::Dot;
    let dims = row.dims().min(query.len());
    let blocks = whole_blocks::<N>(dims);
    // Checked after every `period` blocks (an even count, about an eighth of the row), so the
    // full path stays close to `dot` when the exit never triggers.
    let period = blocks.checked_div(8).unwrap_or(0).max(1).saturating_mul(2);
    let (even, odd) = accumulate::<N, E>(row, query, blocks, metric, |done, even, odd| {
        !done.is_multiple_of(period)
            || bound.reachable(
                Total::Grouped.combine(even, odd),
                done.saturating_mul(block_dims::<N>()),
            )
    })?;
    let sum = tail::<N, E>(row, query, blocks, dims)
        .fold(Total::Grouped.combine(even, odd), |sum, (v, q)| sum + v * q);
    sum.is_finite().then_some(sum)
}
