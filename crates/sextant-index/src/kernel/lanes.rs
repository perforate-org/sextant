//! Four `f32` lanes: a wasm `v128` where `simd128` is enabled, an array elsewhere.
//!
//! Both backends perform the same IEEE-754 operations lane by lane, and every horizontal sum adds
//! lanes in the same fixed order, so a kernel returns bitwise the same result on either. Loads are
//! built from arrays rather than through `v128_load`, which keeps the backend free of `unsafe`.

#![expect(
    clippy::float_arithmetic,
    reason = "lane arithmetic is floating-point by definition"
)]

#[cfg(all(
    target_family = "wasm",
    target_feature = "simd128",
    target_arch = "wasm32"
))]
use core::arch::wasm32 as simd;
#[cfg(all(
    target_family = "wasm",
    target_feature = "simd128",
    target_arch = "wasm64"
))]
use core::arch::wasm64 as simd;

/// Whether the `v128` backend is compiled in.
macro_rules! simd_backend {
    ($($item:item)*) => {
        $(#[cfg(all(target_family = "wasm", target_feature = "simd128"))] $item)*
    };
}

/// Whether the array backend is compiled in.
macro_rules! array_backend {
    ($($item:item)*) => {
        $(#[cfg(not(all(target_family = "wasm", target_feature = "simd128")))] $item)*
    };
}

simd_backend! {
    type Repr = simd::v128;
}

array_backend! {
    type Repr = [f32; 4];
}

/// Four `f32` lanes.
#[derive(Clone, Copy, Debug)]
pub(super) struct Lanes(Repr);

simd_backend! {
    impl Lanes {
        /// Every lane zero.
        pub(super) fn zero() -> Self {
            Self::splat(0.0)
        }

        /// Lanes `l0 + l1 + l2 + l3`, added left to right.
        pub(super) fn sum(self) -> f32 {
            let [l0, l1, l2, l3] = self.to_array();
            l0 + l1 + l2 + l3
        }

        /// Lanes from four little-endian `f32`s.
        pub(super) const fn from_le_bytes(bytes: [u8; 16]) -> Self {
            let [
                b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
            ] = bytes;
            Self::from_array([
                f32::from_le_bytes([b0, b1, b2, b3]),
                f32::from_le_bytes([b4, b5, b6, b7]),
                f32::from_le_bytes([b8, b9, b10, b11]),
                f32::from_le_bytes([b12, b13, b14, b15]),
            ])
        }

        pub(super) fn splat(value: f32) -> Self {
            Self(simd::f32x4_splat(value))
        }

        pub(super) const fn from_array([l0, l1, l2, l3]: [f32; 4]) -> Self {
            Self(simd::f32x4(l0, l1, l2, l3))
        }

        pub(super) fn to_array(self) -> [f32; 4] {
            [
                simd::f32x4_extract_lane::<0>(self.0),
                simd::f32x4_extract_lane::<1>(self.0),
                simd::f32x4_extract_lane::<2>(self.0),
                simd::f32x4_extract_lane::<3>(self.0),
            ]
        }

        pub(super) fn add(self, other: Self) -> Self {
            Self(simd::f32x4_add(self.0, other.0))
        }

        pub(super) fn sub(self, other: Self) -> Self {
            Self(simd::f32x4_sub(self.0, other.0))
        }

        pub(super) fn mul(self, other: Self) -> Self {
            Self(simd::f32x4_mul(self.0, other.0))
        }

        /// Sixteen signed bytes as four lanes of exact `f32`s, in order.
        pub(super) fn widen_i8(bytes: [u8; 16]) -> [Self; 4] {
            let v = bytes_v128(bytes);
            let low = simd::i16x8_extend_low_i8x16(v);
            let high = simd::i16x8_extend_high_i8x16(v);
            widen_i16(low, high)
        }

        /// Sixteen unsigned bytes as four lanes of exact `f32`s, in order.
        pub(super) fn widen_u8(bytes: [u8; 16]) -> [Self; 4] {
            let v = bytes_v128(bytes);
            let low = simd::u16x8_extend_low_u8x16(v);
            let high = simd::u16x8_extend_high_u8x16(v);
            widen_i16(low, high)
        }
    }

    const fn bytes_v128(bytes: [u8; 16]) -> simd::v128 {
        let [
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
        ] = bytes;
        simd::u8x16(
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
        )
    }

    /// Eight and eight 16-bit lanes (all within `i16`) as four `f32` lanes.
    fn widen_i16(low: simd::v128, high: simd::v128) -> [Lanes; 4] {
        [
            Lanes(simd::f32x4_convert_i32x4(simd::i32x4_extend_low_i16x8(low))),
            Lanes(simd::f32x4_convert_i32x4(simd::i32x4_extend_high_i16x8(low))),
            Lanes(simd::f32x4_convert_i32x4(simd::i32x4_extend_low_i16x8(high))),
            Lanes(simd::f32x4_convert_i32x4(simd::i32x4_extend_high_i16x8(high))),
        ]
    }
}

array_backend! {
    impl Lanes {
        /// Every lane zero.
        pub(super) const fn zero() -> Self {
            Self::splat(0.0)
        }

        /// Lanes `l0 + l1 + l2 + l3`, added left to right.
        pub(super) fn sum(self) -> f32 {
            let [l0, l1, l2, l3] = self.to_array();
            l0 + l1 + l2 + l3
        }

        /// Lanes from four little-endian `f32`s.
        pub(super) const fn from_le_bytes(bytes: [u8; 16]) -> Self {
            let [
                b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
            ] = bytes;
            Self::from_array([
                f32::from_le_bytes([b0, b1, b2, b3]),
                f32::from_le_bytes([b4, b5, b6, b7]),
                f32::from_le_bytes([b8, b9, b10, b11]),
                f32::from_le_bytes([b12, b13, b14, b15]),
            ])
        }

        pub(super) const fn splat(value: f32) -> Self {
            Self([value; 4])
        }

        pub(super) const fn from_array(lanes: [f32; 4]) -> Self {
            Self(lanes)
        }

        pub(super) const fn to_array(self) -> [f32; 4] {
            self.0
        }

        pub(super) fn add(self, other: Self) -> Self {
            self.zip(other, |a, b| a + b)
        }

        pub(super) fn sub(self, other: Self) -> Self {
            self.zip(other, |a, b| a - b)
        }

        pub(super) fn mul(self, other: Self) -> Self {
            self.zip(other, |a, b| a * b)
        }

        /// Sixteen signed bytes as four lanes of exact `f32`s, in order.
        pub(super) fn widen_i8(bytes: [u8; 16]) -> [Self; 4] {
            widen(bytes, |byte| f32::from(i8::from_le_bytes([byte])))
        }

        /// Sixteen unsigned bytes as four lanes of exact `f32`s, in order.
        pub(super) fn widen_u8(bytes: [u8; 16]) -> [Self; 4] {
            widen(bytes, f32::from)
        }

        fn zip(self, other: Self, op: impl Fn(f32, f32) -> f32) -> Self {
            let ([a0, a1, a2, a3], [b0, b1, b2, b3]) = (self.0, other.0);
            Self([op(a0, b0), op(a1, b1), op(a2, b2), op(a3, b3)])
        }
    }

    fn widen(bytes: [u8; 16], value: impl Fn(u8) -> f32) -> [Lanes; 4] {
        let [
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15,
        ] = bytes.map(value);
        [
            Lanes([b0, b1, b2, b3]),
            Lanes([b4, b5, b6, b7]),
            Lanes([b8, b9, b10, b11]),
            Lanes([b12, b13, b14, b15]),
        ]
    }
}
