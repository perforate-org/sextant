//! The kernels against exact reference sums: values within rounding, early exits exactly where
//! the bound says, and total on rows and queries of any length.

#![expect(
    clippy::arithmetic_side_effects,
    clippy::float_arithmetic,
    clippy::as_conversions,
    clippy::cast_possible_wrap,
    clippy::float_cmp,
    clippy::indexing_slicing,
    reason = "reference sums in f64 over small generated inputs; exact values by hand"
)]

use proptest::prelude::*;
use sextant_index::{DotBound, F32Row, I8Row, U8Row, popcount_xnor_words};

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn l2(row: &[f64], query: &[f32]) -> f64 {
    row.iter()
        .zip(query)
        .map(|(v, &q)| (v - f64::from(q)).powi(2))
        .sum()
}

fn dot(row: &[f64], query: &[f32]) -> f64 {
    row.iter().zip(query).map(|(v, &q)| v * f64::from(q)).sum()
}

fn close(found: f32, exact: f64) -> bool {
    (f64::from(found) - exact).abs() <= 1e-4 * exact.abs().max(1.0)
}

fn suffix_norms(query: &[f32]) -> Vec<f32> {
    let mut norms = vec![0.0f32; query.len() + 1];
    for j in (0..query.len()).rev() {
        norms[j] = norms[j + 1].hypot(query[j]);
    }
    norms
}

fn i8_values(bytes: &[u8], scale: f32) -> Vec<f64> {
    let k = f64::from(scale) / 127.0;
    bytes.iter().map(|&b| f64::from(b as i8) * k).collect()
}

fn u8_values(bytes: &[u8], scale: f32) -> Vec<f64> {
    let k = f64::from(scale) / 127.0;
    bytes.iter().map(|&b| (f64::from(b) - 128.0) * k).collect()
}

#[test]
fn small_rows_by_hand() {
    let row = f32_bytes(&[1.0, 2.0, 3.0]);
    let query = [1.0, 0.0, 1.0];
    let row = F32Row::new(&row);
    assert_eq!(row.l2_squared(&query), 8.0);
    assert_eq!(row.dot(&query), 4.0);
    assert_eq!(row.dot_and_norm2(&query), (4.0, 14.0));
    assert_eq!(row.l2_squared_within(&query, 8.0), Some(8.0));
    assert_eq!(row.l2_squared_within(&query, 7.9), None);
    // 127 is the scale itself; -127 its negation; i8 128 is -128.
    let bytes = [127u8, 0x81, 0];
    assert_eq!(I8Row::new(&bytes, 2.0).dot(&[1.0, 1.0, 1.0]), 0.0);
    assert_eq!(I8Row::new(&bytes, 2.0).l2_squared(&[2.0, -2.0, 0.0]), 0.0);
    let bytes = [255u8, 1, 128];
    assert_eq!(U8Row::new(&bytes, 127.0).dot(&[1.0, 1.0, 1.0]), 0.0);
    assert_eq!(I8Row::new(&bytes, 0.0).l2_squared(&[3.0]), 9.0);
}

#[test]
fn lengths_are_the_shorter_of_row_and_query() {
    let row = f32_bytes(&[1.0; 40]);
    let short = [2.0f32; 7];
    assert_eq!(F32Row::new(&row).l2_squared(&short), 7.0);
    assert_eq!(F32Row::new(&row[..9]).dot(&short), 4.0);
    assert_eq!(F32Row::new(&[]).dot(&short), 0.0);
    assert_eq!(I8Row::new(&[127; 3], 1.0).dot(&[1.0; 50]), 3.0);
    assert_eq!(U8Row::new(&[128; 50], 1.0).l2_squared(&[1.0; 20]), 20.0);
}

#[test]
fn non_finite_distances_are_none() {
    let row = f32_bytes(&[f32::NAN, 1.0]);
    let row = F32Row::new(&row);
    assert!(row.l2_squared(&[0.0, 0.0]).is_nan());
    assert_eq!(row.l2_squared_within(&[0.0, 0.0], f32::INFINITY), None);
    let norms = suffix_norms(&[1.0, 1.0]);
    assert_eq!(
        row.dot_reaching(&[1.0, 1.0], DotBound::unit(&norms, 0.0)),
        None
    );
    let inf = [f32::INFINITY; 20];
    assert_eq!(
        I8Row::new(&[1; 20], 1.0).l2_squared_within(&inf, f32::NAN),
        None
    );
    // A NaN threshold never stops the scan.
    let finite = f32_bytes(&[1.0; 20]);
    assert_eq!(
        F32Row::new(&finite).l2_squared_within(&[0.0; 20], f32::NAN),
        Some(20.0)
    );
}

#[test]
fn popcount_counts_matching_bits() {
    assert_eq!(popcount_xnor_words(&[0; 8], &[0; 8]), 64);
    assert_eq!(popcount_xnor_words(&[0xFF; 8], &[0; 8]), 0);
    assert_eq!(popcount_xnor_words(&[0x0F; 16], &[0; 16]), 64);
    // A partial trailing word and the longer span's extra words are ignored.
    assert_eq!(popcount_xnor_words(&[0; 12], &[0; 20]), 64);
}

fn rows() -> impl Strategy<Value = (Vec<f32>, Vec<f32>, Vec<u8>, f32)> {
    (1usize..200).prop_flat_map(|dims| {
        (
            proptest::collection::vec(-4.0f32..4.0, dims),
            proptest::collection::vec(-4.0f32..4.0, dims),
            proptest::collection::vec(any::<u8>(), dims),
            0.0f32..8.0,
        )
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn kernels_match_exact_sums((row, query, codes, scale) in rows()) {
        let bytes = f32_bytes(&row);
        let exact: Vec<f64> = row.iter().copied().map(f64::from).collect();
        let f = F32Row::new(&bytes);
        prop_assert!(close(f.l2_squared(&query), l2(&exact, &query)));
        prop_assert!(close(f.dot(&query), dot(&exact, &query)));
        let (d, n) = f.dot_and_norm2(&query);
        prop_assert!(close(d, dot(&exact, &query)));
        prop_assert!(close(n, exact.iter().map(|v| v * v).sum()));
        for (values, l2_found, dot_found) in [
            (i8_values(&codes, scale), I8Row::new(&codes, scale).l2_squared(&query), I8Row::new(&codes, scale).dot(&query)),
            (u8_values(&codes, scale), U8Row::new(&codes, scale).l2_squared(&query), U8Row::new(&codes, scale).dot(&query)),
        ] {
            prop_assert!(close(l2_found, l2(&values, &query)));
            prop_assert!(close(dot_found, dot(&values, &query)));
        }
    }

    #[test]
    fn early_exits_agree_with_the_full_kernels(
        (row, query, codes, scale) in rows(),
        slack in -1.0f32..1.0,
    ) {
        let bytes = f32_bytes(&row);
        let f = F32Row::new(&bytes);
        let full = f.l2_squared(&query);
        // Within: the full distance (summed block by block, so within rounding), or None exactly
        // when it exceeds the threshold.
        let threshold = full + slack;
        match f.l2_squared_within(&query, threshold) {
            Some(found) => prop_assert!(close(found, f64::from(full))),
            None => prop_assert!(full > threshold - 1e-3),
        }
        prop_assert!(f.l2_squared_within(&query, f32::INFINITY).is_some_and(|w| close(w, f64::from(full))));
        for (within, l2) in [
            (I8Row::new(&codes, scale).l2_squared_within(&query, f32::INFINITY), I8Row::new(&codes, scale).l2_squared(&query)),
            (U8Row::new(&codes, scale).l2_squared_within(&query, f32::INFINITY), U8Row::new(&codes, scale).l2_squared(&query)),
        ] {
            prop_assert!(within.is_some_and(|w| close(w, f64::from(l2))));
        }
        // Reaching: a row whose dot reaches the threshold is never dropped.
        let norms = suffix_norms(&query);
        let max_norm = row.iter().map(|v| v * v).sum::<f32>().sqrt() * 1.001;
        let full_dot = f.dot(&query);
        let threshold = full_dot + slack;
        let found = f.dot_reaching(&query, DotBound::new(&norms, max_norm, threshold));
        if full_dot >= threshold + 1e-3 {
            prop_assert!(found.is_some());
        }
        if let Some(found) = found {
            prop_assert!(close(found, f64::from(full_dot)));
        }
        let unbounded = DotBound::new(&norms, max_norm, f32::NEG_INFINITY);
        prop_assert!(I8Row::new(&codes, scale).dot_reaching(&query, unbounded).is_some());
        prop_assert!(U8Row::new(&codes, scale).dot_reaching(&query, unbounded).is_some());
        // Missing suffix norms disable the check.
        prop_assert!(f.dot_reaching(&query, DotBound::unit(&[], f32::INFINITY)).is_some());
    }

    #[test]
    fn any_bytes_and_queries_are_total(
        bytes in proptest::collection::vec(any::<u8>(), 0..300),
        query in proptest::collection::vec(any::<f32>(), 0..100),
        scale in any::<f32>(),
        threshold in any::<f32>(),
    ) {
        let norms = suffix_norms(&query);
        let bound = DotBound::new(&norms, 1.0, threshold);
        let f = F32Row::new(&bytes);
        let _ = (f.l2_squared(&query), f.dot(&query), f.dot_and_norm2(&query));
        let _ = (f.l2_squared_within(&query, threshold), f.dot_reaching(&query, bound));
        let i = I8Row::new(&bytes, scale);
        let _ = (i.l2_squared(&query), i.dot(&query));
        let _ = (i.l2_squared_within(&query, threshold), i.dot_reaching(&query, bound));
        let u = U8Row::new(&bytes, scale);
        let _ = (u.l2_squared(&query), u.dot(&query));
        let _ = (u.l2_squared_within(&query, threshold), u.dot_reaching(&query, bound));
        let _ = popcount_xnor_words(&bytes, &bytes);
    }
}
