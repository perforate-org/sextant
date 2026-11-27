//! Public API snapshot: any change to the exported surface must show up in review as a diff of
//! `tests/public-api.txt`. Keeping the API simple is a reviewed decision, not an accident.
//!
//! The snapshot covers the default features: what a user of the crate sees.
//!
//! Regenerate after a deliberate change with `UPDATE_EXPECT=1 cargo test -p sextant-index --test public_api`.

/// Rustdoc JSON needs a nightly; the toolchain lara-graph and Gleaph pin.
/// Bump both together.
const NIGHTLY: &str = "nightly-2026-10-26";

#[test]
fn public_api_matches_snapshot() {
    let json = rustdoc_json::Builder::default()
        .toolchain(NIGHTLY)
        .manifest_path(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .silent(true)
        .build()
        .expect("rustdoc JSON builds");
    // Blanket, auto-trait and derived impls follow from the items themselves; omitting them keeps
    // the snapshot to the decisions a reviewer actually makes.
    let api = public_api::Builder::from_rustdoc_json(json)
        .omit_blanket_impls(true)
        .omit_auto_trait_impls(true)
        .omit_auto_derived_impls(true)
        .build()
        .expect("public API parses");
    expect_test::expect_file!["public-api.txt"].assert_eq(&api.to_string());
}
