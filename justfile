# Gates for `sextant-index`.

crate := "sextant-index"

# Every gate; CI and pre-commit run this.
gate: fmt clippy wasm test doc deslop smells

fmt:
    cargo fmt --all --check

clippy:
    cargo clippy -p {{crate}} --all-targets --all-features -- -D warnings

# The modules' documentation is the specification: every link in it must resolve.
doc:
    RUSTDOCFLAGS="-D warnings" cargo doc -p {{crate}} --all-features --document-private-items --no-deps

# Architecture: modules depend one way, stay deep, and keep their fields to themselves
# (deslop-please: `cargo install deslop-please`).
deslop:
    deslop-please {{crate}} --all-features

# The kernels' wasm SIMD backend compiles only for wasm with `simd128`; lint it there too.
wasm:
    RUSTFLAGS="-C target-feature=+simd128" cargo clippy -p {{crate}} --target wasm32-unknown-unknown -- -D warnings

# Includes the public API snapshot (`tests/public_api.rs`).
test:
    cargo test -p {{crate}} --all-features

# Proofs over every input of the decoders (`#[cfg(kani)] mod proofs`). Needs Kani
# (`cargo install --locked kani-verifier && cargo kani setup`).
kani:
    cargo kani -p {{crate}}

# Every surviving mutant is a test the suite is missing.
# A timed-out mutant hung the suite, so the tests noticed it: only a missed one (exit 2) fails.
mutants:
    #!/usr/bin/env bash
    cargo mutants -p {{crate}} --test-tool cargo || [ $? -eq 3 ]

# Duplication and complexity (thresholds in `.qlty/qlty.toml`). `qlty smells` exits 0 even with
# findings, so the gate counts the SARIF results and prints them before failing.
smells:
    #!/usr/bin/env bash
    set -euo pipefail
    count=$(qlty smells --all --sarif --no-upgrade-check | jq '[.runs[].results[]] | length')
    if [ "$count" -ne 0 ]; then
        qlty smells --all --no-snippets --no-upgrade-check
        echo "qlty: $count smell(s)" >&2
        exit 1
    fi

# Use the repository's git hooks (`.githooks`): format and lints before a commit, the gate before a
# push. Once per clone.
hooks:
    git config core.hooksPath .githooks

# Regenerate `tests/public-api.txt` after a deliberate API change; review the diff.
bless-api:
    UPDATE_EXPECT=1 cargo test -p {{crate}} --test public_api
