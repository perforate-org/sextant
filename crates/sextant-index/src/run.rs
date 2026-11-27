//! The run table: rows appended together share one shard, recorded once per run.
//!
//! An entry is `shard_id u32 LE | run_len u32 LE`. Runs are opened at the page's tail as rows are
//! appended; a page records at most its header's `run_capacity` of them.

/// One run: a shard and how many contiguous rows share it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RunEntry {
    shard_id: u32,
    run_len: u32,
}

impl RunEntry {
    /// Bytes of an entry.
    pub const SIZE: usize = 8;

    /// A run of `run_len` rows of shard `shard_id`.
    #[must_use]
    pub const fn new(shard_id: u32, run_len: u32) -> Self {
        Self { shard_id, run_len }
    }

    /// The shard every row of the run belongs to.
    #[must_use]
    pub const fn shard_id(self) -> u32 {
        self.shard_id
    }

    /// The rows in the run.
    #[must_use]
    pub const fn run_len(self) -> u32 {
        self.run_len
    }

    /// The entry's bytes.
    #[must_use]
    pub const fn to_bytes(self) -> [u8; Self::SIZE] {
        let [s0, s1, s2, s3] = self.shard_id.to_le_bytes();
        let [l0, l1, l2, l3] = self.run_len.to_le_bytes();
        [s0, s1, s2, s3, l0, l1, l2, l3]
    }

    /// The entry stored as `bytes`.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; Self::SIZE]) -> Self {
        let [s0, s1, s2, s3, l0, l1, l2, l3] = bytes;
        Self {
            shard_id: u32::from_le_bytes([s0, s1, s2, s3]),
            run_len: u32::from_le_bytes([l0, l1, l2, l3]),
        }
    }
}

/// Entry `index` of the run table `table`, or `None` past its end.
#[must_use]
pub fn read_run(table: &[u8], index: usize) -> Option<RunEntry> {
    let bytes = table.as_chunks::<{ RunEntry::SIZE }>().0.get(index)?;
    Some(RunEntry::from_bytes(*bytes))
}

/// Writes `entry` as entry `index` of the run table `table`; `None` past its end.
pub fn write_run(table: &mut [u8], index: usize, entry: RunEntry) -> Option<()> {
    let slot = table
        .as_chunks_mut::<{ RunEntry::SIZE }>()
        .0
        .get_mut(index)?;
    *slot = entry.to_bytes();
    Some(())
}
