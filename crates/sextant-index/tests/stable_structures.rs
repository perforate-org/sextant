//! The blanket [`Memory`] impl over ic-stable-structures memories forwards every call.

use ic_stable_structures::VectorMemory;
use sextant_index::{Memory, Slab};

#[test]
fn stable_structures_memories_forward_every_call() {
    let memory = VectorMemory::default();
    assert_eq!(Memory::size(&memory), 0);
    assert_eq!(Memory::grow(&memory, 2), Ok(0));
    assert_eq!(Memory::grow(&memory, 1), Ok(2));
    assert_eq!(Memory::size(&memory), 3);
    Memory::write(&memory, 100, &[1, 2, 3]);
    let mut bytes = [0; 3];
    Memory::read(&memory, 100, &mut bytes);
    assert_eq!(bytes, [1, 2, 3]);
    assert!(Slab::open_or_init(&memory).is_ok_and(|slab| slab.occupied_tail() == 32));
}
