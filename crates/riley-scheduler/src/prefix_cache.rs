//! Index of prompt-prefix KV blocks kept for later requests.
//!
//! Each entry is one full [`KV_BLOCK_SIZE`]-token block of a prompt, keyed by
//! its exact token IDs and the entry of the block before it. Keys are compared
//! in full, so a hash collision cannot alias two prefixes. The index holds one
//! shared reference on every block it lists. A request that reuses an entry
//! holds every ancestor as well, so an entry without children and without
//! other holders always exists while any entry without other holders does.

use std::collections::HashMap;

use riley_runtime::paged_kv::{BlockId, KV_BLOCK_SIZE, KvBlockPool};

use crate::error::{SchedulerError, SchedulerResult};

/// Position in the index. Stable while any request holds the entry.
pub(crate) type EntryId = u32;

/// Parent of the first block of every prompt.
pub(crate) const ROOT: EntryId = u32::MAX;

const NONE: EntryId = u32::MAX;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct Key {
    parent: EntryId,
    tokens: [u32; KV_BLOCK_SIZE],
}

#[derive(Debug)]
struct Entry {
    key: Key,
    block: BlockId,
    children: u32,
    older: EntryId,
    newer: EntryId,
}

/// Bounded recency-ordered index; all containers are sized at construction.
#[derive(Debug)]
pub(crate) struct PrefixCache {
    capacity: usize,
    entries: Vec<Option<Entry>>,
    free_entries: Vec<EntryId>,
    index: HashMap<Key, EntryId>,
    oldest: EntryId,
    newest: EntryId,
    length: usize,
}

fn key(parent: EntryId, tokens: &[u32]) -> SchedulerResult<Key> {
    let tokens = tokens
        .try_into()
        .map_err(|_| SchedulerError::InvalidPlan {
            field: "prefix cache key",
            reason: "a key covers exactly one full KV block",
        })?;
    Ok(Key { parent, tokens })
}

impl PrefixCache {
    pub(crate) fn new(capacity: usize) -> SchedulerResult<Self> {
        let host_allocation = |resource| SchedulerError::HostAllocation {
            resource,
            requested_elements: capacity,
        };
        let top = EntryId::try_from(capacity).map_err(|_| SchedulerError::InvalidConfiguration {
            field: "prefix_cache.capacity",
            reason: "must fit the U32 entry index",
        })?;
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(capacity)
            .map_err(|_| host_allocation("prefix cache entries"))?;
        entries.resize_with(capacity, || None);
        let mut free_entries = Vec::new();
        free_entries
            .try_reserve_exact(capacity)
            .map_err(|_| host_allocation("prefix cache free entries"))?;
        free_entries.extend((0..top).rev());
        let mut index = HashMap::new();
        index
            .try_reserve(capacity)
            .map_err(|_| host_allocation("prefix cache index"))?;
        Ok(Self {
            capacity,
            entries,
            free_entries,
            index,
            oldest: NONE,
            newest: NONE,
            length: 0,
        })
    }

    pub(crate) const fn len(&self) -> usize {
        self.length
    }

    pub(crate) fn find(
        &self,
        parent: EntryId,
        tokens: &[u32],
    ) -> SchedulerResult<Option<(EntryId, BlockId)>> {
        let found = self.index.get(&key(parent, tokens)?).copied();
        found
            .map(|id| Ok((id, self.entry(id)?.block)))
            .transpose()
    }

    /// Lists a block under `parent`. The caller hands over one shared hold,
    /// must have checked that the key is absent, and must have made room.
    pub(crate) fn insert(
        &mut self,
        parent: EntryId,
        tokens: &[u32],
        block: BlockId,
    ) -> SchedulerResult<EntryId> {
        let key = key(parent, tokens)?;
        let id = self.free_entries.pop().ok_or(SchedulerError::InvalidPlan {
            field: "prefix cache",
            reason: "insert without a free entry",
        })?;
        if parent != ROOT {
            self.entry_mut(parent)?.children += 1;
        }
        self.entries[id as usize] = Some(Entry {
            key,
            block,
            children: 0,
            older: NONE,
            newer: NONE,
        });
        self.link_newest(id)?;
        self.index.insert(key, id);
        self.length += 1;
        Ok(id)
    }

    pub(crate) fn has_room(&self) -> bool {
        self.length < self.capacity
    }

    /// Marks `id` and its ancestors as used, each ancestor newer than its
    /// child, so that eviction reaches the entries without children first.
    pub(crate) fn touch_path(&mut self, id: EntryId) -> SchedulerResult<()> {
        let mut current = id;
        while current != ROOT {
            self.unlink(current)?;
            self.link_newest(current)?;
            current = self.entry(current)?.key.parent;
        }
        Ok(())
    }

    /// Frees the least recently used entry that nothing else holds.
    ///
    /// Returns `false` when every listed block is in use.
    pub(crate) fn evict_one(&mut self, pool: &mut KvBlockPool) -> SchedulerResult<bool> {
        let mut candidate = self.oldest;
        while candidate != NONE {
            let entry = self.entry(candidate)?;
            let newer = entry.newer;
            if entry.children == 0 && pool.shared_references(entry.block)? == 1 {
                self.remove(candidate, pool)?;
                return Ok(true);
            }
            candidate = newer;
        }
        Ok(false)
    }

    /// Unlists everything. Blocks that requests still hold live on until the
    /// last holder releases them, but no later request can find them.
    pub(crate) fn clear(&mut self, pool: &mut KvBlockPool) -> SchedulerResult<()> {
        for index in 0..self.entries.len() {
            if let Some(entry) = self.entries[index].take() {
                pool.release_shared_block(entry.block)?;
                self.free_entries.push(index as EntryId);
            }
        }
        self.index.clear();
        self.oldest = NONE;
        self.newest = NONE;
        self.length = 0;
        Ok(())
    }

    fn remove(&mut self, id: EntryId, pool: &mut KvBlockPool) -> SchedulerResult<()> {
        self.unlink(id)?;
        let entry = self.entries[id as usize]
            .take()
            .ok_or(SchedulerError::InvalidPlan {
                field: "prefix cache",
                reason: "removed entry is vacant",
            })?;
        pool.release_shared_block(entry.block)?;
        self.index.remove(&entry.key);
        if entry.key.parent != ROOT {
            self.entry_mut(entry.key.parent)?.children -= 1;
        }
        self.free_entries.push(id);
        self.length -= 1;
        Ok(())
    }

    fn unlink(&mut self, id: EntryId) -> SchedulerResult<()> {
        let (older, newer) = {
            let entry = self.entry(id)?;
            (entry.older, entry.newer)
        };
        if older == NONE {
            self.oldest = newer;
        } else {
            self.entry_mut(older)?.newer = newer;
        }
        if newer == NONE {
            self.newest = older;
        } else {
            self.entry_mut(newer)?.older = older;
        }
        Ok(())
    }

    fn link_newest(&mut self, id: EntryId) -> SchedulerResult<()> {
        let previous = self.newest;
        {
            let entry = self.entry_mut(id)?;
            entry.older = previous;
            entry.newer = NONE;
        }
        if previous == NONE {
            self.oldest = id;
        } else {
            self.entry_mut(previous)?.newer = id;
        }
        self.newest = id;
        Ok(())
    }

    fn entry(&self, id: EntryId) -> SchedulerResult<&Entry> {
        self.entries
            .get(id as usize)
            .and_then(Option::as_ref)
            .ok_or(SchedulerError::InvalidPlan {
                field: "prefix cache",
                reason: "entry is vacant",
            })
    }

    fn entry_mut(&mut self, id: EntryId) -> SchedulerResult<&mut Entry> {
        self.entries
            .get_mut(id as usize)
            .and_then(Option::as_mut)
            .ok_or(SchedulerError::InvalidPlan {
                field: "prefix cache",
                reason: "entry is vacant",
            })
    }
}

#[cfg(test)]
mod tests {
    use riley_runtime::paged_kv::{KvLayout, SequenceState};

    use super::*;

    fn pool(blocks: usize) -> KvBlockPool {
        KvBlockPool::new(KvLayout::checked(1, blocks, 1, 64).expect("layout")).expect("pool")
    }

    fn written(pool: &mut KvBlockPool, blocks: usize) -> SequenceState {
        let mut sequence = pool.create_sequence(1024).expect("sequence");
        let reservation = sequence
            .reserve_to(pool, blocks * KV_BLOCK_SIZE)
            .expect("reserve");
        sequence.commit(pool, reservation).expect("commit");
        sequence
    }

    fn tokens(value: u32) -> [u32; KV_BLOCK_SIZE] {
        [value; KV_BLOCK_SIZE]
    }

    #[test]
    fn keys_compare_every_token_and_the_parent() {
        let mut pool = pool(4);
        let mut sequence = written(&mut pool, 2);
        let mut cache = PrefixCache::new(4).expect("cache");
        let first = sequence.share_next_block(&mut pool).expect("share");
        let root = cache.insert(ROOT, &tokens(1), first).expect("insert");
        let second = sequence.share_next_block(&mut pool).expect("share");
        cache.insert(root, &tokens(2), second).expect("insert");

        assert_eq!(cache.find(ROOT, &tokens(1)).unwrap(), Some((root, first)));
        assert_eq!(cache.find(ROOT, &tokens(2)).unwrap(), None);
        let mut nearly = tokens(1);
        nearly[KV_BLOCK_SIZE - 1] = 9;
        assert_eq!(cache.find(ROOT, &nearly).unwrap(), None);
        assert!(cache.find(ROOT, &[1; 15]).is_err());
    }

    #[test]
    fn eviction_takes_unheld_leaves_oldest_first_and_never_a_held_block() {
        let mut pool = pool(6);
        let mut cache = PrefixCache::new(6).expect("cache");
        let mut sequence = written(&mut pool, 3);
        let mut other = written(&mut pool, 1);
        let a = sequence.share_next_block(&mut pool).expect("share");
        let a = cache.insert(ROOT, &tokens(1), a).expect("insert");
        let b = sequence.share_next_block(&mut pool).expect("share");
        let b = cache.insert(a, &tokens(2), b).expect("insert");
        let c = sequence.share_next_block(&mut pool).expect("share");
        cache.insert(b, &tokens(3), c).expect("insert");
        let d = other.share_next_block(&mut pool).expect("share");
        cache.insert(ROOT, &tokens(4), d).expect("insert");

        assert!(!cache.evict_one(&mut pool).expect("evict"), "all held");
        sequence.close(&mut pool).expect("close");
        assert_eq!(pool.stats().allocated_block_count(), 4);

        cache.touch_path(a).expect("touch");
        assert!(cache.evict_one(&mut pool).expect("evict"));
        assert_eq!(cache.find(a, &tokens(2)).unwrap().map(|e| e.0), Some(b));
        assert_eq!(cache.find(b, &tokens(3)).unwrap(), None);
        assert_eq!(cache.len(), 3);

        assert!(cache.evict_one(&mut pool).expect("evict"));
        assert!(cache.evict_one(&mut pool).expect("evict"));
        assert!(!cache.evict_one(&mut pool).expect("evict"), "other holds d");
        assert_eq!(cache.find(ROOT, &tokens(4)).unwrap().is_some(), true);
    }

    #[test]
    fn clear_leaves_held_blocks_to_their_holders() {
        let mut pool = pool(2);
        let mut cache = PrefixCache::new(2).expect("cache");
        let mut sequence = written(&mut pool, 1);
        let block = sequence.share_next_block(&mut pool).expect("share");
        cache.insert(ROOT, &tokens(1), block).expect("insert");
        cache.clear(&mut pool).expect("clear");
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.find(ROOT, &tokens(1)).unwrap(), None);
        assert_eq!(pool.shared_references(block), Ok(1));
        sequence.close(&mut pool).expect("close");
        assert_eq!(pool.stats().allocated_block_count(), 0);
    }
}
