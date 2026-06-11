//! Stable-slot table with generation keys.
//!
//! This is useful for planner state that wants reusable holes and stale-handle detection.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
/// Opaque stable handle into a [`SlotTable`].
pub struct SlotKey {
    /// Dense slot index.
    pub index: u32,
    /// Generation counter used to invalidate stale keys.
    pub generation: u32,
}

#[derive(Clone, Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

#[derive(Clone, Debug)]
/// Stable-slot container that reuses holes and invalidates stale keys by generation.
pub struct SlotTable<T> {
    slots: Vec<Slot<T>>,
    free: Vec<usize>,
    len: usize,
}

impl<T> Default for SlotTable<T> {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            len: 0,
        }
    }
}

impl<T> SlotTable<T> {
    /// Insert one value and return its stable key.
    pub fn insert(&mut self, value: T) -> SlotKey {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index];
            slot.value = Some(value);
            self.len += 1;
            return SlotKey {
                index: index as u32,
                generation: slot.generation,
            };
        }

        let index = self.slots.len();
        self.slots.push(Slot {
            generation: 0,
            value: Some(value),
        });
        self.len += 1;
        SlotKey {
            index: index as u32,
            generation: 0,
        }
    }

    /// Remove one value by key, returning `None` for stale or missing keys.
    pub fn remove(&mut self, key: SlotKey) -> Option<T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        let value = slot.value.take()?;
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(key.index as usize);
        self.len -= 1;
        Some(value)
    }

    /// Borrow one value by key.
    pub fn get(&self, key: SlotKey) -> Option<&T> {
        let slot = self.slots.get(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_ref()
    }

    /// Mutably borrow one value by key.
    pub fn get_mut(&mut self, key: SlotKey) -> Option<&mut T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_mut()
    }

    /// Number of live values.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return whether there are no live values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total slot capacity including holes.
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Remove all live values while preserving slot storage for reuse.
    pub fn clear_reuse(&mut self) {
        self.free.clear();
        self.len = 0;
        for (index, slot) in self.slots.iter_mut().enumerate() {
            if slot.value.take().is_some() {
                slot.generation = slot.generation.wrapping_add(1);
            }
            self.free.push(index);
        }
    }

    /// Iterate over live keys and values.
    pub fn iter(&self) -> impl Iterator<Item = (SlotKey, &T)> {
        self.slots.iter().enumerate().filter_map(|(index, slot)| {
            slot.value.as_ref().map(|value| {
                (
                    SlotKey {
                        index: index as u32,
                        generation: slot.generation,
                    },
                    value,
                )
            })
        })
    }

    /// Mutably iterate over live keys and values.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (SlotKey, &mut T)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(index, slot)| {
                let generation = slot.generation;
                slot.value.as_mut().map(|value| {
                    (
                        SlotKey {
                            index: index as u32,
                            generation,
                        },
                        value,
                    )
                })
            })
    }
}

#[cfg(test)]
mod tests {
    use super::SlotTable;

    #[test]
    fn reuses_holes_and_bumps_generation() {
        let mut table = SlotTable::default();
        let first = table.insert(10);
        let second = table.insert(20);
        assert_eq!(table.remove(first), Some(10));

        let third = table.insert(30);
        assert_eq!(third.index, first.index);
        assert_ne!(third.generation, first.generation);
        assert_eq!(table.get(second), Some(&20));
        assert_eq!(table.get(third), Some(&30));
        assert_eq!(table.get(first), None);
    }

    #[test]
    fn clear_reuse_marks_unused_slots_and_allows_reinsertion() {
        let mut table = SlotTable::default();
        let first = table.insert(1);
        let second = table.insert(2);
        assert_eq!(table.remove(first), Some(1));
        assert!(table.get(first).is_none());

        table.clear_reuse();
        assert_eq!(table.len(), 0);

        let next = table.insert(3);
        assert_eq!(table.len(), 1);
        assert_ne!(next.generation, second.generation);
        assert!(table.get_mut(next).is_some());
    }

    #[test]
    fn iteration_visitor_visits_all_live_values_once() {
        let mut table = SlotTable::default();
        let first = table.insert(10);
        let second = table.insert(20);
        let _third = table.insert(30);
        assert_eq!(table.remove(second), Some(20));

        let mut seen = 0;
        for (_, value) in table.iter() {
            seen += value;
        }
        assert_eq!(seen, 40);

        let mut seen_mut = 0;
        for (_, value) in table.iter_mut() {
            *value += 5;
            seen_mut += *value;
        }
        assert_eq!(seen_mut, 50);
        assert_eq!(table.get(first), Some(&15));
    }
}
