#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SlotKey {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Debug)]
struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

#[derive(Clone, Debug)]
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

    pub fn get(&self, key: SlotKey) -> Option<&T> {
        let slot = self.slots.get(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_ref()
    }

    pub fn get_mut(&mut self, key: SlotKey) -> Option<&mut T> {
        let slot = self.slots.get_mut(key.index as usize)?;
        if slot.generation != key.generation {
            return None;
        }
        slot.value.as_mut()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

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
}
