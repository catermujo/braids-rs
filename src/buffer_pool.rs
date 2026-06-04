use crate::error::{BraidError, BraidResult};
use std::sync::Mutex;

#[derive(Debug, Default)]
pub(crate) struct ReusablePool<T> {
    items: Mutex<Vec<T>>,
}

impl<T: Default> ReusablePool<T> {
    pub(crate) fn checkout(&self, name: &'static str) -> BraidResult<T> {
        let mut items = self.items.lock().map_err(|_| BraidError::poisoned(name))?;
        Ok(items.pop().unwrap_or_default())
    }

    pub(crate) fn give_back(&self, name: &'static str, item: T) -> BraidResult<()> {
        let mut items = self.items.lock().map_err(|_| BraidError::poisoned(name))?;
        items.push(item);
        Ok(())
    }
}
