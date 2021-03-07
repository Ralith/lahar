pub struct Arena {
    cursor: u64,
    capacity: u64,
}

impl ArenaAlloc {
    pub fn new(capacity: u64) -> Self {
        Self {
            cursor: 0,
            capacity,
        }
    }

    pub fn alloc(&mut self, size: u64) -> Option<u64> {
        let start = self.cursor;
        let next = start + size;
        if next > self.capacity {
            return None;
        }
        self.cursor = next;
        Some(start)
    }

    pub fn reset(&mut self) {
        self.cursor = 0;
    }
}
