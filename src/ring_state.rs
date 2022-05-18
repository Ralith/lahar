pub struct RingState {
    /// Offset of the most recently allocated slot
    pub head: usize,
    /// Offset of the most recently freed storage
    pub tail: usize,
}

impl RingState {
    pub fn new() -> Self {
        Self { head: 0, tail: 0 }
    }

    pub fn alloc(&mut self, capacity: usize, size: usize, align: usize) -> Option<usize> {
        // self.head moves downwards
        if self.head > self.tail {
            // Try allocating between head and tail
            let unaligned = self.head - size;
            let aligned = unaligned - unaligned % align;
            if aligned <= self.tail {
                return None;
            }
            self.head = aligned;
            Some(self.head)
        } else {
            // Try allocating between head and 0
            if self.head >= size {
                // Aligning is guaranteed to be feasible, since 0 is always aligned
                self.head -= size;
                self.head -= self.head % align;
                return Some(self.head);
            }
            // Try allocating between the end of the buffer and tail
            let unaligned = capacity - size;
            let aligned = unaligned - unaligned % align;
            if aligned <= self.tail {
                return None;
            }
            self.head = aligned;
            Some(self.head)
        }
    }
}
