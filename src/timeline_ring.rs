use std::collections::VecDeque;

/// A circular allocator for tracking resources released by timeline semaphores
pub struct TimelineRing {
    size: usize,
    allocations: VecDeque<Alloc>,
    /// Offset of the most recently allocated storage
    head: usize,
    /// Offset of the most recently freed storage
    tail: usize,
}

impl TimelineRing {
    pub fn new(capacity: usize) -> Self {
        let size = capacity + 1;
        Self {
            size,
            allocations: VecDeque::new(),
            head: 0,
            tail: 0,
        }
    }

    /// Returns an offset into the ring to be freed when `tick` is called with `free_at`, or `None`
    /// if there is not currently enough space
    pub fn alloc(&mut self, size: usize, align: usize, free_at: u64) -> Option<usize> {
        // self.head moves downwards
        if self.head > self.tail {
            // Try allocating between head and tail
            let unaligned = self.head - size;
            let aligned = unaligned - unaligned % align;
            if aligned <= self.tail {
                return None;
            }
            self.head = aligned;
            self.allocations.push_back(Alloc {
                free_at,
                offset: self.head,
            });
            return Some(self.head);
        } else {
            // Try allocating between head and 0
            if self.head >= size {
                // Aligning is guaranteed to be feasible, since 0 is always aligned
                self.head = self.head - size;
                self.head = self.head - self.head % align;
                self.allocations.push_back(Alloc {
                    free_at,
                    offset: self.head,
                });
                return Some(self.head);
            }
            // Try allocating between the end of the buffer and tail
            let unaligned = self.size - size;
            let aligned = unaligned - unaligned % align;
            if aligned <= self.tail {
                return None;
            }
            self.head = aligned;
            self.allocations.push_back(Alloc {
                free_at,
                offset: self.head,
            });
            return Some(self.head);
        }
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.size - 1
    }

    #[inline]
    pub fn free(&self) -> usize {
        if self.head > self.tail {
            return self.head - self.tail - 1;
        }
        self.head.max(self.size - self.tail - 1)
    }

    /// Free allocations that expire at or before `time`, returning whether any allocations were
    /// freed
    pub fn tick(&mut self, time: u64) -> bool {
        let alloc_count = self.allocations.len();
        loop {
            let alloc = match self.allocations.front() {
                Some(&x) => x,
                None => break,
            };
            if alloc.free_at > time {
                break;
            }
            self.tail = alloc.offset;
            self.allocations.pop_front();
            // Ensure we can support a maximum size allocation
            if self.tail == self.head {
                debug_assert!(self.allocations.is_empty());
                self.tail = self.size - 1;
                self.head = self.size - 1;
            }
        }
        self.allocations.len() != alloc_count
    }
}

#[derive(Copy, Clone, Debug)]
struct Alloc {
    free_at: u64,
    offset: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        let mut ring = TimelineRing::new(5);
        assert_eq!(ring.free(), 5);
        assert_eq!(ring.alloc(3, 1, 0), Some(3));
        assert_eq!(ring.free(), 2);
        assert_eq!(ring.alloc(3, 1, 1), None);
        assert_eq!(ring.alloc(2, 2, 1), None);
        assert_eq!(ring.free(), 2);
        assert_eq!(ring.alloc(2, 1, 1), Some(1));
        assert_eq!(ring.free(), 0);
        assert_eq!(ring.alloc(1, 1, 0), None);
        ring.tick(0);
        assert_eq!(ring.free(), 2);
        assert_eq!(ring.alloc(2, 2, 2), Some(4));
        assert_eq!(ring.free(), 0);
        ring.tick(2);
        assert_eq!(ring.free(), 5);
    }
}
