use std::collections::VecDeque;

use crate::RingState;

/// A circular allocator for tracking resources released by timeline semaphores
pub struct TimelineRing {
    size: usize,
    allocations: VecDeque<Alloc>,
    state: RingState,
}

impl TimelineRing {
    pub fn new(capacity: usize) -> Self {
        let size = capacity + 1;
        Self {
            size,
            allocations: VecDeque::new(),
            state: RingState::new(),
        }
    }

    /// Returns an offset into the ring to be freed when `tick` is called with `free_at`, or `None`
    /// if there is not currently enough space
    pub fn alloc(&mut self, size: usize, align: usize, free_at: u64) -> Option<usize> {
        let offset = self.state.alloc(self.size, size, align)?;
        self.allocations.push_back(Alloc {
            free_at,
            offset: self.state.head,
        });
        Some(offset)
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.size - 1
    }

    #[inline]
    pub fn free(&self) -> usize {
        if self.state.head > self.state.tail {
            return self.state.head - self.state.tail - 1;
        }
        self.state.head.max(self.size - self.state.tail - 1)
    }

    /// Free allocations that expire at or before `time`, returning whether any allocations were
    /// freed
    pub fn tick(&mut self, time: u64) -> bool {
        let alloc_count = self.allocations.len();
        while let Some(&alloc) = self.allocations.front() {
            if alloc.free_at > time {
                break;
            }
            self.state.tail = alloc.offset;
            self.allocations.pop_front();
            // Ensure we can support a maximum size allocation
            if self.state.tail == self.state.head {
                debug_assert!(self.allocations.is_empty());
                self.state.tail = self.size - 1;
                self.state.head = self.size - 1;
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
