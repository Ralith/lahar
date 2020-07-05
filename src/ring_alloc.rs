use std::collections::VecDeque;

/// State tracker for a ring buffer of contiguous variable-sized allocations with random frees
pub struct RingAlloc {
    /// Total size available to allocate
    capacity: usize,
    /// List of starting offsets, and whether they've been freed
    allocations: VecDeque<(usize, bool)>,
    /// Offset at which the next allocation will start
    head: usize,
    /// Number of allocations which have been freed
    ///
    /// Tracking this supports random freeing by making it easy to keep track of a single element
    /// inside `allocations` even as items are added/removed.
    freed: u64,
}

impl RingAlloc {
    pub fn new(capacity: usize) -> Self {
        RingAlloc {
            capacity,
            allocations: VecDeque::new(),
            head: 0,
            freed: 0,
        }
    }

    /// Returns the starting offset of a contiguous run of `size` units, or `None` if none exists.
    pub fn alloc(&mut self, size: usize, align: usize) -> Option<(usize, Id)> {
        let tail = if let Some(&(tail, _)) = self.allocations.front() {
            tail
        } else {
            if size > self.capacity {
                return None;
            }
            // No allocations, reset to initial state
            self.allocations.push_back((0, false));
            self.head = size;
            self.freed = 0;
            return Some((0, Id(0)));
        };
        let misalignment = self.head % align;
        let padding = if misalignment == 0 {
            0
        } else {
            align - misalignment
        };
        let size = size + padding;
        let id = Id(self.freed.wrapping_add(self.allocations.len() as u64));
        if self.head > tail {
            // There's a run from the head to the end of the buffer
            let free = self.capacity - self.head;
            if free >= size {
                let start = self.head;
                self.allocations.push_back((start, false));
                self.head = (start + size) % self.capacity;
                return Some((start + padding, id));
            }
            // and from the start of the buffer to the tail
            if tail >= size {
                self.allocations.push_back((0, false));
                self.head = size;
                return Some((0, id));
            }
            return None;
        }
        // Only one run, from head to tail
        let free = tail - self.head;
        if free >= size {
            let start = self.head;
            self.allocations.push_back((start, false));
            self.head = start + size;
            return Some((start + padding, id));
        }
        None
    }

    pub fn free(&mut self, id: Id) {
        self.allocations[id.0.wrapping_sub(self.freed) as usize].1 = true;
        while let Some(&(_, true)) = self.allocations.front() {
            self.allocations.pop_front();
            self.freed += 1;
        }
    }

    /// Largest allocation of alignment 1 that can currently succeed
    pub fn available(&self) -> usize {
        let tail = match self.allocations.front() {
            Some(&(x, _)) => x,
            None => return self.capacity,
        };
        if self.head == tail {
            // The empty case is caught above, so we must be full.
            return 0;
        }
        if self.head < tail {
            return tail - self.head;
        }
        tail.max(self.capacity - self.head)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Id(u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc() {
        let mut r = RingAlloc::new(4);
        let a = r.alloc(3, 1).unwrap();
        assert!(r.alloc(2, 1).is_none());
        let b = r.alloc(1, 1).unwrap();
        assert_eq!(b.0, 3);
        assert!(r.alloc(1, 1).is_none());
        r.free(a.1);
        let c = r.alloc(1, 1).unwrap();
        assert_eq!(c.0, 0);
        let d = r.alloc(2, 1).unwrap();
        assert_eq!(d.0, 1);
        assert!(r.alloc(1, 1).is_none());
        r.free(c.1);
        r.free(b.1);
        let e = r.alloc(1, 1).unwrap();
        assert_eq!(e.0, 3);
        let f = r.alloc(1, 1).unwrap();
        assert_eq!(f.0, 0);
    }

    #[test]
    fn alignment() {
        let mut r = RingAlloc::new(4);
        let _ = r.alloc(1, 1).unwrap();
        let b = r.alloc(2, 2).unwrap();
        assert!(r.alloc(1, 1).is_none());
        assert_eq!(b.0, 2);
    }

    #[test]
    fn available() {
        let mut r = RingAlloc::new(4);
        assert_eq!(r.available(), 4);
        let a = r.alloc(3, 1).unwrap();
        assert_eq!(r.available(), 1);
        let _ = r.alloc(1, 1).unwrap();
        assert_eq!(r.available(), 0);
        r.free(a.1);
        assert_eq!(r.available(), 3);
        let _ = r.alloc(1, 1).unwrap();
        assert_eq!(r.available(), 2);

        let mut r = RingAlloc::new(4);
        let a = r.alloc(1, 1).unwrap();
        let b = r.alloc(1, 1).unwrap();
        r.free(a.1);
        assert_eq!(r.available(), 2);
        r.free(b.1);
        assert_eq!(r.available(), 4);
    }
}
