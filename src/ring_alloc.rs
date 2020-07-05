use std::collections::VecDeque;

/// State tracker for a ring buffer of contiguous variable-sized allocations with random frees
pub struct RingAlloc {
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
    pub fn new() -> Self {
        RingAlloc {
            allocations: VecDeque::new(),
            head: 0,
            freed: 0,
        }
    }

    /// Returns the starting offset of a contiguous run of `size` units, or `None` if none exists.
    ///
    /// `capacity` is the total capacity of the ring.
    pub fn alloc(&mut self, capacity: usize, size: usize, align: usize) -> Option<(usize, Id)> {
        let tail = if let Some(&(tail, _)) = self.allocations.front() {
            tail
        } else {
            if size > capacity {
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
            let free = capacity - self.head;
            if free >= size {
                let start = self.head;
                self.allocations.push_back((start, false));
                self.head = (start + size) % capacity;
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
}

#[derive(Debug, Copy, Clone)]
pub struct Id(u64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity() {
        let mut r = RingAlloc::new();
        const CAP: usize = 4;
        let a = r.alloc(CAP, 3, 1).unwrap();
        assert!(r.alloc(CAP, 2, 1).is_none());
        let b = r.alloc(CAP, 1, 1).unwrap();
        assert_eq!(b.0, 3);
        assert!(r.alloc(CAP, 1, 1).is_none());
        r.free(a.1);
        let c = r.alloc(CAP, 1, 1).unwrap();
        assert_eq!(c.0, 0);
        let d = r.alloc(CAP, 2, 1).unwrap();
        assert_eq!(d.0, 1);
        assert!(r.alloc(CAP, 1, 1).is_none());
        r.free(c.1);
        r.free(b.1);
        let e = r.alloc(CAP, 1, 1).unwrap();
        assert_eq!(e.0, 3);
        let f = r.alloc(CAP, 1, 1).unwrap();
        assert_eq!(f.0, 0);
    }

    #[test]
    fn alignment() {
        let mut r = RingAlloc::new();
        const CAP: usize = 4;
        let _ = r.alloc(CAP, 1, 1).unwrap();
        let b = r.alloc(CAP, 2, 2).unwrap();
        assert!(r.alloc(CAP, 1, 1).is_none());
        assert_eq!(b.0, 2);
    }
}
