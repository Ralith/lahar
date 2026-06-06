pub struct RingState {
    /// Offset of the most recently allocated slot
    pub head: usize,
    /// Offset of the most recently freed storage
    pub tail: usize,
    /// Maximum cursor value plus one
    pub capacity: usize,
}

impl RingState {
    pub fn new(capacity: usize) -> Self {
        Self {
            head: 0,
            tail: 0,
            capacity,
        }
    }

    /// Returns the size of the largest allocation that will succeed with the
    /// given alignment. Returns `0` if no allocation of size 1 or greater can
    /// succeed.
    ///
    /// `align` must be greater than zero.
    pub fn max_alloc(&self, align: usize) -> usize {
        // The smallest address strictly greater than `tail` that is aligned
        // to `align`. Any successful allocation in the lower region must
        // leave `head` at least at this value.
        let min_head = (self.tail / align + 1) * align;

        if self.head > self.tail {
            // head is above tail: only the region (tail..head) is available
            self.head.saturating_sub(min_head)
        } else {
            // head is at or below tail: two disjoint regions are available
            //   1. [0..head] — alignment always feasible (0 is aligned)
            //   2. [tail+1..capacity) — bounded below by min_head
            let from_head = self.head;
            let from_tail = self.capacity.saturating_sub(min_head);
            from_head.max(from_tail)
        }
    }

    pub fn alloc(&mut self, size: usize, align: usize) -> Option<usize> {
        // self.head moves downwards
        if self.head > self.tail {
            // Try allocating between head and tail
            let unaligned = self.head.checked_sub(size)?;
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
            let unaligned = self.capacity.checked_sub(size)?;
            let aligned = unaligned - unaligned % align;
            if aligned <= self.tail {
                return None;
            }
            self.head = aligned;
            Some(self.head)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn larger_than_capacity_while_empty() {
        let mut r = RingState::new(128);
        assert_eq!(r.alloc(256, 1), None);
    }

    #[test]
    fn larger_than_capacity_wrapped() {
        let mut r = RingState {
            tail: 16,
            head: 32,
            capacity: 128,
        };
        assert_eq!(r.alloc(256, 1), None);
    }

    #[test]
    fn max_alloc_matches_alloc() {
        let mut r = RingState::new(10);
        // head=0, tail=0 — full capacity minus one slot
        assert_eq!(r.max_alloc(1), 9);

        r.alloc(2, 1); // head=8, tail=0 — can't reach tail (0), so max is 7
        assert_eq!(r.max_alloc(1), 7);

        r.alloc(1, 1); // head=7
        assert_eq!(r.max_alloc(1), 6);

        r.alloc(6, 1); // head=1
        r.tail = 8; // head=1, tail=8
        assert_eq!(r.max_alloc(2), 1);
        assert_eq!(r.max_alloc(1), 1);

        r.alloc(1, 2); // head=0
        assert_eq!(r.max_alloc(1), 1); // region above tail: 10 - 9 = 1
        assert_eq!(r.max_alloc(2), 0);

        r.alloc(1, 1); // head=9
        assert_eq!(r.max_alloc(1), 0); // head=9, tail=8, only 1 slot gap

        r.tail = 7; // head=9, tail=7
        assert_eq!(r.max_alloc(1), 1);
        assert_eq!(r.max_alloc(16), 0); // alignment too large

        r.alloc(1, 1); // head=8
        assert_eq!(r.max_alloc(1), 0); // head=8, tail=7
    }

    #[test]
    fn smoke() {
        let mut r = RingState::new(10);
        assert_eq!(r.alloc(2, 1), Some(8));
        assert_eq!(r.alloc(1, 1), Some(7));
        assert_eq!(r.alloc(7, 1), None);
        assert_eq!(r.alloc(6, 1), Some(1));
        r.tail = 8;
        assert_eq!(r.alloc(1, 2), Some(0));
        assert_eq!(r.alloc(1, 1), Some(9));
        assert_eq!(r.alloc(1, 1), None);
        r.tail = 7;
        assert_eq!(r.alloc(2, 1), None);
        assert_eq!(r.alloc(1, 16), None);
        assert_eq!(r.alloc(1, 1), Some(8));
        assert_eq!(r.alloc(1, 1), None);
    }
}
