use std::collections::VecDeque;

/// Sub-allocates memory from a supplied region in FIFO order.
///
/// Calling code must manage waiting for old memory to become available using metadata associated
/// with each allocation.
#[derive(Debug)]
pub struct RingAlloc<T> {
    region: *mut [u8],
    /// (start, metadata) per allocation, in order
    used: VecDeque<(usize, T)>,
    /// Offset at which unused bytes begin
    head: usize,
}

unsafe impl<T: Send> Send for RingAlloc<T> {}

impl<T> RingAlloc<T> {
    /// Manage `region` of memory.
    ///
    /// `region` should henceforth only be accessed through return values of `alloc`.
    pub fn new(region: *mut [u8]) -> Self {
        Self {
            region,
            used: VecDeque::new(),
            head: 0,
        }
    }

    /// Allocate `bytes`, associated with `meta`.
    ///
    /// If the allocator is full, returns the supplied metadata alongside the metadata of the oldest
    /// allocation, which must become free before `alloc` may be called again.
    ///
    /// # Panics
    /// - if `bytes` is larger than the space provided to the allocator
    ///
    /// # Safety
    /// In addition to the usual safety rules, dereferencing the returned memory produces undefined
    /// behavior when:
    /// - called after a previous call returned `AllocatorFull` while other code still retains a
    ///   reference to the memory associated with the `wait_for` metadata.
    pub fn alloc(&mut self, bytes: usize, meta: T) -> Result<&'static mut [u8], Full> {
        assert!(
            bytes <= self.size(),
            "allocation larger than allocator size"
        );
        loop {
            if self.used.is_empty() {
                self.head = 0;
                self.used.push_back((0, meta));
                self.head = bytes;
                return Ok(unsafe { &mut (*self.region)[0..self.head] });
            }

            let tail = self.used.front().unwrap().0;
            let working_head;
            if tail < self.head {
                // Consider memory from head to end of buffer
                if self.size() - self.head >= bytes {
                    let start = self.head;
                    self.head += bytes;
                    self.used.push_back((start, meta));
                    return Ok(unsafe { &mut (*self.region)[start..self.head] });
                } else {
                    // Not enough room towards end of buffer; wrap around to start
                    working_head = 0;
                }
            } else {
                // Free memory is contiguous from head to tail
                working_head = self.head;
            }

            // Consider memory from head to tail
            if tail - working_head >= bytes {
                let start = working_head;
                self.head = working_head + bytes;
                self.used.push_back((start, meta));
                return Ok(unsafe { &mut (*self.region)[start..self.head] });
            }

            // Not enough space left
            return Err(Full);
        }
    }

    pub fn oldest(&self) -> Option<&T> {
        self.used.front().map(|x| &x.1)
    }

    pub fn free_oldest(&mut self) -> Option<T> {
        self.used.pop_front().map(|x| x.1)
    }

    /// Address at which the allocator's owned memory begins.
    pub fn base(&self) -> *const u8 {
        unsafe { (*self.region).as_ptr() }
    }
    /// Number of bytes managed by the allocator.
    pub fn size(&self) -> usize {
        unsafe { (*self.region).len() }
    }
}

/// Error produced by a full `RingAlloc`.
#[derive(Debug)]
pub struct Full;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanity() {
        let mut buf = [0; 4];
        let mut ring = RingAlloc::new(&mut buf[..]);
        assert_eq!(unsafe { (*ring.alloc(4, "initial").unwrap()).len() }, 4);
        assert_eq!(ring.alloc(4, "").unwrap_err().wait_for, "initial");
        assert_eq!(unsafe { (*ring.alloc(4, "reused").unwrap()).len() }, 4);
        assert_eq!(ring.alloc(1, "").unwrap_err().wait_for, "reused");
        assert_eq!(unsafe { (*ring.alloc(2, "a").unwrap()).len() }, 2);
        assert_eq!(unsafe { (*ring.alloc(1, "b").unwrap()).len() }, 1);
        assert_eq!(ring.alloc(3, "").unwrap_err().wait_for, "a");
        assert_eq!(ring.alloc(3, "").unwrap_err().wait_for, "b");
        assert_eq!(unsafe { (*ring.alloc(3, "c").unwrap()).len() }, 3);
        assert_eq!(unsafe { (*ring.alloc(1, "d").unwrap()).len() }, 1);
        assert_eq!(ring.alloc(1, "").unwrap_err().wait_for, "c");
        assert_eq!(unsafe { (*ring.alloc(1, "e").unwrap()).len() }, 1);
    }
}
