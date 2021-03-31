use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex};

use ash::{vk, Device};
use futures_intrusive::sync::Semaphore;

use crate::ring_alloc::{self, RingAlloc};
use crate::DedicatedMapping;

/// A host-visible circular buffer for short-lived allocations
///
/// Best for transient uses like streaming transfers. Retaining an allocation of any size will block
/// future allocations once the buffer wraps back aground.
pub struct StagingRing {
    device: Arc<Device>,
    buffer: DedicatedMapping<[u8]>,
    state: Mutex<State>,
    free: Semaphore,
}

struct State {
    alloc: RingAlloc,
}

impl StagingRing {
    pub fn new(
        device: Arc<Device>,
        props: &vk::PhysicalDeviceMemoryProperties,
        capacity: usize,
    ) -> Self {
        let buffer = unsafe {
            DedicatedMapping::zeroed_array(
                &*device,
                props,
                vk::BufferUsageFlags::TRANSFER_SRC,
                capacity,
            )
        };
        Self {
            device,
            buffer,
            state: Mutex::new(State {
                alloc: RingAlloc::new(capacity),
            }),
            free: Semaphore::new(false, capacity),
        }
    }

    #[inline]
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer.buffer()
    }

    /// Largest possible allocation
    #[inline]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Completes when sufficient space is available
    ///
    /// Yields `None` if `(size + align - 1) > self.capacity()`. No fairness guarantees, i.e. small
    /// allocations may starve large ones.
    pub async fn alloc(&self, size: usize, align: usize) -> Option<Alloc<'_>> {
        let worst_case = size + align - 1;
        if worst_case > self.capacity() {
            return None;
        }
        loop {
            self.free.acquire(worst_case).await;
            let (result, consumed) = {
                let mut state = self.state.lock().unwrap();
                let before = state.alloc.available();
                let result = state.alloc.alloc(size, align);
                (result, before - state.alloc.available())
            };
            if consumed < worst_case {
                self.free.release(worst_case - consumed);
            }
            if let Some((offset, id)) = result {
                return Some(Alloc {
                    buf: self,
                    bytes: unsafe {
                        std::slice::from_raw_parts_mut(
                            (self.buffer.as_ptr() as *const u8).add(offset) as *mut u8,
                            size,
                        )
                    },
                    id,
                });
            }
        }
    }

    fn free(&self, id: ring_alloc::Id) {
        let released = {
            let mut state = self.state.lock().unwrap();
            let before = state.alloc.available();
            state.alloc.free(id);
            state.alloc.available() - before
        };
        self.free.release(released);
    }
}

impl Drop for StagingRing {
    fn drop(&mut self) {
        unsafe {
            self.buffer.destroy(&*self.device);
        }
    }
}

/// An allocation from a `StagingRing`
pub struct Alloc<'a> {
    buf: &'a StagingRing,
    bytes: &'a mut [u8],
    id: ring_alloc::Id,
}

impl Alloc<'_> {
    pub fn offset(&self) -> vk::DeviceSize {
        self.bytes.as_ptr() as vk::DeviceSize
            - self.buf.buffer.as_ptr() as *const u8 as vk::DeviceSize
    }

    pub fn size(&self) -> vk::DeviceSize {
        self.bytes.len() as _
    }
}

impl Deref for Alloc<'_> {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        &self.bytes
    }
}

impl DerefMut for Alloc<'_> {
    fn deref_mut(&mut self) -> &mut [u8] {
        self.bytes
    }
}

impl Drop for Alloc<'_> {
    fn drop(&mut self) {
        self.buf.free(self.id);
    }
}
