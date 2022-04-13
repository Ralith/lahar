use std::{mem, ptr::NonNull};

use crate::Graveyard;
use ash::{vk, Device};

/// A self-growing circular allocator that frees memory
pub struct StagingRing {
    state: RingState,
    memory_type: u32,
    /// VkPhysicalDeviceLimits::optimalBufferCopyOffsetAlignment
    align: usize,
    buffer: BackingMem,
    old: Vec<BackingMem>,
    frames: Box<[usize]>,
    current_frame: usize,
}

impl StagingRing {
    pub unsafe fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        limits: &vk::PhysicalDeviceLimits,
        frames: usize,
        capacity: usize,
    ) -> Self {
        let (buffer, memory_type) =
            BackingMem::new_from_props(device, props, capacity as vk::DeviceSize);
        Self {
            state: RingState::new(),
            memory_type,
            align: limits.optimal_buffer_copy_offset_alignment as usize,
            buffer,
            old: Vec::new(),
            frames: (0..frames).map(|_| 0).collect(),
            current_frame: 0,
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        for buffer in Some(&self.buffer).into_iter().chain(self.old.iter()) {
            device.destroy_buffer(buffer.buffer, None);
            device.free_memory(buffer.memory, None);
        }
    }

    pub unsafe fn push<T: ?Sized>(&mut self, device: &Device, value: &T) -> Alloc {
        let alloc = self.alloc(device, mem::size_of_val(value));
        self.buffer
            .ptr
            .as_ptr()
            .add(alloc.offset as usize)
            .cast::<u8>()
            .copy_from_nonoverlapping(value as *const _ as *const u8, mem::size_of_val(value));
        alloc
    }

    pub unsafe fn alloc(&mut self, device: &Device, mut n: usize) -> Alloc {
        let rem = n % self.align;
        if rem != 0 {
            n += self.align - rem;
        }
        let offset = match self.state.alloc(self.buffer.size, n) {
            Some(x) => x,
            None => {
                self.grow(device, n);
                self.state
                    .alloc(self.buffer.size, n)
                    .expect("insufficient space after growing")
            }
        };
        Alloc {
            buffer: self.buffer.buffer,
            offset: offset as vk::DeviceSize,
        }
    }

    unsafe fn grow(&mut self, device: &Device, min_increment: usize) {
        let new_size = min_increment.max(self.buffer.size * 2);
        let old = mem::replace(
            &mut self.buffer,
            BackingMem::new_from_ty(device, self.memory_type, new_size as vk::DeviceSize),
        );
        self.old.push(old);
        self.state = RingState::new();
    }

    /// Get the storage for an allocation
    pub unsafe fn get_mut(&self, alloc: Alloc) -> *mut u8 {
        for buffer in Some(&self.buffer).into_iter().chain(self.old.iter()) {
            if alloc.buffer == buffer.buffer {
                return buffer.ptr.as_ptr().add(alloc.offset as usize);
            }
        }
        panic!("buffer does not exist in this arena");
    }

    pub unsafe fn write<T: ?Sized>(&mut self, alloc: Alloc, data: &T) {
        self.get_mut(alloc)
            .cast::<u8>()
            .copy_from_nonoverlapping(data as *const _ as *const u8, mem::size_of_val(data));
    }

    pub fn begin_frame(&mut self, graveyard: &mut Graveyard) {
        // When the previous frame is recycled, free everything that's been allocated so far.
        self.frames[self.current_frame] = self.state.head;
        // Free everything that was allocated for the oldest frame, which we're now recycling
        self.current_frame = (self.current_frame + 1) % self.frames.len();
        self.state.tail = self.frames[self.current_frame];

        // Move pre-resize buffers from previous frame into graveyard
        for buffer in self.old.drain(..) {
            graveyard.inter(buffer.buffer);
            graveyard.inter(buffer.memory);
        }
    }
}

struct BackingMem {
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
    ptr: NonNull<u8>,
    size: usize,
}

impl BackingMem {
    unsafe fn new_from_props(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        size: vk::DeviceSize,
    ) -> (Self, u32) {
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .unwrap();
        let reqs = device.get_buffer_memory_requirements(buffer);
        let memory_ty = crate::find_memory_type(
            props,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )
        .expect("no matching memory type");
        (
            Self::new_from_buffer(device, memory_ty, size, buffer, &reqs),
            memory_ty,
        )
    }

    unsafe fn new_from_ty(device: &Device, memory_ty: u32, size: vk::DeviceSize) -> Self {
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .unwrap();
        let reqs = device.get_buffer_memory_requirements(buffer);
        Self::new_from_buffer(device, memory_ty, size, buffer, &reqs)
    }

    unsafe fn new_from_buffer(
        device: &Device,
        memory_ty: u32,
        size: vk::DeviceSize,
        buffer: vk::Buffer,
        reqs: &vk::MemoryRequirements,
    ) -> Self {
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .allocation_size(reqs.size)
                    .memory_type_index(memory_ty)
                    .push_next(&mut vk::MemoryDedicatedAllocateInfo::builder().buffer(buffer)),
                None,
            )
            .unwrap();
        device.bind_buffer_memory(buffer, memory, 0).unwrap();
        let ptr = NonNull::new_unchecked(
            device
                .map_memory(
                    memory,
                    0,
                    size as vk::DeviceSize,
                    vk::MemoryMapFlags::default(),
                )
                .unwrap(),
        )
        .cast();
        Self {
            memory,
            buffer,
            ptr,
            size: size as usize,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Alloc {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
}

struct RingState {
    /// Next empty slot
    head: usize,
    /// Oldest occupied slot
    tail: usize,
}

impl RingState {
    fn new() -> Self {
        Self { head: 0, tail: 0 }
    }

    fn alloc(&mut self, slots: usize, n: usize) -> Option<usize> {
        if self.head < self.tail {
            // head..tail is free
            if self.tail - self.head > n {
                let offset = self.head;
                self.head += n;
                Some(offset)
            } else {
                None
            }
        } else if let Some(max) = self.tail.checked_sub(1) {
            // head..slots and 0..tail is free
            if slots - self.head >= n {
                let offset = self.head;
                self.head = (self.head + n) % slots;
                Some(offset)
            } else if max >= n {
                self.head = n;
                Some(0)
            } else {
                None
            }
        } else {
            // head..slots-1 is free
            if slots - self.head > n {
                let offset = self.head;
                self.head += n;
                Some(offset)
            } else {
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke() {
        let mut r = RingState::new();
        assert_eq!(r.alloc(10, 2), Some(0));
        assert_eq!(r.alloc(10, 1), Some(2));
        assert_eq!(r.alloc(10, 7), None);
        assert_eq!(r.alloc(10, 6), Some(3));
        r.tail = 1;
        assert_eq!(r.alloc(10, 1), Some(9));
        assert_eq!(r.alloc(10, 1), None);
        r.tail = 2;
        assert_eq!(r.alloc(10, 2), None);
        assert_eq!(r.alloc(10, 1), Some(0));
        assert_eq!(r.alloc(10, 1), None);
    }
}
