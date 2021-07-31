use std::{mem, ptr::NonNull, slice};

use ash::{vk, Device};

use crate::DedicatedBuffer;

/// Self-growing linear allocator of host-visible coherent memory, good for a per-frame staging
/// buffer
pub struct StagingArena {
    props: vk::PhysicalDeviceMemoryProperties,
    mem: BackingMem,
    old: Vec<BackingMem>,
    cursor: usize,
    capacity: usize,
}

impl StagingArena {
    pub unsafe fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        capacity: usize,
    ) -> Self {
        Self {
            props: *props,
            mem: BackingMem::new(device, props, capacity as vk::DeviceSize),
            old: Vec::new(),
            cursor: 0,
            capacity,
        }
    }

    /// Allocate a range of `size` bytes
    pub unsafe fn alloc(&mut self, device: &Device, size: usize) -> Alloc {
        let mut offset = self.cursor;
        let mut end = offset + size;
        if end > self.capacity {
            let new = BackingMem::new(
                device,
                &self.props,
                self.capacity.max(1024) as vk::DeviceSize * 2,
            );
            let old = mem::replace(&mut self.mem, new);
            self.old.push(old);
            offset = 0;
            end = size;
        }
        self.cursor = end;
        Alloc {
            buffer: self.mem.buffer.handle,
            offset: offset as _,
            size: size as _,
        }
    }

    /// Get the storage for an allocation
    pub unsafe fn get_mut(&mut self, alloc: &Alloc) -> &mut [mem::MaybeUninit<u8>] {
        for mem in Some(&self.mem).into_iter().chain(self.old.iter().rev()) {
            if alloc.buffer == mem.buffer.handle {
                return slice::from_raw_parts_mut(
                    mem.ptr.as_ptr().add(alloc.offset as usize) as _,
                    alloc.size as usize,
                );
            }
        }
        panic!("buffer does not exist in this arena");
    }

    /// Convenience method to copy a value into an allocation
    pub unsafe fn write<T: ?Sized>(&mut self, alloc: &Alloc, src: &T) {
        let mem = self.get_mut(alloc);
        assert_eq!(mem.len(), mem::size_of_val(src), "size mismatch");
        mem.as_mut_ptr()
            .cast::<u8>()
            .copy_from_nonoverlapping(src as *const T as *const u8, mem::size_of_val(src));
    }

    /// Convenience method for alloc followed by write
    pub unsafe fn push<T: ?Sized>(&mut self, device: &Device, value: &T) -> Alloc {
        let alloc = self.alloc(device, mem::size_of_val(value));
        self.write(&alloc, value);
        alloc
    }

    /// Invalidate all prior allocations
    pub unsafe fn reset(&mut self, device: &Device) {
        self.cursor = 0;
        for mut mem in self.old.drain(..) {
            mem.buffer.destroy(device);
        }
    }

    /// Free the underlying resources
    pub unsafe fn destroy(&mut self, device: &Device) {
        self.mem.buffer.destroy(device);
        for mut mem in self.old.drain(..) {
            mem.buffer.destroy(device);
        }
    }
}

/// An allocation from a [`StagingArena`]
#[derive(Debug, Copy, Clone)]
pub struct Alloc {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub size: vk::DeviceSize,
}

struct BackingMem {
    buffer: DedicatedBuffer,
    ptr: NonNull<u8>,
}

impl BackingMem {
    unsafe fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        size: vk::DeviceSize,
    ) -> Self {
        let buffer = DedicatedBuffer::new(
            device,
            props,
            &vk::BufferCreateInfo::builder()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let ptr = NonNull::new_unchecked(
            device
                .map_memory(buffer.memory, 0, size, vk::MemoryMapFlags::default())
                .unwrap(),
        )
        .cast();
        Self { buffer, ptr }
    }
}
