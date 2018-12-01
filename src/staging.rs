use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::{mem, slice};

use futures::{future::Shared, lock::Mutex, FutureExt};

use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};

use crate::{find_memory_type, ring_alloc::RingAlloc, Fence, FenceFactory, FenceSignaled};

/// Staging memory allocator.
///
/// Staging memory is host memory that can be both directly written to by the CPU and transferred
/// from by the Vulkan device.
///
/// Each allocation is associated with a [`Fence`], which must be signaled to free the associated
/// memory.
///
/// See also [`SyncAllocator`].
pub struct Allocator {
    fence_factory: FenceFactory,
    buffer: Buffer,
    alloc: RingAlloc<Shared<FenceSignaled>>,
}

impl Allocator {
    pub fn new(
        instance: &ash::Instance,
        device: Arc<ash::Device>,
        physical: vk::PhysicalDevice,
        fence_factory: FenceFactory,
        size: usize,
    ) -> Self {
        let buffer = Buffer::new(instance, device.clone(), physical, size);
        let mem = unsafe {
            let x = device
                .map_memory(buffer.memory, 0, vk::WHOLE_SIZE, Default::default())
                .unwrap();
            slice::from_raw_parts_mut(x as *mut _, size)
        };

        Self {
            fence_factory,
            buffer,
            alloc: RingAlloc::new(mem),
        }
    }

    pub async fn alloc(&mut self, bytes: usize) -> Allocation {
        let (fence, signaled) = self.fence_factory.get();
        let signaled = signaled.shared();
        loop {
            if let Ok(memory) = self.alloc.alloc(bytes, signaled.clone()) {
                return Allocation {
                    buffer: self.buffer.handle,
                    offset: (memory.as_ptr() as usize - self.alloc.base() as usize)
                        as vk::DeviceSize,
                    memory,
                    fence,
                    freed: signaled,
                };
            }
            await!(self.alloc.free_oldest().unwrap());
        }
    }
}

/// Like [`Allocator`], but thread-safe.
pub struct SyncAllocator {
    // Naive implementation. Future work: thread-local fast-path
    inner: Mutex<Allocator>,
}

impl SyncAllocator {
    pub fn new(
        instance: &ash::Instance,
        device: Arc<ash::Device>,
        physical: vk::PhysicalDevice,
        fence_factory: FenceFactory,
        size: usize,
    ) -> Self {
        Self {
            inner: Mutex::new(Allocator::new(
                instance,
                device,
                physical,
                fence_factory,
                size,
            )),
        }
    }

    pub async fn alloc(&self, bytes: usize) -> Allocation {
        let mut inner = await!(self.inner.lock());
        let (fence, signaled) = inner.fence_factory.get();
        let signaled = signaled.shared();
        loop {
            if let Ok(memory) = inner.alloc.alloc(bytes, signaled.clone()) {
                return Allocation {
                    buffer: inner.buffer.handle,
                    offset: (memory.as_ptr() as usize - inner.alloc.base() as usize)
                        as vk::DeviceSize,
                    memory,
                    fence,
                    freed: signaled,
                };
            }
            let oldest = inner.alloc.oldest().unwrap().clone();
            mem::drop(inner);
            await!(oldest);
            inner = await!(self.inner.lock());
            let _ = inner.alloc.free_oldest().unwrap();
        }
    }
}

/// Vulkan-visible host memory returned by an [`Allocator`].
///
/// # Safety
/// The behavior is undefined if memory is accessed after the [`Allocator`] is dropped.
pub struct Allocation {
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    fence: Fence,
    freed: Shared<FenceSignaled>,
    memory: *mut [u8],
}

unsafe impl Send for Allocation {}

impl Deref for Allocation {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        unsafe { &*self.memory }
    }
}

impl DerefMut for Allocation {
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { &mut *self.memory }
    }
}

impl Allocation {
    /// The Vulkan buffer associated with this memory.
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }
    /// Offset of this memory within the buffer.
    pub fn offset(&self) -> vk::DeviceSize {
        self.offset
    }
    /// The [`Fence`] that must be signaled to free the allocation.
    pub fn fence(&self) -> &Fence {
        &self.fence
    }
    /// Clonable future that completes when the memory is no longer being accessed.
    pub fn freed(&self) -> &Shared<FenceSignaled> {
        &self.freed
    }
}

struct Buffer {
    device: Arc<ash::Device>,
    memory: vk::DeviceMemory,
    handle: vk::Buffer,
}

impl Buffer {
    fn new(
        instance: &ash::Instance,
        device: Arc<ash::Device>,
        physical: vk::PhysicalDevice,
        size: usize,
    ) -> Self {
        unsafe {
            let device_props = instance.get_physical_device_memory_properties(physical);
            let handle = device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(size as vk::DeviceSize)
                        .usage(vk::BufferUsageFlags::TRANSFER_SRC),
                    None,
                )
                .unwrap();
            let reqs = device.get_buffer_memory_requirements(handle);
            let ty = find_memory_type(
                &device_props,
                reqs.memory_type_bits,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
            .unwrap();

            let dedicated_info = vk::MemoryDedicatedAllocateInfo::builder()
                .buffer(handle)
                .build();
            let memory_info = vk::MemoryAllocateInfo {
                p_next: &dedicated_info as *const _ as *const _,
                ..vk::MemoryAllocateInfo::builder()
                    .allocation_size(reqs.size)
                    .memory_type_index(ty)
                    .build()
            };
            let memory = device.allocate_memory(&memory_info, None).unwrap();
            device.bind_buffer_memory(handle, memory, 0).unwrap();

            Self {
                device,
                memory,
                handle,
            }
        }
    }
}

unsafe impl Send for Buffer {}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.handle, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
