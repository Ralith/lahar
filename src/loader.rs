use std::future::Future;
use std::slice;
use std::sync::Arc;
use std::cell::RefCell;

use ash::{
    version::{DeviceV1_0, InstanceV1_0},
    vk,
};
use futures::{executor::ThreadPool, task::SpawnExt};

use crate::{Fence, FenceFactory, RingAlloc, find_memory_type};

/// Helper for using a thread pool to stage Vulkan resources in host memory.
///
/// Can be cloned cheaply, producing multiple references to the same loader.
///
/// # Safety
/// The behavior is undefined if a `Loader` is dropped after the `device` or `instance` it was
/// constructed with is destroyed.
///
/// # Usage
/// A `Loader` allows long-running tasks to be performed on background threads and write results to
/// Vulkan-visible memory. For example, a texture can be loaded asynchronously by spawning a
/// function to read it from disk and decompress it into Vulkan-visible memory, returning
/// information about the memory region to the caller. The caller can then initiate a transfer from
/// the returned memory, signaling the associated fence when the transfer is complete to allow the
/// memory to be reused.
#[derive(Clone)]
pub struct Loader {
    threadpool: ThreadPool,
}

impl Loader {
    // Future work: interop with an existing threadpool
    pub fn new(instance: Arc<ash::Instance>, device: Arc<ash::Device>, physical: vk::PhysicalDevice, factory: FenceFactory, threads: usize, allocator_size: usize) -> Self {
        Self {
            threadpool: ThreadPool::builder()
                .pool_size(threads)
                .name_prefix("loader")
                .after_start(move |_| {
                    CONTEXT.with(|ctx| {
                        *ctx.borrow_mut() = Some(AsyncContext::new(&instance, device.clone(), physical, factory.clone(), allocator_size));
                    });
                })
                .before_stop(|_| {
                    CONTEXT.with(|ctx| {
                        *ctx.borrow_mut() = None;
                    })
                })
                .create().unwrap(),
        }
    }

    /// Run an asynchronous function on the thread pool.
    ///
    /// The function is given access to an allocator for Vulkan memory via `AsyncContext`.
    pub fn spawn<T: Send + 'static, F: Future<Output = T> + Send>(
        &mut self,
        f: impl for<'a> FnOnce(&'a mut AsyncContext) -> F + Send + 'static,
    ) -> impl Future<Output = T> {
        self.threadpool
            .spawn_with_handle(
                async {
                    await!(CONTEXT.with(|ctx| f(ctx.borrow_mut().as_mut().unwrap())))
                },
            )
            .unwrap()
    }
}

thread_local! {
    static CONTEXT: RefCell<Option<AsyncContext>> = RefCell::new(None);
}

/// Interface for staging resources within a `Loader`.
pub struct AsyncContext {
    fence_factory: FenceFactory,
    staging: StagingBuffer,
}

impl AsyncContext {
    fn new(instance: &ash::Instance, device: Arc<ash::Device>, physical: vk::PhysicalDevice, factory: FenceFactory, allocator_size: usize) -> Self {
        AsyncContext {
            fence_factory: factory,
            staging: StagingBuffer::new(instance, device, physical, allocator_size),
        }
    }

    /// Returns staging memory, and a fence to signal to free it.
    pub fn alloc(&mut self, bytes: usize) -> (*mut [u8], Fence) {
        let fence = self.fence_factory.get();
        loop {
            match self.staging.alloc.alloc(bytes, fence.clone()) {
                Ok(bytes) => {
                    return (bytes, fence);
                }
                Err(e) => {
                    e.wait_for.block();
                }
            }
        }
    }

    /// Get the Vulkan buffer allocated memory exists within.
    pub fn buffer(&self) -> vk::Buffer { self.staging.handle }

    /// Get the base address of the Vulkan buffer for computing offsets.
    pub fn base(&self) -> *const u8 { self.staging.alloc.base() }
}

struct StagingBuffer {
    device: Arc<ash::Device>,
    memory: vk::DeviceMemory,
    handle: vk::Buffer,
    alloc: RingAlloc<Fence>,
}

impl StagingBuffer {
    pub fn new(
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

            let dedicated_info = vk::MemoryDedicatedAllocateInfo::builder().buffer(handle).build();
            let memory_info = vk::MemoryAllocateInfo {
                p_next: &dedicated_info as *const _ as *const _,
                ..vk::MemoryAllocateInfo::builder()
                    .allocation_size(reqs.size)
                    .memory_type_index(ty)
                    .build()
            };
            let memory = device.allocate_memory(&memory_info, None).unwrap();
            device.bind_buffer_memory(handle, memory, 0).unwrap();
            let mapped = device
                .map_memory(memory, 0, vk::WHOLE_SIZE, Default::default())
                .unwrap();
            let mapped = slice::from_raw_parts_mut(mapped as *mut _, size);

            Self {
                device,
                memory,
                handle,
                alloc: RingAlloc::new(mapped),
            }
        }
    }
}

impl Drop for StagingBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.handle, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
