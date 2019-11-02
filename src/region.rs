use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::{vk, Device};

use crate::memory::find_memory_type;

/// Simple region allocator for buffer data
pub struct BufferRegion {
    inner: Region<vk::Buffer>,
    usage: vk::BufferUsageFlags,
}

impl BufferRegion {
    pub unsafe fn new(
        device: Arc<Device>,
        props: &vk::PhysicalDeviceMemoryProperties,
        chunk_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let buffer = device
            .create_buffer(
                &vk::BufferCreateInfo::builder()
                    .size(1)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .unwrap();
        let reqs = device.get_buffer_memory_requirements(buffer);
        device.destroy_buffer(buffer, None);
        let memory_type_index = find_memory_type(
            props,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("vulkan guarantees a device local memory type exists");
        Self {
            inner: Region::new(device, memory_type_index, chunk_size),
            usage,
        }
    }

    /// Allocate `size` bytes positioned at a multiple of `alignment`
    pub fn alloc(&mut self, size: vk::DeviceSize, alignment: vk::DeviceSize) -> BufferRegionAlloc {
        if self.inner.has_capacity_for(size, alignment) {
            self.grow(size);
        }

        let offset = self.inner.alloc(size, alignment);
        BufferRegionAlloc {
            buffer: self.inner.chunks.last().unwrap().handle,
            offset,
        }
    }

    pub fn used(&self) -> vk::DeviceSize {
        self.inner.used
    }

    pub fn wasted(&self) -> vk::DeviceSize {
        self.inner.wasted
    }

    fn grow(&mut self, minimum_size: vk::DeviceSize) {
        let size = self.inner.chunk_size(minimum_size);
        unsafe {
            let handle = self
                .inner
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(size)
                        .usage(self.usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let reqs = self.inner.device.get_buffer_memory_requirements(handle);
            let memory = self
                .inner
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(reqs.size)
                        .memory_type_index(self.inner.memory_type_index)
                        .push_next(&mut vk::MemoryDedicatedAllocateInfo::builder().buffer(handle)),
                    None,
                )
                .unwrap();
            self.inner.grow(Chunk { handle, memory }, size);
        }
    }
}

impl Drop for BufferRegion {
    fn drop(&mut self) {
        for chunk in &self.inner.chunks {
            unsafe {
                self.inner.device.destroy_buffer(chunk.handle, None);
            }
        }
    }
}

pub struct BufferRegionAlloc {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
}

/// Simple region allocator for buffer data
pub struct ImageRegion {
    inner: Region<()>,
}

impl ImageRegion {
    pub unsafe fn new(
        device: Arc<Device>,
        props: &vk::PhysicalDeviceMemoryProperties,
        chunk_size: vk::DeviceSize,
    ) -> Self {
        let image = device
            .create_image(
                &vk::ImageCreateInfo::builder()
                    .image_type(vk::ImageType::TYPE_1D)
                    .format(vk::Format::R8_UNORM)
                    .extent(vk::Extent3D {
                        width: 1,
                        height: 1,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1),
                None,
            )
            .unwrap();
        let reqs = device.get_image_memory_requirements(image);
        device.destroy_image(image, None);
        let memory_type_index = find_memory_type(
            props,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("vulkan guarantees a device local memory type exists");
        Self {
            inner: Region::new(device, memory_type_index, chunk_size),
        }
    }

    /// Allocate `size` bytes positioned at a multiple of `alignment`
    pub unsafe fn alloc(&mut self, info: &vk::ImageCreateInfo) -> vk::Image {
        let handle = self.inner.device.create_image(info, None).unwrap();
        let reqs = self.inner.device.get_image_memory_requirements(handle);
        if self.inner.has_capacity_for(reqs.size, reqs.alignment) {
            self.grow(reqs.size);
        }
        let offset = self.inner.alloc(reqs.size, reqs.alignment);
        self.inner.device.bind_image_memory(handle, self.inner.chunks.last().unwrap().memory, offset).unwrap();
        handle
    }

    pub fn used(&self) -> vk::DeviceSize {
        self.inner.used
    }

    pub fn wasted(&self) -> vk::DeviceSize {
        self.inner.wasted
    }

    fn grow(&mut self, minimum_size: vk::DeviceSize) {
        let size = self.inner.chunk_size(minimum_size);
        unsafe {
            let memory = self
                .inner
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(size)
                        .memory_type_index(self.inner.memory_type_index),
                    None,
                )
                .unwrap();
            self.inner.grow(Chunk { handle: (), memory }, size);
        }
    }
}

struct Region<T> {
    device: Arc<Device>,
    chunk_size: vk::DeviceSize,
    memory_type_index: u32,
    chunks: Vec<Chunk<T>>,
    chunk_fill: vk::DeviceSize,
    used: vk::DeviceSize,
    wasted: vk::DeviceSize,
}

impl<T> Region<T> {
    unsafe fn new(device: Arc<Device>, memory_type_index: u32, chunk_size: vk::DeviceSize) -> Self {
        Self {
            device,
            chunk_size,
            memory_type_index,
            chunks: Vec::new(),
            chunk_fill: 0,
            used: 0,
            wasted: 0,
        }
    }

    /// Allocate `size` bytes positioned at a multiple of `alignment`
    fn alloc(&mut self, size: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize
    where
        T: Copy,
    {
        debug_assert!(self.has_capacity_for(size, alignment));
        let offset = align(self.chunk_fill, alignment);
        self.chunk_fill = offset + size;
        self.used += size;
        offset
    }

    fn grow(&mut self, chunk: Chunk<T>, size: vk::DeviceSize) {
        self.chunks.push(chunk);
        self.wasted += self.chunk_size.saturating_sub(self.chunk_fill);
        self.chunk_fill = 0;
        self.chunk_size = size;
    }

    fn has_capacity_for(&self, size: vk::DeviceSize, alignment: vk::DeviceSize) -> bool {
        !self.chunks.is_empty() && align(self.chunk_fill, alignment) + size <= self.chunk_size
    }

    fn chunk_size(&self, alloc_size: vk::DeviceSize) -> vk::DeviceSize {
        self.chunk_size.max(alloc_size)
    }
}

impl<T> Drop for Region<T> {
    fn drop(&mut self) {
        for chunk in &self.chunks {
            unsafe {
                self.device.free_memory(chunk.memory, None);
            }
        }
    }
}

struct Chunk<T> {
    memory: vk::DeviceMemory,
    handle: T,
}

fn align(x: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    assert!(alignment.is_power_of_two());
    (x + alignment - 1) & (!alignment + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_sanity() {
        assert_eq!(align(3, 4), 4);
        assert_eq!(align(4, 4), 4);
        assert_eq!(align(5, 4), 8);
    }
}
