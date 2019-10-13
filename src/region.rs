use std::sync::Arc;

use ash::version::DeviceV1_0;
use ash::{vk, Device};

use crate::memory::find_memory_type;

/// Simple region allocator for buffer data
pub struct BufferRegion {
    device: Arc<Device>,
    chunk_size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    memory_type_index: u32,
    chunks: Vec<Chunk>,
    chunk_fill: vk::DeviceSize,
    used: vk::DeviceSize,
    wasted: vk::DeviceSize,
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
        Self {
            device,
            chunk_size,
            usage,
            memory_type_index: find_memory_type(
                props,
                reqs.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("vulkan guarantees a device local memory type exists"),
            chunks: Vec::new(),
            chunk_fill: 0,
            used: 0,
            wasted: 0,
        }
    }

    /// Allocate `size` bytes positioned at a multiple of `alignment`
    pub fn alloc(&mut self, size: vk::DeviceSize, alignment: vk::DeviceSize) -> BufferRegionAlloc {
        if self.chunks.is_empty() || align(self.chunk_fill, alignment) + size > self.chunk_size {
            self.chunks.push(self.make_chunk(size));
            self.wasted += self.chunk_size.saturating_sub(self.chunk_fill);
            self.chunk_fill = 0;
        }

        let offset = align(self.chunk_fill, alignment);
        let ret = BufferRegionAlloc {
            buffer: self.chunks.last().unwrap().buffer,
            offset,
        };
        self.chunk_fill = offset + size;
        self.used += size;
        ret
    }

    pub fn used(&self) -> vk::DeviceSize {
        self.used
    }

    pub fn wasted(&self) -> vk::DeviceSize {
        self.wasted
    }

    fn make_chunk(&self, minimum_size: vk::DeviceSize) -> Chunk {
        let size = self.chunk_size.max(minimum_size);
        unsafe {
            let buffer = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(size)
                        .usage(self.usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let reqs = self.device.get_buffer_memory_requirements(buffer);
            let memory = self
                .device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(reqs.size)
                        .memory_type_index(self.memory_type_index)
                        .push_next(&mut vk::MemoryDedicatedAllocateInfo::builder().buffer(buffer)),
                    None,
                )
                .unwrap();
            Chunk { buffer, memory }
        }
    }
}

impl Drop for BufferRegion {
    fn drop(&mut self) {
        for chunk in &self.chunks {
            unsafe {
                self.device.destroy_buffer(chunk.buffer, None);
                self.device.free_memory(chunk.memory, None);
            }
        }
    }
}

pub struct BufferRegionAlloc {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
}

struct Chunk {
    memory: vk::DeviceMemory,
    buffer: vk::Buffer,
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
