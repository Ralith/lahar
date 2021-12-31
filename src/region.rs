use ash::{vk, Device};

use crate::memory::find_memory_type;

/// Simple region allocator for buffer data
pub struct BufferRegion {
    inner: Region<vk::Buffer>,
    usage: vk::BufferUsageFlags,
}

impl BufferRegion {
    pub unsafe fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        capacity: vk::DeviceSize,
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
            inner: Region::new(memory_type_index, capacity),
            usage,
        }
    }

    /// Allocate `size` bytes positioned at a multiple of `alignment`
    pub fn alloc(
        &mut self,
        device: &Device,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> BufferRegionAlloc {
        if !self.inner.has_capacity_for(size) {
            self.grow(device, size);
        }

        let offset = self.inner.alloc(size, alignment);
        BufferRegionAlloc {
            buffer: self.inner.chunks.last().unwrap().handle,
            offset,
        }
    }

    /// Bytes returned via `alloc`
    pub fn used(&self) -> vk::DeviceSize {
        self.inner.used
    }

    /// Unreachable bytes
    pub fn wasted(&self) -> vk::DeviceSize {
        self.inner.wasted
    }

    fn grow(&mut self, device: &Device, minimum_size: vk::DeviceSize) {
        let size = self.inner.next_chunk_size(minimum_size);
        unsafe {
            let handle = device
                .create_buffer(
                    &vk::BufferCreateInfo::builder()
                        .size(size)
                        .usage(self.usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE),
                    None,
                )
                .unwrap();
            let reqs = device.get_buffer_memory_requirements(handle);
            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo::builder()
                        .allocation_size(reqs.size)
                        .memory_type_index(self.inner.memory_type_index)
                        .push_next(&mut vk::MemoryDedicatedAllocateInfo::builder().buffer(handle)),
                    None,
                )
                .unwrap();
            device.bind_buffer_memory(handle, memory, 0).unwrap();
            self.inner.grow(Chunk { handle, memory }, size);
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        for chunk in &self.inner.chunks {
            device.destroy_buffer(chunk.handle, None);
        }
        self.inner.destroy(device);
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct BufferRegionAlloc {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
}

/// Simple region allocator for color images
pub struct ImageRegion {
    inner: Region<()>,
}

impl ImageRegion {
    pub unsafe fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        capacity: vk::DeviceSize,
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
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(vk::ImageUsageFlags::SAMPLED), // Aside from transient-ness, memory requirements are the same for any usage
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
            inner: Region::new(memory_type_index, capacity),
        }
    }

    /// Allocate an image
    ///
    /// The caller is responsible for freeing the `vk::Image`. `info.format` must not be a depth or
    /// stencil format.
    pub unsafe fn alloc(&mut self, device: &Device, info: &vk::ImageCreateInfo) -> vk::Image {
        debug_assert!(![
            vk::Format::D16_UNORM,
            vk::Format::X8_D24_UNORM_PACK32,
            vk::Format::D32_SFLOAT,
            vk::Format::S8_UINT,
            vk::Format::D16_UNORM_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
            vk::Format::D32_SFLOAT_S8_UINT
        ]
        .contains(&info.format));
        let handle = device.create_image(info, None).unwrap();
        let reqs = device.get_image_memory_requirements(handle);
        if !self.inner.has_capacity_for(reqs.size) {
            self.grow(device, reqs.size);
        }
        let offset = self.inner.alloc(reqs.size, reqs.alignment);
        device
            .bind_image_memory(handle, self.inner.chunks.last().unwrap().memory, offset)
            .unwrap();
        handle
    }

    /// Bytes used in images returned via `alloc`
    pub fn used(&self) -> vk::DeviceSize {
        self.inner.used
    }

    /// Unreachable bytes
    pub fn wasted(&self) -> vk::DeviceSize {
        self.inner.wasted
    }

    unsafe fn grow(&mut self, device: &Device, minimum_size: vk::DeviceSize) {
        let size = self.inner.next_chunk_size(minimum_size);
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::builder()
                    .allocation_size(size)
                    .memory_type_index(self.inner.memory_type_index),
                None,
            )
            .unwrap();
        self.inner.grow(Chunk { handle: (), memory }, size);
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.inner.destroy(device);
    }
}

struct Region<T> {
    capacity: vk::DeviceSize,
    memory_type_index: u32,
    chunks: Vec<Chunk<T>>,
    cursor: vk::DeviceSize,
    used: vk::DeviceSize,
    wasted: vk::DeviceSize,
}

impl<T> Region<T> {
    unsafe fn new(memory_type_index: u32, capacity: vk::DeviceSize) -> Self {
        Self {
            capacity,
            memory_type_index,
            chunks: Vec::new(),
            cursor: 0,
            used: 0,
            wasted: 0,
        }
    }

    /// Allocate `size` bytes positioned at a multiple of `alignment`
    fn alloc(&mut self, size: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize
    where
        T: Copy,
    {
        let off = align_down(self.cursor - size, alignment);
        self.cursor = off;
        off
    }

    fn grow(&mut self, chunk: Chunk<T>, size: vk::DeviceSize) {
        self.chunks.push(chunk);
        self.wasted += self.cursor;
        self.cursor = size;
        self.capacity = size;
    }

    fn has_capacity_for(&self, size: vk::DeviceSize) -> bool {
        self.cursor > size
    }

    fn next_chunk_size(&self, alloc_size: vk::DeviceSize) -> vk::DeviceSize {
        self.capacity.max(alloc_size) * 2
    }

    unsafe fn destroy(&mut self, device: &Device) {
        for chunk in &self.chunks {
            device.free_memory(chunk.memory, None);
        }
    }
}

struct Chunk<T> {
    memory: vk::DeviceMemory,
    handle: T,
}

fn align_down(x: vk::DeviceSize, alignment: vk::DeviceSize) -> vk::DeviceSize {
    debug_assert!(alignment.is_power_of_two());
    x & !(alignment - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align_sanity() {
        assert_eq!(align_down(3, 4), 0);
        assert_eq!(align_down(4, 4), 4);
        assert_eq!(align_down(5, 4), 4);
    }
}
