//! Primitive synchronous allocation helpers

use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};
use std::ptr::{self, NonNull};

use ash::prelude::VkResult as Result;
use ash::{vk, Device};

use crate::graveyard::{DeferredCleanup, Graveyard};

/// Helper for repeatedly copying fixed-size data into the same GPU buffer
pub struct Staged<T: Copy> {
    // Future work: Only allocate two buffers if non-host-visible memory exists
    buffer: DedicatedBuffer,
    staging: DedicatedMapping<MaybeUninit<T>>,
}

impl<T: Copy> Staged<T> {
    pub unsafe fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let buffer = DedicatedBuffer::new(
            device,
            props,
            &vk::BufferCreateInfo::default()
                .size(mem::size_of::<T>() as _)
                .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let staging = DedicatedMapping::uninit(device, props, vk::BufferUsageFlags::TRANSFER_SRC);
        Self { buffer, staging }
    }

    pub unsafe fn write(&mut self, x: T) {
        ptr::write(self.staging.as_mut_ptr(), x);
    }

    pub unsafe fn record_transfer(&self, device: &Device, cmd: vk::CommandBuffer) {
        device.cmd_copy_buffer(
            cmd,
            self.staging.buffer(),
            self.buffer.handle,
            &[vk::BufferCopy {
                src_offset: 0,
                dst_offset: 0,
                size: mem::size_of::<T>() as _,
            }],
        );
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer.handle
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.buffer.destroy(device);
        self.staging.destroy(device);
    }
}

/// A buffer accessible directly by the host
pub struct DedicatedMapping<T: ?Sized> {
    buffer: DedicatedBuffer,
    ptr: NonNull<T>,
}

impl<T> DedicatedMapping<T> {
    /// Create a mapped buffer
    ///
    /// # Safety
    ///
    /// `props` must be from `device`, and `T`'s alignment must not be greater than the physical
    /// device's `minMemoryMapAlignment`.
    pub unsafe fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        value: T,
    ) -> Self {
        let mut x = DedicatedMapping::uninit(device, props, usage);
        ptr::write(x.as_mut_ptr(), value);
        x.assume_init()
    }

    pub unsafe fn zeroed(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self::new(device, props, usage, mem::zeroed())
    }
}

impl<T> DedicatedMapping<[T]> {
    pub unsafe fn from_iter<I>(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        values: I,
    ) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let values = values.into_iter();
        let len = values.len();
        let mut x = DedicatedMapping::uninit_array(device, props, usage, len);
        let mut i = 0;
        for value in values {
            if i >= len {
                panic!("iterator length grew unexpectedy");
            }
            ptr::write(x[i].as_mut_ptr(), value);
            i += 1;
        }
        if i < len {
            panic!("iterator length shrank unexpectedy");
        }
        x.assume_init()
    }

    pub unsafe fn from_slice(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        values: &[T],
    ) -> Self
    where
        T: Copy,
    {
        let mut x = DedicatedMapping::uninit_array(device, props, usage, values.len());
        ptr::copy_nonoverlapping(values.as_ptr(), x[0].as_mut_ptr(), values.len());
        x.assume_init()
    }

    pub unsafe fn zeroed_array(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        size: usize,
    ) -> Self {
        let mut x = DedicatedMapping::uninit_array(device, props, usage, size);
        for elt in &mut *x {
            ptr::write(elt.as_mut_ptr(), mem::zeroed());
        }
        x.assume_init()
    }
}

impl<T> DedicatedMapping<MaybeUninit<T>> {
    pub unsafe fn uninit(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let buffer = DedicatedBuffer::new(
            device,
            props,
            &vk::BufferCreateInfo::default()
                .size(mem::size_of::<T>() as _)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let ptr = NonNull::new_unchecked(
            device
                .map_memory(
                    buffer.memory,
                    0,
                    mem::size_of::<T>() as _,
                    vk::MemoryMapFlags::default(),
                )
                .unwrap(),
        )
        .cast();
        Self { buffer, ptr }
    }

    pub unsafe fn assume_init(self) -> DedicatedMapping<T> {
        DedicatedMapping {
            buffer: self.buffer,
            ptr: self.ptr.cast(),
        }
    }
}

impl<T> DedicatedMapping<[MaybeUninit<T>]> {
    pub unsafe fn uninit_array(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        size: usize,
    ) -> Self {
        let buffer = DedicatedBuffer::new(
            device,
            props,
            &vk::BufferCreateInfo::default()
                .size((size * mem::size_of::<T>()) as vk::DeviceSize)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        let ptr = std::slice::from_raw_parts_mut(
            device
                .map_memory(
                    buffer.memory,
                    0,
                    (size * mem::size_of::<T>()) as _,
                    vk::MemoryMapFlags::default(),
                )
                .unwrap() as *mut _,
            size,
        )
        .into();
        Self { buffer, ptr }
    }

    pub unsafe fn assume_init(self) -> DedicatedMapping<[T]> {
        DedicatedMapping {
            buffer: self.buffer,
            ptr: NonNull::new_unchecked(self.ptr.as_ptr() as *mut [T]),
        }
    }
}

impl<T: ?Sized> DedicatedMapping<T> {
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }

    pub fn memory(&self) -> vk::DeviceMemory {
        self.buffer.memory
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer.handle
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        ptr::drop_in_place(self.as_ptr() as *mut T);
        self.buffer.destroy(device);
    }
}

unsafe impl<T: ?Sized> Send for DedicatedMapping<T> {}
unsafe impl<T: ?Sized> Sync for DedicatedMapping<T> {}

impl<T: ?Sized> Deref for DedicatedMapping<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { self.ptr.as_ref() }
    }
}

impl<T: ?Sized> DerefMut for DedicatedMapping<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { self.ptr.as_mut() }
    }
}

/// A buffer with its own memory allocation
#[derive(Copy, Clone)]
pub struct DedicatedBuffer {
    pub memory: vk::DeviceMemory,
    pub handle: vk::Buffer,
}

impl DedicatedBuffer {
    pub unsafe fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        info: &vk::BufferCreateInfo,
        flags: vk::MemoryPropertyFlags,
    ) -> Self {
        let handle = device.create_buffer(info, None).unwrap();
        let reqs = device.get_buffer_memory_requirements(handle);
        let memory_ty =
            find_memory_type(props, reqs.memory_type_bits, flags).expect("no matching memory type");
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(memory_ty)
                    .push_next(&mut vk::MemoryDedicatedAllocateInfo::default().buffer(handle)),
                None,
            )
            .unwrap();
        device.bind_buffer_memory(handle, memory, 0).unwrap();
        Self { handle, memory }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_buffer(self.handle, None);
        device.free_memory(self.memory, None);
    }
}

impl Default for DedicatedBuffer {
    #[inline]
    fn default() -> Self {
        Self {
            memory: vk::DeviceMemory::null(),
            handle: vk::Buffer::null(),
        }
    }
}

impl DeferredCleanup for DedicatedBuffer {
    fn inter_into(self, graveyard: &mut Graveyard) {
        graveyard.inter(self.memory);
        graveyard.inter(self.handle);
    }
}

/// An image with its own memory allocation
#[derive(Copy, Clone)]
pub struct DedicatedImage {
    pub memory: vk::DeviceMemory,
    pub handle: vk::Image,
}

impl DedicatedImage {
    pub unsafe fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        info: &vk::ImageCreateInfo,
    ) -> Self {
        let handle = device.create_image(info, None).unwrap();
        let reqs = device.get_image_memory_requirements(handle);
        let memory_ty = find_memory_type(
            props,
            reqs.memory_type_bits,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .expect("no matching memory type");
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(reqs.size)
                    .memory_type_index(memory_ty)
                    .push_next(&mut vk::MemoryDedicatedAllocateInfo::default().image(handle)),
                None,
            )
            .unwrap();
        device.bind_image_memory(handle, memory, 0).unwrap();
        Self { handle, memory }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_image(self.handle, None);
        device.free_memory(self.memory, None);
    }
}

impl Default for DedicatedImage {
    #[inline]
    fn default() -> Self {
        Self {
            memory: vk::DeviceMemory::null(),
            handle: vk::Image::null(),
        }
    }
}

impl DeferredCleanup for DedicatedImage {
    fn inter_into(self, graveyard: &mut Graveyard) {
        graveyard.inter(self.memory);
        graveyard.inter(self.handle);
    }
}

/// Allocate and bind memory for use by one or more buffers
pub unsafe fn alloc_bind(
    device: &Device,
    props: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
    resources: &[impl MemoryResource],
) -> Result<vk::DeviceMemory> {
    let mut total = vk::MemoryRequirements::default();
    total.memory_type_bits = !0;
    let mut offsets = Vec::with_capacity(resources.len());
    for resource in resources {
        let reqs = resource.get_memory_requirements(device);
        let offset = align(total.size, reqs.alignment);
        total.size = offset + reqs.size;
        total.memory_type_bits &= reqs.memory_type_bits;
        offsets.push(offset);
    }
    let ty = find_memory_type(props, total.memory_type_bits, flags)
        .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
    let memory = if resources.len() == 1 {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(total.size)
                .memory_type_index(ty)
                .push_next(&mut resources[0].memory_dedicated_allocate_info()),
            None,
        )
    } else {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(total.size)
                .memory_type_index(ty),
            None,
        )
    }?;
    for (resource, offset) in resources.iter().zip(offsets) {
        resource.bind_memory(device, memory, offset)?;
    }
    Ok(memory)
}

/// Resources that can be bound to a `vk::DeviceMemory`
pub trait MemoryResource: Copy {
    unsafe fn get_memory_requirements(self, device: &Device) -> vk::MemoryRequirements;
    unsafe fn bind_memory(
        self,
        device: &Device,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    ) -> Result<()>;
    fn memory_dedicated_allocate_info(self) -> vk::MemoryDedicatedAllocateInfo<'static>;
}

impl MemoryResource for vk::Buffer {
    #[inline]
    unsafe fn get_memory_requirements(self, device: &Device) -> vk::MemoryRequirements {
        device.get_buffer_memory_requirements(self)
    }

    #[inline]
    unsafe fn bind_memory(
        self,
        device: &Device,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    ) -> Result<()> {
        device.bind_buffer_memory(self, memory, offset)
    }

    #[inline]
    fn memory_dedicated_allocate_info(self) -> vk::MemoryDedicatedAllocateInfo<'static> {
        vk::MemoryDedicatedAllocateInfo {
            buffer: self,
            ..Default::default()
        }
    }
}

impl MemoryResource for vk::Image {
    #[inline]
    unsafe fn get_memory_requirements(self, device: &Device) -> vk::MemoryRequirements {
        device.get_image_memory_requirements(self)
    }

    #[inline]
    unsafe fn bind_memory(
        self,
        device: &Device,
        memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    ) -> Result<()> {
        device.bind_image_memory(self, memory, offset)
    }

    #[inline]
    fn memory_dedicated_allocate_info(self) -> vk::MemoryDedicatedAllocateInfo<'static> {
        vk::MemoryDedicatedAllocateInfo {
            image: self,
            ..Default::default()
        }
    }
}

pub fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..props.memory_type_count {
        if type_bits & (1 << i) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(flags)
        {
            return Some(i);
        }
    }
    None
}

/// Round `offset` up to the next multiple of `alignment`
pub fn align(offset: u64, alignment: u64) -> u64 {
    let misalignment = offset % alignment;
    let padding = if misalignment == 0 {
        0
    } else {
        alignment - misalignment
    };
    offset + padding
}

/// A single linearly-allocated buffer to be populated with transfers
///
/// Convenient for vertex/index buffers and other rarely-written random-access storage.
pub struct AppendBuffer {
    usage: vk::BufferUsageFlags,
    buffer: DedicatedBuffer,
    capacity: vk::DeviceSize,
    fill: vk::DeviceSize,
}

impl AppendBuffer {
    /// Create a buffer that can fit `capacity` bytes without growing
    pub unsafe fn with_capacity(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        capacity: vk::DeviceSize,
    ) -> Self {
        let buffer = DedicatedBuffer::new(
            device,
            props,
            &vk::BufferCreateInfo::default().size(capacity).usage(
                usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            ),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        Self {
            usage,
            buffer,
            capacity,
            fill: 0,
        }
    }

    /// The current buffer
    ///
    /// Calls to [`AppendBuffer::alloc`] invalidate previously fetched buffer handles.
    #[inline]
    pub fn buffer(&self) -> &DedicatedBuffer {
        &self.buffer
    }

    /// Allocate `size` bytes
    ///
    /// Returns the allocated offset within the buffer. Up to `size` bytes may
    /// be written to [`AppendBuffer::buffer`] at that offset via
    /// `TRANSFER_WRITE` operations recorded to `cmd`. If the buffer must be
    /// grown, writes a copy command to `cmd`, and returns the `DedicatedBuffer`
    /// to destroy after `cmd` and any other accesses finish executing.
    #[inline]
    pub unsafe fn alloc(
        &mut self,
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        cmd: vk::CommandBuffer,
        size: vk::DeviceSize,
    ) -> (vk::DeviceSize, Option<DedicatedBuffer>) {
        let offset = self.fill;
        let new_fill = self.fill + size;
        let mut old_buffer = None;
        if new_fill > self.capacity {
            old_buffer = self.grow(device, props, cmd, new_fill);
        }
        self.fill = new_fill;

        (offset, old_buffer)
    }

    unsafe fn grow(
        &mut self,
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        cmd: vk::CommandBuffer,
        new_fill: vk::DeviceSize,
    ) -> Option<DedicatedBuffer> {
        // Grow to the greater of twice our current capacity or the exact space required
        let new_cap = new_fill.max(self.capacity * 2);
        let new = DedicatedBuffer::new(
            device,
            props,
            &vk::BufferCreateInfo::default().size(new_cap).usage(
                self.usage
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
            ),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let mut old_buffer = None;
        if self.fill > 0 {
            device.cmd_pipeline_barrier2(
                cmd,
                &vk::DependencyInfo::default().buffer_memory_barriers(&[
                    vk::BufferMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
                        .buffer(self.buffer.handle)
                        .size(self.fill),
                ]),
            );
            device.cmd_copy_buffer(
                cmd,
                self.buffer.handle,
                new.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: self.fill,
                }],
            );
            old_buffer = Some(mem::replace(&mut self.buffer, new));
        }
        self.buffer = new;
        self.capacity = new_cap;
        old_buffer
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.buffer.destroy(device);
    }
}
