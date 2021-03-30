//! Primitive synchronous allocation helpers

use std::mem::{self, MaybeUninit};
use std::ops::{Bound, Deref, DerefMut, RangeBounds};
use std::ptr::{self, NonNull};

use ash::prelude::VkResult as Result;
use ash::version::DeviceV1_0;
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
        non_coherent_atom_size: vk::DeviceSize,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let buffer = DedicatedBuffer::new(
            device,
            props,
            &vk::BufferCreateInfo::builder()
                .size(mem::size_of::<T>() as _)
                .usage(usage | vk::BufferUsageFlags::TRANSFER_DST)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        let staging = DedicatedMapping::uninit(
            device,
            non_coherent_atom_size,
            props,
            vk::BufferUsageFlags::TRANSFER_SRC,
        );
        Self { buffer, staging }
    }

    pub unsafe fn write(&mut self, device: &Device, x: T) {
        ptr::write(self.staging.as_mut_ptr(), x);
        device
            .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::builder()
                .memory(self.staging.memory())
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build()])
            .unwrap();
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
        non_coherent_atom_size: vk::DeviceSize,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        value: T,
    ) -> Self {
        let mut x = DedicatedMapping::uninit(device, non_coherent_atom_size, props, usage);
        ptr::write(x.as_mut_ptr(), value);
        x.assume_init()
    }

    pub unsafe fn zeroed(
        device: &Device,
        non_coherent_atom_size: vk::DeviceSize,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        Self::new(device, non_coherent_atom_size, props, usage, mem::zeroed())
    }
}

impl<T: ?Sized> DedicatedMapping<T> {
    pub unsafe fn flush(&self, device: &Device) {
        device
            .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::builder()
                .memory(self.memory())
                .offset(0)
                .size(vk::WHOLE_SIZE)
                .build()])
            .unwrap();
    }
}

impl<T> DedicatedMapping<[T]> {
    pub unsafe fn from_iter<I>(
        device: &Device,
        non_coherent_atom_size: vk::DeviceSize,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        values: I,
    ) -> Self
    where
        I: IntoIterator<Item = T>,
        I::IntoIter: ExactSizeIterator,
    {
        let mut values = values.into_iter();
        let len = values.len();
        let mut x =
            DedicatedMapping::uninit_array(device, non_coherent_atom_size, props, usage, len);
        let mut i = 0;
        while let Some(value) = values.next() {
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
        non_coherent_atom_size: vk::DeviceSize,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        values: &[T],
    ) -> Self
    where
        T: Copy,
    {
        let mut x = DedicatedMapping::uninit_array(
            device,
            non_coherent_atom_size,
            props,
            usage,
            values.len(),
        );
        ptr::copy_nonoverlapping(values.as_ptr(), x[0].as_mut_ptr(), values.len());
        x.assume_init()
    }

    pub unsafe fn zeroed_array(
        device: &Device,
        non_coherent_atom_size: vk::DeviceSize,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        size: usize,
    ) -> Self {
        let mut x =
            DedicatedMapping::uninit_array(device, non_coherent_atom_size, props, usage, size);
        for elt in &mut *x {
            ptr::write(elt.as_mut_ptr(), mem::zeroed());
        }
        x.assume_init()
    }

    pub unsafe fn flush_elts(
        &self,
        device: &Device,
        non_coherent_atom_size: vk::DeviceSize,
        range: impl RangeBounds<usize>,
    ) {
        use Bound::*;
        let first = match range.start_bound() {
            Included(&x) => x as vk::DeviceSize,
            Excluded(&x) => x as vk::DeviceSize + 1,
            Unbounded => 0,
        };
        let raw_offset = first * mem::size_of::<T>() as vk::DeviceSize;
        // Round down to first preceding non-coherent atom
        let rounded_offset = (raw_offset / non_coherent_atom_size) * non_coherent_atom_size;
        let prefix = raw_offset - rounded_offset;
        let mut size = match range.end_bound() {
            Included(&x) => {
                prefix + mem::size_of::<T>() as vk::DeviceSize * (x as vk::DeviceSize - first + 1)
            }
            Excluded(&x) => {
                prefix + mem::size_of::<T>() as vk::DeviceSize * (x as vk::DeviceSize - first)
            }
            Unbounded => vk::WHOLE_SIZE,
        };
        if size != vk::WHOLE_SIZE {
            // Round up to the end of the non-coherent atom
            size = ((size + non_coherent_atom_size - 1) / non_coherent_atom_size)
                * non_coherent_atom_size;
        }
        device
            .flush_mapped_memory_ranges(&[vk::MappedMemoryRange::builder()
                .memory(self.memory())
                .offset(rounded_offset)
                .size(size)
                .build()])
            .unwrap();
    }
}

impl<T> DedicatedMapping<MaybeUninit<T>> {
    pub unsafe fn uninit(
        device: &Device,
        non_coherent_atom_size: vk::DeviceSize,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
    ) -> Self {
        let buffer = DedicatedBuffer::new(
            device,
            props,
            &vk::BufferCreateInfo::builder()
                .size(align(mem::size_of::<T>() as _, non_coherent_atom_size))
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            vk::MemoryPropertyFlags::HOST_VISIBLE,
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
        non_coherent_atom_size: vk::DeviceSize,
        props: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        size: usize,
    ) -> Self {
        let buffer = DedicatedBuffer::new(
            device,
            props,
            &vk::BufferCreateInfo::builder()
                .size(align(
                    (size * mem::size_of::<T>()) as vk::DeviceSize,
                    non_coherent_atom_size,
                ))
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            vk::MemoryPropertyFlags::HOST_VISIBLE,
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
                &vk::MemoryAllocateInfo::builder()
                    .allocation_size(reqs.size)
                    .memory_type_index(memory_ty)
                    .push_next(&mut vk::MemoryDedicatedAllocateInfo::builder().buffer(handle)),
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
                &vk::MemoryAllocateInfo::builder()
                    .allocation_size(reqs.size)
                    .memory_type_index(memory_ty)
                    .push_next(&mut vk::MemoryDedicatedAllocateInfo::builder().image(handle)),
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
        total.size += reqs.size;
        total.memory_type_bits &= reqs.memory_type_bits;
        offsets.push(offset);
    }
    let ty = find_memory_type(props, total.memory_type_bits, flags)
        .ok_or(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY)?;
    let memory = if resources.len() == 1 {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::builder()
                .allocation_size(total.size)
                .memory_type_index(ty)
                .push_next(&mut resources[0].memory_dedicated_allocate_info()),
            None,
        )
    } else {
        device.allocate_memory(
            &vk::MemoryAllocateInfo::builder()
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
    fn memory_dedicated_allocate_info(self) -> vk::MemoryDedicatedAllocateInfo;
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
    fn memory_dedicated_allocate_info(self) -> vk::MemoryDedicatedAllocateInfo {
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
    fn memory_dedicated_allocate_info(self) -> vk::MemoryDedicatedAllocateInfo {
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
