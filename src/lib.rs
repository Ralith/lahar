//! Tools for asynchronously uploading data to a Vulkan device.

pub mod fence;
pub use self::fence::Fence;

pub mod staging;
pub mod transfer;

mod ring_alloc;

use ash::vk;

pub fn find_memory_type(
    device_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..device_props.memory_type_count {
        if type_bits & (1 << i) != 0
            && device_props.memory_types[i as usize]
                .property_flags
                .contains(flags)
        {
            return Some(i);
        }
    }
    None
}
