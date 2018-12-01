//! Tools for asynchronously uploading data to a Vulkan device.
#![feature(await_macro, async_await, futures_api, pin, arbitrary_self_types)]

extern crate ash;
extern crate futures;

mod fence;
pub use self::fence::{Fence, FenceFactory};

mod loader;
pub use self::loader::{Loader, AsyncContext};

mod ring_alloc;
pub use self::ring_alloc::{AllocFull, RingAlloc};

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
