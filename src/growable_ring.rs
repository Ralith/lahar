use std::{collections::VecDeque, mem, ptr::NonNull, sync::Mutex};

use crate::{DedicatedBuffer, TimelineRing};
use ash::{Device, vk};

/// A circular memory allocator
pub struct GrowableRing {
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    state: Mutex<State>,
    old: Mutex<VecDeque<(u64, DedicatedBuffer)>>,
}

impl GrowableRing {
    pub fn new(
        device: &Device,
        memory_properties: vk::PhysicalDeviceMemoryProperties,
        debug_utils: Option<&ash::ext::debug_utils::Device>,
        capacity: usize,
    ) -> Self {
        Self {
            memory_properties,
            state: Mutex::new(State::new(
                device,
                &memory_properties,
                debug_utils,
                capacity,
            )),
            old: Mutex::new(VecDeque::new()),
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            self.state.get_mut().unwrap().memory.destroy(device);
            for (_, mut buffer) in self.old.get_mut().unwrap().drain(..) {
                buffer.destroy(device);
            }
        }
    }

    /// The returned pointer must not be dereffed after `tick` is called with `free_at`
    pub fn alloc<T>(
        &self,
        device: &Device,
        debug_utils: Option<&ash::ext::debug_utils::Device>,
        count: usize,
        align: usize,
        free_at: u64,
    ) -> (vk::Buffer, u64, NonNull<T>) {
        let size = count * mem::size_of::<T>();
        let (buffer, offset, mapping) = {
            let mut state = self.state.lock().unwrap();
            let offset = if let Some(offset) =
                state
                    .alloc
                    .alloc(size, align.max(mem::align_of::<T>()), free_at)
            {
                offset
            } else {
                let old = mem::replace(
                    &mut *state,
                    State::new(device, &self.memory_properties, debug_utils, size * 2),
                );
                // free_at is
                self.old
                    .lock()
                    .unwrap()
                    .push_back((old.free_at, old.memory));
                state
                    .alloc
                    .alloc(size, align.max(mem::align_of::<T>()), free_at)
                    .expect("alloc failed after growing")
            };
            state.free_at = state.free_at.max(free_at);
            (state.memory.handle, offset, unsafe {
                NonNull::new_unchecked(state.mapping.as_ptr().add(offset).cast())
            })
        };
        (buffer, offset as u64, mapping)
    }

    pub unsafe fn tick(&self, device: &Device, time: u64) {
        unsafe {
            self.state.lock().unwrap().alloc.tick(time);
            let mut old = self.old.lock().unwrap();
            while let Some(&(t, _)) = old.front() {
                if t > time {
                    break;
                }
                let (_, mut buffer) = old.pop_front().unwrap();
                buffer.destroy(device);
            }
        }
    }
}

struct State {
    alloc: TimelineRing,
    memory: DedicatedBuffer,
    mapping: NonNull<u8>,
    /// Largest timeline value any allocation may be in use before
    free_at: u64,
}

impl State {
    fn new(
        device: &Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        debug_utils: Option<&ash::ext::debug_utils::Device>,
        capacity: usize,
    ) -> Self {
        let alloc = TimelineRing::new(capacity);
        let memory = unsafe {
            DedicatedBuffer::new(
                device,
                memory_properties,
                &vk::BufferCreateInfo::default()
                    .size(alloc.capacity() as vk::DeviceSize + 1)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            )
        };
        let mapping = unsafe {
            debug_utils.map(|ex| {
                ex.set_debug_utils_object_name(
                    &vk::DebugUtilsObjectNameInfoEXT::default()
                        .object_handle(memory.handle)
                        .object_name(c"staging"),
                )
                .unwrap()
            });
            NonNull::new_unchecked(
                device
                    .map_memory(
                        memory.memory,
                        0,
                        vk::WHOLE_SIZE,
                        vk::MemoryMapFlags::default(),
                    )
                    .unwrap()
                    .cast(),
            )
        };
        Self {
            alloc,
            memory,
            mapping,
            free_at: 0,
        }
    }
}

unsafe impl Send for GrowableRing {}
unsafe impl Sync for GrowableRing {}
