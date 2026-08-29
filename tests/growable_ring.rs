use ash::vk;

use lahar::{DedicatedMapping, GrowableRing};

struct Fixture {
    instance: ash::Instance,
    device: ash::Device,
    props: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    queue_family_index: u32,
}

impl Fixture {
    fn new() -> Self {
        unsafe {
            let entry = ash::Entry::linked();
            let app_info =
                vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&app_info),
                    None,
                )
                .unwrap();

            let physical_devices = instance.enumerate_physical_devices().unwrap();
            let physical_device = physical_devices
                .iter()
                .find(|&pd| {
                    let props = instance.get_physical_device_properties(*pd);
                    props.api_version >= vk::make_api_version(0, 1, 2, 0)
                })
                .copied()
                .expect("No Vulkan 1.2+ physical device found");

            let props = instance.get_physical_device_memory_properties(physical_device);

            let queue_families =
                instance.get_physical_device_queue_family_properties(physical_device);
            let queue_family_index = queue_families
                .iter()
                .position(|q| {
                    q.queue_flags
                        .intersects(vk::QueueFlags::TRANSFER | vk::QueueFlags::GRAPHICS)
                })
                .expect("No transfer-capable queue family found")
                .try_into()
                .unwrap();

            let device = instance
                .create_device(
                    physical_device,
                    &vk::DeviceCreateInfo::default().queue_create_infos(&[
                        vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(queue_family_index)
                            .queue_priorities(&[1.0]),
                    ]),
                    None,
                )
                .unwrap();

            let queue = device.get_device_queue(queue_family_index, 0);

            Fixture {
                instance,
                device,
                props,
                queue,
                queue_family_index,
            }
        }
    }

    /// Copy `size` bytes starting at `offset` in `src` into a fresh host-visible buffer, then
    /// submit the copy and wait for it to complete
    unsafe fn read_back(&self, src: vk::Buffer, offset: u64, size: usize) -> Vec<u8> {
        unsafe {
            let mut dst = DedicatedMapping::zeroed_array(
                &self.device,
                &self.props,
                vk::BufferUsageFlags::TRANSFER_DST,
                size,
            );

            let pool = self
                .device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.queue_family_index),
                    None,
                )
                .unwrap();
            let cmd = self
                .device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];
            self.device
                .begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            self.device.cmd_copy_buffer(
                cmd,
                src,
                dst.buffer(),
                &[vk::BufferCopy::default()
                    .src_offset(offset)
                    .dst_offset(0)
                    .size(size as u64)],
            );
            self.device.end_command_buffer(cmd).unwrap();

            let fence = self
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap();
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default().command_buffers(&[cmd])],
                    fence,
                )
                .unwrap();
            self.device
                .wait_for_fences(&[fence], true, u64::MAX)
                .unwrap();

            let bytes = {
                let data: &[u8] = dst.as_ref();
                data.to_vec()
            };

            self.device.destroy_fence(fence, None);
            self.device.free_command_buffers(pool, &[cmd]);
            self.device.destroy_command_pool(pool, None);
            dst.destroy(&self.device);
            bytes
        }
    }

    fn destroy(self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

/// Write through the pointer returned by `alloc` and verify the data reaches the device via a
/// buffer copy from the returned buffer and offset
#[test]
fn alloc_write_and_copy() {
    unsafe {
        let fx = Fixture::new();

        let mut ring = GrowableRing::new(&fx.device, fx.props, None, 2048);

        let pattern: Vec<u32> = (0..64).map(|i| i * 3 + 1).collect();
        let (buffer, offset, ptr) = ring.alloc::<u32>(&fx.device, None, 64, 1, 1);
        ptr.as_ptr()
            .copy_from_nonoverlapping(pattern.as_ptr(), pattern.len());

        // A second live allocation must not overlap the first
        let pattern2: Vec<u32> = (0..64).map(|i| i * 5 + 2).collect();
        let (buffer2, offset2, ptr2) = ring.alloc::<u32>(&fx.device, None, 64, 1, 1);
        let (start1, end1) = (offset, offset + 256);
        let (start2, end2) = (offset2, offset2 + 256);
        assert!(
            start1 >= end2 || start2 >= end1,
            "Live allocations must not overlap: [{start1}..{end1}) vs [{start2}..{end2})"
        );
        ptr2.as_ptr()
            .copy_from_nonoverlapping(pattern2.as_ptr(), pattern2.len());

        let expected: Vec<u8> = pattern.iter().flat_map(|x| x.to_ne_bytes()).collect();
        let expected2: Vec<u8> = pattern2.iter().flat_map(|x| x.to_ne_bytes()).collect();
        assert_eq!(fx.read_back(buffer, offset, 256), expected);
        assert_eq!(fx.read_back(buffer2, offset2, 256), expected2);

        ring.tick(&fx.device, 1);
        ring.destroy(&fx.device);
        fx.destroy();
    }
}

/// Growing the ring must not invalidate buffers allocated from the pre-growth state, which stay
/// alive until `tick` reaches their timeline value
#[test]
fn grow_preserves_live_allocations() {
    unsafe {
        let fx = Fixture::new();

        // Capacity only fits one 400-byte allocation at a time
        let mut ring = GrowableRing::new(&fx.device, fx.props, None, 512);

        let pattern: Vec<u8> = (0..400u32).map(|i| (i * 3 + 1) as u8).collect();
        let (buffer1, offset1, ptr1) = ring.alloc::<u8>(&fx.device, None, 400, 1, 1);
        ptr1.as_ptr()
            .copy_from_nonoverlapping(pattern.as_ptr(), pattern.len());

        // Doesn't fit, so the ring grows and hands out a fresh buffer
        let pattern2: Vec<u8> = (0..400u32).map(|i| (i * 5 + 2) as u8).collect();
        let (buffer2, offset2, ptr2) = ring.alloc::<u8>(&fx.device, None, 400, 1, 2);
        assert_ne!(
            buffer1, buffer2,
            "Allocation after growth should come from a new buffer"
        );
        ptr2.as_ptr()
            .copy_from_nonoverlapping(pattern2.as_ptr(), pattern2.len());

        // Growth alone must not destroy the old buffer
        ring.tick(&fx.device, 0);

        assert_eq!(fx.read_back(buffer1, offset1, 400), pattern);
        assert_eq!(fx.read_back(buffer2, offset2, 400), pattern2);

        // Past the old state's highest `free_at`, the old buffer is destroyed, and allocations
        // from the current state are unaffected
        ring.tick(&fx.device, 1);

        let pattern3: Vec<u8> = (0..400u32).map(|i| (i * 7 + 3) as u8).collect();
        let (buffer3, offset3, ptr3) = ring.alloc::<u8>(&fx.device, None, 400, 1, 5);
        ptr3.as_ptr()
            .copy_from_nonoverlapping(pattern3.as_ptr(), pattern3.len());
        assert_eq!(fx.read_back(buffer3, offset3, 400), pattern3);

        ring.tick(&fx.device, 5);
        ring.destroy(&fx.device);
        fx.destroy();
    }
}

/// `tick` frees ring space, allowing subsequent allocations to fit without growing
#[test]
fn tick_frees_ring_space() {
    unsafe {
        let fx = Fixture::new();

        let mut ring = GrowableRing::new(&fx.device, fx.props, None, 1024);

        // Fill most of the ring with allocations that all expire at 1
        let (buffer1, _, _) = ring.alloc::<u8>(&fx.device, None, 256, 1, 1);
        let (buffer2, _, _) = ring.alloc::<u8>(&fx.device, None, 256, 1, 1);
        let (buffer3, _, _) = ring.alloc::<u8>(&fx.device, None, 256, 1, 1);
        assert_eq!(buffer1, buffer2);
        assert_eq!(buffer2, buffer3);

        // Once they expire, a 700-byte allocation fits in the same buffer without growing
        ring.tick(&fx.device, 1);
        let pattern: Vec<u8> = (0..700u32).map(|i| (i * 9 + 4) as u8).collect();
        let (buffer4, offset4, ptr4) = ring.alloc::<u8>(&fx.device, None, 700, 1, 2);
        assert_eq!(
            buffer4, buffer1,
            "Allocation after `tick` should reuse the existing buffer rather than grow"
        );
        ptr4.as_ptr()
            .copy_from_nonoverlapping(pattern.as_ptr(), pattern.len());
        assert_eq!(fx.read_back(buffer4, offset4, 700), pattern);

        ring.tick(&fx.device, 2);
        ring.destroy(&fx.device);
        fx.destroy();
    }
}
