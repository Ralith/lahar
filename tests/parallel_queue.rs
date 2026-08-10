use ash::vk;

use lahar::{DedicatedMapping, ParallelQueue};

struct Fixture {
    instance: ash::Instance,
    device: ash::Device,
    props: vk::PhysicalDeviceMemoryProperties,
    pq: ParallelQueue,
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
                    &vk::DeviceCreateInfo::default()
                        .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(queue_family_index)
                            .queue_priorities(&[1.0])])
                        .push_next(
                            &mut vk::PhysicalDeviceVulkan12Features::default()
                                .timeline_semaphore(true),
                        ),
                    None,
                )
                .unwrap();

            let queue = device.get_device_queue(queue_family_index, 0);

            let pq = ParallelQueue::new(&device, queue_family_index, queue, None);

            Fixture {
                instance,
                device,
                props,
                pq,
            }
        }
    }

    fn destroy(mut self) {
        unsafe {
            self.pq.destroy(&self.device);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[test]
fn happy_path_buffer_copy() {
    unsafe {
        let mut fx = Fixture::new();

        // Create source buffer with known data
        let data: [u8; 256] = [42u8; 256];
        let mut src_mapping = DedicatedMapping::from_slice(
            &fx.device,
            &fx.props,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &data,
        );

        // Create destination buffer
        let mut dst_mapping = DedicatedMapping::zeroed_array(
            &fx.device,
            &fx.props,
            vk::BufferUsageFlags::TRANSFER_DST,
            data.len(),
        );

        // Create ParallelQueue handle and perform buffer copy
        let mut handle = fx.pq.handle(&fx.device);

        // Record buffer copy command
        let work = handle.begin(&fx.device);
        fx.device.cmd_copy_buffer(
            work.cmd(),
            src_mapping.buffer(),
            dst_mapping.buffer(),
            &[vk::BufferCopy::default().size(data.len() as u64)],
        );
        work.end();

        // Drive the queue to submit work
        fx.pq.drive(&fx.device);

        // Drain the queue to wait for completion
        fx.pq.drain(&fx.device);

        // Read back and verify
        let dst_data: &[u8] = dst_mapping.as_ref();
        assert_eq!(dst_data, &data, "Buffer copy did not produce expected data");

        // Clean up
        handle.destroy(&fx.device);
        src_mapping.destroy(&fx.device);
        dst_mapping.destroy(&fx.device);
        fx.destroy();
    }
}

#[test]
fn dropped_work_is_not_executed() {
    unsafe {
        let mut fx = Fixture::new();

        let data: [u8; 256] = [7u8; 256];
        let mut src_mapping = DedicatedMapping::from_slice(
            &fx.device,
            &fx.props,
            vk::BufferUsageFlags::TRANSFER_SRC,
            &data,
        );
        let mut dst_mapping = DedicatedMapping::zeroed_array(
            &fx.device,
            &fx.props,
            vk::BufferUsageFlags::TRANSFER_DST,
            data.len(),
        );

        let mut handle = fx.pq.handle(&fx.device);

        // Record a copy into a work item, then drop it without calling `end`. The command buffer
        // is reset and a reset message is sent, so the recorded work must never execute.
        let dropped = handle.begin(&fx.device);
        let dropped_time = dropped.time();
        fx.device.cmd_copy_buffer(
            dropped.cmd(),
            src_mapping.buffer(),
            dst_mapping.buffer(),
            &[vk::BufferCopy::default().size(data.len() as u64)],
        );
        drop(dropped);

        // Driving processes the reset message without submitting anything.
        fx.pq.drive(&fx.device);
        fx.pq.drain(&fx.device);

        let dst_data: &[u8] = dst_mapping.as_ref();
        assert!(
            dst_data.iter().all(|&byte| byte == 0),
            "Dropped work must not execute"
        );

        // A subsequent work item must still be submitted across the gap left by the dropped one.
        let work = handle.begin(&fx.device);
        assert_eq!(
            work.time().get(),
            dropped_time.get() + 1,
            "Dropped work should still consume a timeline slot"
        );
        fx.device.cmd_copy_buffer(
            work.cmd(),
            src_mapping.buffer(),
            dst_mapping.buffer(),
            &[vk::BufferCopy::default().size(data.len() as u64)],
        );
        work.end();

        fx.pq.drive(&fx.device);
        fx.pq.drain(&fx.device);

        let dst_data: &[u8] = dst_mapping.as_ref();
        assert_eq!(dst_data, &data, "Work after a dropped item did not execute");

        // Clean up
        handle.destroy(&fx.device);
        src_mapping.destroy(&fx.device);
        dst_mapping.destroy(&fx.device);
        fx.destroy();
    }
}
