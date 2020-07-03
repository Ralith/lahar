use std::{
    collections::VecDeque,
    mem,
    sync::{Arc, Mutex},
};

use ash::{
    prelude::VkResult,
    version::{DeviceV1_0, DeviceV1_2},
    vk, Device,
};
use futures_intrusive::sync::ManualResetEvent;

/// Helper for running Vulkan commands asynchronously
///
/// Requires Vulkan 1.2
pub struct AsyncQueue {
    queue: vk::Queue,
    shared: Arc<Shared>,
    in_flight: VecDeque<Arc<ManualResetEvent>>,
    batches_received: u64,
    batches_completed: u64,
}

impl AsyncQueue {
    /// Construct an `AsyncQueue` that submits work to `queue` on `device`
    pub unsafe fn new(
        device: &Device,
        queue_family: u32,
        queue: vk::Queue,
    ) -> VkResult<(Self, Handle)> {
        let batches_complete = device.create_semaphore(
            &vk::SemaphoreCreateInfo::builder().push_next(
                &mut vk::SemaphoreTypeCreateInfo::builder()
                    .semaphore_type(vk::SemaphoreType::TIMELINE),
            ),
            None,
        )?;
        let batch_available = device.create_semaphore(
            &vk::SemaphoreCreateInfo::builder().push_next(
                &mut vk::SemaphoreTypeCreateInfo::builder()
                    .semaphore_type(vk::SemaphoreType::TIMELINE),
            ),
            None,
        )?;
        let shared = Arc::new(Shared {
            queue_family,
            batch_available,
            batches_complete,
            pending: Mutex::new(PendingBatch {
                batch: Batch::new(),
                seq: 1,
            }),
            highest_batch_sent: Mutex::new(0),
        });
        Ok((
            Self {
                queue,
                shared: shared.clone(),
                in_flight: VecDeque::new(),
                batches_received: 0,
                batches_completed: 0,
            },
            Handle::new(shared, device),
        ))
    }

    /// Run until all handles have been dropped
    ///
    /// Convenient for use in a dedicated background thread, driving a dedicated queue.
    pub unsafe fn run(mut self, device: &Device) {
        while Arc::get_mut(&mut self.shared).is_none() {
            device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::builder()
                        .flags(vk::SemaphoreWaitFlags::ANY)
                        .semaphores(&[self.shared.batches_complete, self.shared.batch_available])
                        .values(&[self.batches_completed + 1, self.batches_received + 1]),
                    !0,
                )
                .unwrap();
            self.drive(device);
        }
        self.destroy(device);
    }

    /// Submit currently queued work and notify completions, then return without blocking
    ///
    /// Call regularly (e.g. every frame) to make progress. Prefer using `run` unless a dedicated
    /// queue is not available.
    pub unsafe fn drive(&mut self, device: &Device) {
        self.submit_work(device);

        // Broadcast completions
        let completed = device
            .get_semaphore_counter_value(self.shared.batches_complete)
            .unwrap();
        let newly_completed = (completed - self.batches_completed) as usize;
        self.batches_completed = completed;
        for event in self.in_flight.drain(0..newly_completed) {
            event.set();
        }
    }

    unsafe fn submit_work(&mut self, device: &Device) {
        let batch = {
            let mut pending = self.shared.pending.lock().unwrap();
            if pending.batch.cmds.is_empty() {
                return;
            }
            pending.seq += 1;
            mem::replace(&mut pending.batch, Batch::new())
        };
        self.batches_received += 1;
        device
            .queue_submit(
                self.queue,
                &[vk::SubmitInfo::builder()
                    .command_buffers(&batch.cmds)
                    .signal_semaphores(&[self.shared.batches_complete])
                    .push_next(
                        &mut vk::TimelineSemaphoreSubmitInfo::builder()
                            .signal_semaphore_values(&[self.batches_received]),
                    )
                    .build()],
                vk::Fence::null(),
            )
            .unwrap();
        self.in_flight.push_back(batch.event);
    }

    /// Free Vulkan resources
    ///
    /// Must not be called until all handles have been destroyed
    pub unsafe fn destroy(&mut self, device: &Device) {
        assert_eq!(
            Arc::strong_count(&self.shared),
            1,
            "all handles must be destroyed first"
        );
        // Wait for in-flight work to complete so batches_complete can be destroyed safely
        device
            .wait_semaphores(
                &vk::SemaphoreWaitInfo::builder()
                    .semaphores(&[self.shared.batches_complete])
                    .values(&[self.batches_completed + self.in_flight.len() as u64]),
                !0,
            )
            .unwrap();
        // Unblock waiters
        for event in self.in_flight.drain(..) {
            event.set();
        }

        // Release resources
        device.destroy_semaphore(self.shared.batches_complete, None);
        device.destroy_semaphore(self.shared.batch_available, None);
    }
}

struct Shared {
    queue_family: u32,
    batches_complete: vk::Semaphore,
    batch_available: vk::Semaphore,
    pending: Mutex<PendingBatch>,
    // Must be locked while semaphore is signaled to ensure we don't signal a value that's <= a
    // previously signaled value, which is UB. We could inline this into `pending`, but it's nice to
    // be able to release that lock before waking the background thread, thereby avoiding
    // contention.
    highest_batch_sent: Mutex<u64>,
}

struct PendingBatch {
    batch: Batch,
    seq: u64,
}

struct Batch {
    cmds: Vec<vk::CommandBuffer>,
    event: Arc<ManualResetEvent>,
}

impl Batch {
    fn new() -> Self {
        Self {
            cmds: Vec::new(),
            event: Arc::new(ManualResetEvent::new(false)),
        }
    }
}

/// Allows recording commands for asynchronous execution
///
/// Record commands to the buffer returned by `cmd`, then call `flush` to dispatch work and
/// prepare a fresh command buffer.
pub struct Handle {
    cmd_pool: vk::CommandPool,
    spare_cmds: Vec<vk::CommandBuffer>,
    cmd: vk::CommandBuffer,
    in_flight: VecDeque<InFlightCmd>,
    shared: Arc<Shared>,
}

impl Handle {
    unsafe fn new(shared: Arc<Shared>, device: &Device) -> Self {
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(shared.queue_family)
                    .flags(
                        vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                            | vk::CommandPoolCreateFlags::TRANSIENT,
                    ),
                None,
            )
            .unwrap();
        let mut cmds = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(cmd_pool)
                    .command_buffer_count(32),
            )
            .unwrap();
        let cmd = cmds.pop().unwrap();
        device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

        Self {
            cmd_pool,
            spare_cmds: cmds,
            cmd,
            in_flight: VecDeque::new(),
            shared,
        }
    }

    /// Fetch the current command buffer
    pub fn cmd(&mut self) -> vk::CommandBuffer {
        self.cmd
    }

    /// Send recorded commands to be executed, returning a handle to an event that will be signaled
    /// when the command buffer has completed.
    ///
    /// Command buffers from previous calls to `cmd` must not be accessed after calling this.
    pub unsafe fn flush(&mut self, device: &Device) -> Flush {
        device.end_command_buffer(self.cmd).unwrap();
        // Enqueue the command buffer
        let (event, seq) = {
            let mut pending = self.shared.pending.lock().unwrap();
            pending.batch.cmds.push(self.cmd);
            (pending.batch.event.clone(), pending.seq)
        };
        // Send notice that there's work to be done
        {
            let mut highest_batch_sent = self.shared.highest_batch_sent.lock().unwrap();
            if seq > *highest_batch_sent {
                *highest_batch_sent = seq;
                device
                    .signal_semaphore(&vk::SemaphoreSignalInfo {
                        semaphore: self.shared.batch_available,
                        value: seq,
                        ..Default::default()
                    })
                    .unwrap();
            }
        }
        // Keep track of the sent command buffer for later reuse
        self.in_flight.push_back(InFlightCmd {
            batch: seq,
            cmd: self.cmd,
        });
        // Prepare the next command buffer to record to
        self.cmd = {
            match self.spare_cmds.pop() {
                Some(x) => x,
                None => {
                    // Recover completed command buffers
                    let completed_batch = device
                        .get_semaphore_counter_value(self.shared.batches_complete)
                        .unwrap();
                    while self
                        .in_flight
                        .front()
                        .map_or(false, |x| x.batch <= completed_batch)
                    {
                        self.spare_cmds
                            .push(self.in_flight.pop_front().unwrap().cmd);
                    }

                    // Failing that, allocate fresh ones
                    if self.spare_cmds.is_empty() {
                        let fresh = device
                            .allocate_command_buffers(
                                &vk::CommandBufferAllocateInfo::builder()
                                    .command_pool(self.cmd_pool)
                                    .command_buffer_count(32),
                            )
                            .unwrap();
                        self.spare_cmds.extend(fresh);
                    }
                    self.spare_cmds.pop().unwrap()
                }
            }
        };
        device
            .begin_command_buffer(
                self.cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
        Flush(event)
    }

    /// Construct a new handle to the same `AsyncQueue`
    pub unsafe fn clone(&self, device: &Device) -> Self {
        Self::new(self.shared.clone(), device)
    }

    /// Destroy the handle's Vulkan resources. No further calls to `flush` may be made.
    pub unsafe fn destroy(&mut self, device: &Device) {
        // Wait for in-flight command buffers to complete so it's safe to destroy the pool
        if let Some(last_sent) = self.in_flight.back() {
            device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::builder()
                        .semaphores(&[self.shared.batches_complete])
                        .values(&[last_sent.batch]),
                    !0,
                )
                .unwrap();
        }
        device.destroy_command_pool(self.cmd_pool, None);
        self.cmd = vk::CommandBuffer::null();
    }
}

struct InFlightCmd {
    batch: u64,
    cmd: vk::CommandBuffer,
}

#[derive(Clone)]
pub struct Flush(Arc<ManualResetEvent>);

impl Flush {
    /// Wait for the flushed work to complete
    pub async fn wait(&mut self) {
        self.0.wait().await
    }
}
