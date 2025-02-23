use std::{
    collections::{BinaryHeap, VecDeque},
    num::NonZeroU64,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

use ash::{Device, vk};

pub struct ParallelQueue {
    shared: Arc<Shared>,
    recv: mpsc::Receiver<Message>,
    /// The lowest value that will not be signaled by work submitted to the queue so far
    first_unsubmitted: u64,
    /// The lowest value not yet reached by the semaphore
    first_unsignaled: u64,
    pending: BinaryHeap<Message>,
    queue: vk::Queue,
}

impl ParallelQueue {
    /// # Safety
    /// `queue_family_index` must be associated with `queue` under `device`
    pub unsafe fn new(device: &Device, queue_family_index: u32, queue: vk::Queue) -> Self {
        unsafe {
            let (send, recv) = mpsc::channel();
            let semaphore = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(
                        &mut vk::SemaphoreTypeCreateInfo::default()
                            .semaphore_type(vk::SemaphoreType::TIMELINE),
                    ),
                    None,
                )
                .unwrap();
            let shared = Arc::new(Shared {
                queue_family_index,
                first_unallocated: AtomicU64::new(1),
                semaphore,
                send,
            });
            Self {
                shared,
                recv,
                first_unsubmitted: 1,
                first_unsignaled: 1,
                pending: BinaryHeap::new(),
                queue,
            }
        }
    }

    /// # Safety
    /// `device` must match that passed to `new` and no work may be in flight, as determined by
    /// calling `drain` after all work has been submitted.
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.shared.semaphore, None);
        }
    }

    #[inline]
    pub fn semaphore(&self) -> vk::Semaphore {
        self.shared.semaphore
    }

    /// # Safety
    /// `device` must match that passed to `new`
    pub unsafe fn drive(&mut self, device: &Device) {
        unsafe {
            while let Ok(work) = self.recv.try_recv() {
                self.pending.push(work);
            }
            let mut cmds = Vec::new();
            // Queue up the next contiguous run of work. By mandating that work be submitted in order
            // without gaps, we make the semaphore counter value a reliable indicator of when a work
            // item's execution is complete.
            while self
                .pending
                .peek()
                .map_or(false, |work| work.time().get() == self.first_unsubmitted)
            {
                if let Message::Execute(work) = self.pending.pop().unwrap() {
                    cmds.push(work.cmd);
                }
                self.first_unsubmitted += 1;
            }
            if cmds.is_empty() {
                return;
            }
            device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::default()
                        .command_buffers(&cmds)
                        .signal_semaphores(&[self.shared.semaphore])
                        .push_next(
                            &mut vk::TimelineSemaphoreSubmitInfo::default()
                                .signal_semaphore_values(&[self.first_unsubmitted - 1]),
                        )],
                    vk::Fence::null(),
                )
                .unwrap();
        }
    }

    /// Wait until a submission is complete or `wake` reaches `wake_value`, returning the current
    /// timeline value
    ///
    /// Useful for driving the queue on a dedicated background thread.
    ///
    /// # Safety
    ///
    /// `device` must match that passed to `new`, and `wake` must be a valid timeline semaphore from
    /// `device`
    #[inline]
    pub unsafe fn park(&mut self, device: &Device, wake: vk::Semaphore, wake_value: u64) -> u64 {
        unsafe {
            device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.shared.semaphore, wake])
                        .values(&[self.first_unsignaled, wake_value]),
                    !0,
                )
                .unwrap();
            let complete = device
                .get_semaphore_counter_value(self.shared.semaphore)
                .unwrap();
            self.first_unsignaled = complete + 1;
            complete
        }
    }

    /// Wait until all work submitted so far is complete
    ///
    /// Does not wait for work that has been allocated but not submitted.
    ///
    /// # Safety
    ///
    /// `device` must match that passed to `new`
    #[inline]
    pub unsafe fn drain(&mut self, device: &Device) {
        unsafe {
            device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.shared.semaphore])
                        .values(&[self.first_unsubmitted - 1]),
                    !0,
                )
                .unwrap();
            self.first_unsignaled = self.first_unsubmitted;
        }
    }

    /// # Safety
    /// `device` must match that passed to `new`
    pub unsafe fn handle(&self, device: &Device) -> Handle {
        unsafe { self.shared.handle(device) }
    }
}

struct Shared {
    queue_family_index: u32,
    semaphore: vk::Semaphore,
    /// The lowest value that has not yet been associated with any work
    first_unallocated: AtomicU64,
    /// For cloning by handles.
    send: mpsc::Sender<Message>,
}

impl Shared {
    /// # Safety
    /// `device` must match that passed to [`ParallelQueue::new`]
    unsafe fn handle(self: &Arc<Self>, device: &Device) -> Handle {
        unsafe {
            let cmd_pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(self.queue_family_index)
                        .flags(
                            vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER
                                | vk::CommandPoolCreateFlags::TRANSIENT,
                        ),
                    None,
                )
                .unwrap();
            let spare_cmds = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(cmd_pool)
                        .command_buffer_count(32),
                )
                .unwrap();
            Handle {
                send: self.send.clone(),
                shared: self.clone(),
                cmd_pool,
                spare_cmds,
                in_flight: VecDeque::new(),
            }
        }
    }
}

#[derive(Copy, Clone)]
pub struct Work {
    /// Command buffer to record commands onto
    pub cmd: vk::CommandBuffer,
    /// Value the timeline semaphore will reach when `cmd` has been executed
    pub time: NonZeroU64,
}

pub struct Handle {
    send: mpsc::Sender<Message>,
    shared: Arc<Shared>,
    cmd_pool: vk::CommandPool,
    spare_cmds: Vec<vk::CommandBuffer>,
    in_flight: VecDeque<Work>,
}

impl Handle {
    /// # Safety
    /// `device` must match that passed to [`ParallelQueue::new`] and no work may be in flight
    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_command_pool(self.cmd_pool, None);
        }
    }

    #[inline]
    pub fn semaphore(&self) -> vk::Semaphore {
        self.shared.semaphore
    }

    #[inline]
    pub fn cmd_pool(&self) -> vk::CommandPool {
        self.cmd_pool
    }

    /// Obtain a command buffer that commands may be recorded into
    ///
    /// [`end`](Handle::end) or [`reset`](Handle::reset) must be called soon with the returned
    /// [`Work`] to avoid stalling the queue.
    ///
    /// # Safety
    /// `device` must match that passed to [`ParallelQueue::new`]
    pub unsafe fn begin(&mut self, device: &Device) -> Work {
        unsafe {
            let time = NonZeroU64::new_unchecked(
                self.shared
                    .first_unallocated
                    .fetch_add(1, Ordering::Relaxed),
            );
            let cmd = match self.spare_cmds.pop() {
                Some(cmd) => cmd,
                None => {
                    let complete = device
                        .get_semaphore_counter_value(self.shared.semaphore)
                        .unwrap();
                    while self
                        .in_flight
                        .front()
                        .map_or(false, |work| work.time.get() <= complete)
                    {
                        self.spare_cmds
                            .push(self.in_flight.pop_front().unwrap().cmd);
                    }
                    if self.spare_cmds.is_empty() {
                        self.spare_cmds.extend(
                            device
                                .allocate_command_buffers(
                                    &vk::CommandBufferAllocateInfo::default()
                                        .command_pool(self.cmd_pool)
                                        .command_buffer_count(32),
                                )
                                .unwrap(),
                        );
                    }
                    self.spare_cmds.pop().unwrap()
                }
            };
            device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            let work = Work { cmd, time };
            self.in_flight.push_back(work);
            work
        }
    }

    /// Send recorded commands out for execution
    ///
    /// # Safety
    /// - `device` must match that passed to [`ParallelQueue::new`]
    /// - `work` must have been obtained from a prior call to [`begin`](Handle::begin) on the same
    ///   `Handle`, and have not been previously passed to [`end`](Handle::begin) or
    ///   [`reset`](Handle::reset)
    pub unsafe fn end(&mut self, device: &Device, work: Work) {
        unsafe {
            device.end_command_buffer(work.cmd).unwrap();
            self.send.send(Message::Execute(work)).unwrap();
        }
    }

    /// Release a command buffer without executing any recorded commands
    ///
    /// # Safety
    /// - `device` must match that passed to [`ParallelQueue::new`]
    /// - `work` must have been obtained from a prior call to [`begin`](Handle::begin) on the same
    ///   `Handle`, and have not been previously passed to [`end`](Handle::begin) or
    ///   [`reset`](Handle::reset)
    pub unsafe fn reset(&mut self, device: &Device, work: Work) {
        unsafe {
            device
                .reset_command_buffer(work.cmd, vk::CommandBufferResetFlags::empty())
                .unwrap();
            self.send.send(Message::Reset(work.time)).unwrap();
        }
    }

    /// Create another handle to the same underlying [`ParallelQueue``]
    ///
    /// # Safety
    /// `device` must match that passed to `new`
    pub unsafe fn handle(&self, device: &Device) -> Handle {
        unsafe { self.shared.handle(device) }
    }
}

enum Message {
    Execute(Work),
    Reset(NonZeroU64),
}

impl Message {
    fn time(&self) -> NonZeroU64 {
        match *self {
            Message::Execute(ref work) => work.time,
            Message::Reset(time) => time,
        }
    }
}

impl Eq for Message {}

impl PartialEq for Message {
    fn eq(&self, other: &Self) -> bool {
        self.time() == other.time()
    }
}

impl Ord for Message {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.time().cmp(&other.time()).reverse()
    }
}

impl PartialOrd for Message {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
