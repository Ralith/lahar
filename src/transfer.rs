use std::convert::TryFrom;
use std::future::Future;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use ash::version::DeviceV1_0;
use ash::vk;
use futures_channel::oneshot;
use futures_util::FutureExt;

#[derive(Clone)]
pub struct TransferHandle {
    send: crossbeam_channel::Sender<Message>,
}

impl TransferHandle {
    pub unsafe fn upload_buffer(
        &self,
        src: vk::Buffer,
        dst: vk::Buffer,
        region: vk::BufferCopy,
    ) -> impl Future<Output = Result<(), ShutDown>> {
        let (sender, recv) = oneshot::channel();
        self.send
            .send(Message {
                sender,
                op: Op::UploadBuffer { src, dst, region },
            })
            .unwrap();
        recv.map(|x| x.map_err(|_| ShutDown))
    }

    /// Copy data from a buffer to an image
    ///
    /// `needs_transition` indicates whether `dst` should be transitioned to `TRANSFER_DST_OPTIMAL`
    pub unsafe fn upload_image(
        &self,
        src: vk::Buffer,
        dst: vk::Image,
        region: vk::BufferImageCopy,
        needs_transition: bool,
    ) -> impl Future<Output = Result<(), ShutDown>> {
        let (sender, recv) = oneshot::channel();
        self.send
            .send(Message {
                sender,
                op: Op::UploadImage {
                    src,
                    dst,
                    region,
                    needs_transition,
                },
            })
            .unwrap();
        recv.map(|x| x.map_err(|_| ShutDown))
    }

    // Extensible with practically any conceivable transfer command
}

pub fn acquire_buffer(
    src_queue_family: u32,
    dst_queue_family: u32,
    buffer: vk::Buffer,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
) -> vk::BufferMemoryBarrier {
    vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .src_queue_family_index(src_queue_family)
        .dst_queue_family_index(dst_queue_family)
        .buffer(buffer)
        .offset(offset)
        .size(size)
        .build()
}

pub fn acquire_image(
    src_queue_family: u32,
    dst_queue_family: u32,
    image: vk::Image,
) -> vk::ImageMemoryBarrier {
    vk::ImageMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .src_queue_family_index(src_queue_family)
        .dst_queue_family_index(dst_queue_family)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image(image)
        .build()
}

#[derive(Debug, Copy, Clone)]
pub struct ShutDown;

struct Message {
    sender: oneshot::Sender<()>,
    op: Op,
}

enum Op {
    UploadBuffer {
        src: vk::Buffer,
        dst: vk::Buffer,
        region: vk::BufferCopy,
    },
    UploadImage {
        src: vk::Buffer,
        dst: vk::Image,
        region: vk::BufferImageCopy,
        needs_transition: bool,
    },
}

pub struct Reactor {
    device: Arc<ash::Device>,
    queue_family: u32,
    queue: vk::Queue,
    /// == queue_family if `None` was supplied on construction
    dst_queue_family: u32, // switch to Option<u32> if we end up needing to branch anyway
    spare_fences: Vec<vk::Fence>,
    spare_cmds: Vec<vk::CommandBuffer>,
    in_flight: Vec<Batch>,
    /// Fences for in-flight transfer operations; directly corresponds to in_flight entries
    in_flight_fences: Vec<vk::Fence>,
    cmd_pool: vk::CommandPool,
    pending: Option<Batch>,
    buffer_barriers: Vec<vk::BufferMemoryBarrier>,
    image_barriers: Vec<vk::ImageMemoryBarrier>,
    recv: crossbeam_channel::Receiver<Message>,
}

impl Reactor {
    /// Safety: valid use use of queue_family, queue
    pub unsafe fn new(
        device: Arc<ash::Device>,
        queue_family: u32,
        queue: vk::Queue,
        dst_queue_family: Option<u32>,
    ) -> (TransferHandle, Self) {
        let (send, recv) = crossbeam_channel::unbounded();
        let cmd_pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
            .unwrap();
        (
            TransferHandle { send },
            Self {
                device,
                queue_family,
                queue,
                dst_queue_family: dst_queue_family.unwrap_or(queue_family),
                spare_fences: Vec::new(),
                spare_cmds: Vec::new(),
                in_flight: Vec::new(),
                in_flight_fences: Vec::new(),
                cmd_pool,
                pending: None,
                buffer_barriers: Vec::new(),
                image_barriers: Vec::new(),
                recv,
            },
        )
    }

    pub unsafe fn spawn(
        device: Arc<ash::Device>,
        queue_family: u32,
        queue: vk::Queue,
        dst_queue_family: Option<u32>,
    ) -> (TransferHandle, thread::JoinHandle<()>) {
        let (transfer, mut core) = Self::new(device, queue_family, queue, dst_queue_family);
        let thread = thread::spawn(
            move || {
                while core.run_for(Duration::from_millis(250)).is_ok() {}
            },
        );
        (transfer, thread)
    }

    pub fn poll(&mut self) -> Result<(), Disconnected> {
        self.run_for(Duration::from_secs(0))
    }

    pub fn run_for(&mut self, timeout: Duration) -> Result<(), Disconnected> {
        self.queue()?;
        self.flush();

        if self.in_flight.is_empty() {
            thread::sleep(timeout);
            return Ok(());
        }

        // We could move this to a background thread and continue to submit new work while it's
        // waiting, but we want to batch up operations a bit anyway.
        let result = unsafe {
            self.device.wait_for_fences(
                &self.in_flight_fences,
                false,
                u64::try_from(timeout.as_nanos()).unwrap_or(u64::max_value()),
            )
        };
        match result {
            Err(vk::Result::TIMEOUT) => return Ok(()),
            Err(e) => panic!("{}", e),
            Ok(()) => {}
        }
        for i in 0..self.in_flight.len() {
            unsafe {
                if self
                    .device
                    .get_fence_status(self.in_flight_fences[i])
                    .is_ok()
                {
                    let fence = self.in_flight_fences.swap_remove(i);
                    self.device.reset_fences(&[fence]).unwrap();
                    self.spare_fences.push(fence);
                    let batch = self.in_flight.swap_remove(i);
                    for sender in batch.senders {
                        let _ = sender.send(());
                    }
                    self.spare_cmds.push(batch.cmd);
                }
            }
        }
        Ok(())
    }

    fn queue(&mut self) -> Result<(), Disconnected> {
        loop {
            use crossbeam_channel::TryRecvError::*;
            match self.recv.try_recv() {
                Ok(Message { sender, op }) => {
                    let cmd = self.prepare(sender);
                    self.queue_op(cmd, op);
                }
                Err(Disconnected) => return Err(self::Disconnected),
                Err(Empty) => return Ok(()),
            }
        }
    }

    fn queue_op(&mut self, cmd: vk::CommandBuffer, op: Op) {
        use Op::*;
        match op {
            UploadBuffer { src, dst, region } => unsafe {
                self.device.cmd_copy_buffer(cmd, src, dst, &[region]);
                self.buffer_barriers.push(
                    vk::BufferMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .src_queue_family_index(self.queue_family)
                        .dst_queue_family_index(self.dst_queue_family)
                        .buffer(dst)
                        .offset(region.dst_offset)
                        .size(region.size)
                        .build(),
                );
            },
            UploadImage {
                src,
                dst,
                region,
                needs_transition,
            } => unsafe {
                if needs_transition {
                    self.device.cmd_pipeline_barrier(
                        cmd,
                        vk::PipelineStageFlags::TOP_OF_PIPE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::DependencyFlags::default(),
                        &[],
                        &[],
                        &[vk::ImageMemoryBarrier::builder()
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                            .old_layout(vk::ImageLayout::UNDEFINED)
                            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                            .image(dst)
                            .subresource_range(vk::ImageSubresourceRange {
                                aspect_mask: region.image_subresource.aspect_mask,
                                base_mip_level: region.image_subresource.mip_level,
                                level_count: 1,
                                base_array_layer: region.image_subresource.base_array_layer,
                                layer_count: region.image_subresource.layer_count,
                            })
                            .build()],
                    );
                }
                self.device.cmd_copy_buffer_to_image(
                    cmd,
                    src,
                    dst,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
                self.image_barriers.push(
                    vk::ImageMemoryBarrier::builder()
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .src_queue_family_index(self.queue_family)
                        .dst_queue_family_index(self.dst_queue_family)
                        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .image(dst)
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: region.image_subresource.aspect_mask,
                            base_mip_level: region.image_subresource.mip_level,
                            level_count: 1,
                            base_array_layer: region.image_subresource.base_array_layer,
                            layer_count: region.image_subresource.layer_count,
                        })
                        .build(),
                );
            },
        }
    }

    fn prepare(&mut self, send: oneshot::Sender<()>) -> vk::CommandBuffer {
        if let Some(ref mut pending) = self.pending {
            pending.senders.push(send);
            return pending.cmd;
        }
        let cmd = if let Some(cmd) = self.spare_cmds.pop() {
            cmd
        } else {
            unsafe {
                self.device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::builder()
                            .command_pool(self.cmd_pool)
                            .command_buffer_count(1),
                    )
                    .unwrap()
                    .into_iter()
                    .next()
                    .unwrap()
            }
        };
        unsafe {
            self.device
                .begin_command_buffer(
                    cmd,
                    &vk::CommandBufferBeginInfo::builder()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
        }
        self.pending = Some(Batch {
            cmd,
            senders: vec![send],
        });
        cmd
    }

    /// Submit queued operations
    fn flush(&mut self) {
        let pending = match self.pending.take() {
            Some(x) => x,
            None => return,
        };
        let fence = if let Some(fence) = self.spare_fences.pop() {
            fence
        } else {
            unsafe {
                self.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .unwrap()
            }
        };
        unsafe {
            self.device.cmd_pipeline_barrier(
                pending.cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::default(),
                &[],
                &self.buffer_barriers,
                &self.image_barriers,
            );
            self.device.end_command_buffer(pending.cmd).unwrap();
            self.device
                .queue_submit(
                    self.queue,
                    &[vk::SubmitInfo::builder()
                        .command_buffers(&[pending.cmd])
                        .build()],
                    fence,
                )
                .unwrap();
        }
        self.buffer_barriers.clear();
        self.image_barriers.clear();
        self.in_flight.push(pending);
        self.in_flight_fences.push(fence);
    }
}

impl Drop for Reactor {
    fn drop(&mut self) {
        unsafe {
            if !self.in_flight.is_empty() {
                self.device
                    .wait_for_fences(&self.in_flight_fences, true, u64::max_value())
                    .unwrap();
            }
            self.device.destroy_command_pool(self.cmd_pool, None);
            for fence in self.spare_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence, None);
            }
        }
    }
}

unsafe impl Send for Reactor {}

struct Batch {
    cmd: vk::CommandBuffer,
    // Future work: efficient broadcast future
    senders: Vec<oneshot::Sender<()>>,
}

#[derive(Debug, Copy, Clone)]
pub struct Disconnected;
