use std::marker::PhantomData;
use std::sync::Arc;

use ash::{self, version::DeviceV1_0, vk};

use crate::staging;

/// Helper for transferring staged memory.
pub struct Context {
    device: Arc<ash::Device>,
    pool: vk::CommandPool,
    queue: vk::Queue,
    queue_family_index: u32,
    dst_queue_family_index: u32,
    _not_sync: PhantomData<*mut ()>,
}

unsafe impl Send for Context {}

impl Context {
    /// Create a context for transferring memory using `queue` from `queue_family_index`, for future
    /// use on `dst_queue_family_index`.
    pub fn new(
        device: Arc<ash::Device>,
        queue: vk::Queue,
        queue_family_index: u32,
        dst_queue_family_index: u32,
    ) -> Self {
        let pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(queue_family_index),
                None,
            )
        }
        .unwrap();
        Self {
            device,
            pool,
            queue,
            queue_family_index,
            dst_queue_family_index,
            _not_sync: PhantomData,
        }
    }

    /// Copy `mem` to `dst` at `dst_offset`, freeing `mem` when finished.
    ///
    /// Inserts a "release" barrier if `queue_family_index != dst_queue_family_index`.
    pub unsafe async fn transfer_buffer(
        &self,
        mem: staging::Allocation,
        dst: vk::Buffer,
        dst_offset: vk::DeviceSize,
    ) {
        let cmd = self.alloc_cmd();
        let cmd = cmd.cmd;
        self.device.cmd_copy_buffer(
            cmd,
            mem.buffer(),
            dst,
            &[vk::BufferCopy {
                size: mem.len() as vk::DeviceSize,
                src_offset: mem.offset(),
                dst_offset,
            }],
        );
        if self.queue_family_index != self.dst_queue_family_index {
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[vk::BufferMemoryBarrier::builder()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::empty())
                    .src_queue_family_index(self.queue_family_index)
                    .dst_queue_family_index(self.dst_queue_family_index)
                    .buffer(dst)
                    .offset(dst_offset)
                    .size(mem.len() as vk::DeviceSize)
                    .build()],
                &[],
            );
        }
        self.run(cmd, mem).await
    }

    /// Copy `mem` to `dst`, freeing `mem` when finished.
    ///
    /// Inserts a "release" barrier if `queue_family_index != dst_queue_family_index`. Transitions
    /// `dst` to `SHADER_READ_ONLY_OPTIMAL` layout.
    pub unsafe async fn transfer_image(&self, mem: staging::Allocation, dst: ImageDst) {
        let cmd = self.alloc_cmd();
        let cmd = cmd.cmd;
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::builder()
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(dst.image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: dst.base_layer,
                    layer_count: dst.layers,
                })
                .build()],
        );
        self.device.cmd_copy_buffer_to_image(
            cmd,
            mem.buffer(),
            dst.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy::builder()
                .buffer_offset(mem.offset())
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: dst.mip_level,
                    base_array_layer: dst.base_layer,
                    layer_count: dst.layers,
                })
                .image_offset(dst.offset)
                .image_extent(dst.extent)
                .build()],
        );
        let mut barrier = vk::ImageMemoryBarrier::builder()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(dst.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: dst.mip_level,
                level_count: 1,
                base_array_layer: dst.base_layer,
                layer_count: dst.layers,
            });
        if self.queue_family_index != self.dst_queue_family_index {
            barrier = barrier
                .src_queue_family_index(self.queue_family_index)
                .dst_queue_family_index(self.dst_queue_family_index);
        }
        self.device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            Default::default(),
            &[],
            &[],
            &[barrier.build()],
        );
        self.run(cmd, mem).await;
    }

    unsafe fn alloc_cmd(&self) -> CmdBufferGuard {
        let cmd = self
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .unwrap()
            .into_iter()
            .next()
            .unwrap();
        self.device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();
        CmdBufferGuard { ctx: self, cmd }
    }

    unsafe async fn run(&self, cmd: vk::CommandBuffer, mem: staging::Allocation) {
        self.device.end_command_buffer(cmd).unwrap();
        self.device
            .queue_submit(
                self.queue,
                &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()],
                mem.fence().handle(),
            )
            .unwrap();
        mem.fence().submitted();
        mem.freed().clone().await;
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.queue_wait_idle(self.queue); // Ensure no cmd buffers in use
            self.device.destroy_command_pool(self.pool, None);
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub struct ImageDst {
    pub image: vk::Image,
    pub offset: vk::Offset3D,
    pub extent: vk::Extent3D,
    pub base_layer: u32,
    pub layers: u32,
    pub mip_level: u32,
}

impl Default for ImageDst {
    fn default() -> Self { Self {
        image: vk::Image::null(),
        offset: vk::Offset3D { x: 0, y: 0, z: 0 },
        extent: vk::Extent3D { width: 0, height: 0, depth: 0 },
        base_layer: 0,
        layers: 1,
        mip_level: 0,
    }}
}

struct CmdBufferGuard<'a> {
    ctx: &'a Context,
    cmd: vk::CommandBuffer,
}

impl<'a> Drop for CmdBufferGuard<'a> {
    fn drop(&mut self) {
        unsafe {
            self.ctx
                .device
                .free_command_buffers(self.ctx.pool, &[self.cmd]);
        }
    }
}
