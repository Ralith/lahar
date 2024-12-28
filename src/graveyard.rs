use ash::{vk, vk::Handle, Device};

use crate::{HandleVisitor, VisitHandles};

/// Helper for deferred destruction of resources used within a frame
pub struct Graveyard {
    frames: Box<[Frame]>,
    cursor: usize,
}

impl Graveyard {
    /// Construct a graveyard that destroys resources `depth` frames after they're passed to `inter`
    pub fn new(depth: usize) -> Self {
        Self {
            frames: (0..depth)
                .map(|_| Frame {
                    handles: Vec::new(),
                })
                .collect(),
            cursor: 0,
        }
    }

    /// Number of frames after which resources passed to `inter` are destroyed
    #[inline]
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Free resources from `depth` frames ago
    pub unsafe fn begin_frame(&mut self, device: &Device) {
        self.cursor = (self.cursor + 1) % self.frames.len();
        let frame = &mut self.frames[self.cursor];
        for (ty, handle) in frame.handles.drain(..) {
            destroy_dynamic(device, ty, handle);
        }
    }

    /// Free the resources in `resources` after `self.depth()` frames
    pub fn inter(&mut self, resources: impl VisitHandles) {
        resources.visit_handles(self);
    }

    /// Free `handle` after `self.depth()` frames
    ///
    /// Escape hatch for stuff that doesn't implement `VisitHandles`
    pub fn inter_handle<T: Handle>(&mut self, handle: T) {
        self.frames[self.cursor].push(handle)
    }

    /// Free `handle` after `self.depth()` frames
    ///
    /// Escape hatch for dynamically typed handles
    pub fn inter_handle_dynamic(&mut self, ty: vk::ObjectType, handle: u64) {
        self.frames[self.cursor].handles.push((ty, handle))
    }

    /// Free all resources immediately
    pub unsafe fn clear(&mut self, device: &Device) {
        for _ in 0..self.frames.len() {
            self.begin_frame(device);
        }
    }
}

impl HandleVisitor for Graveyard {
    fn visit_dynamic(&mut self, ty: vk::ObjectType, handle: u64) {
        self.inter_handle_dynamic(ty, handle);
    }
}

/// A collection of resources to be freed in the future
struct Frame {
    handles: Vec<(vk::ObjectType, u64)>,
}

impl Frame {
    fn push<T: Handle>(&mut self, handle: T) {
        self.handles.push((T::TYPE, handle.as_raw()));
    }
}

pub unsafe fn destroy_dynamic(device: &Device, ty: vk::ObjectType, handle: u64) {
    match ty {
        vk::ObjectType::BUFFER => device.destroy_buffer(vk::Buffer::from_raw(handle), None),
        vk::ObjectType::IMAGE => device.destroy_image(vk::Image::from_raw(handle), None),
        vk::ObjectType::IMAGE_VIEW => {
            device.destroy_image_view(vk::ImageView::from_raw(handle), None)
        }
        vk::ObjectType::DEVICE_MEMORY => {
            device.free_memory(vk::DeviceMemory::from_raw(handle), None)
        }
        vk::ObjectType::FRAMEBUFFER => {
            device.destroy_framebuffer(vk::Framebuffer::from_raw(handle), None)
        }
        _ => unimplemented!("cannot destroy {:?} handles", ty),
    }
}
