use ash::{vk, vk::Handle, Device};

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
            destroy(device, ty, handle);
        }
    }

    /// Free the resources in `resources` after `self.depth()` frames
    pub fn inter(&mut self, resource: impl DeferredCleanup) {
        resource.inter_into(self);
    }

    /// Free all resources immediately
    pub unsafe fn clear(&mut self, device: &Device) {
        for _ in 0..self.frames.len() {
            self.begin_frame(device);
        }
    }
}

/// Owners of Vulkan resources
pub trait DeferredCleanup {
    fn inter_into(self, graveyard: &mut Graveyard);
}

macro_rules! impl_handles {
    ( $($ty:ident,)* ) => {
        $(
            impl DeferredCleanup for vk::$ty {
                fn inter_into(self, graveyard: &mut Graveyard) {
                    graveyard.frames[graveyard.cursor].push(self);
                }
            }
        )*
    };
}

impl_handles!(Buffer, Image, ImageView, DeviceMemory, Framebuffer,);

/// A collection of resources to be freed in the future
struct Frame {
    handles: Vec<(vk::ObjectType, u64)>,
}

impl Frame {
    fn push<T: Handle>(&mut self, handle: T) {
        self.handles.push((T::TYPE, handle.as_raw()));
    }
}

unsafe fn destroy(device: &Device, ty: vk::ObjectType, handle: u64) {
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
