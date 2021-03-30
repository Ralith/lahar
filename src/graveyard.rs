use ash::{version::DeviceV1_0, vk, Device};

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
                    buffers: Vec::new(),
                    images: Vec::new(),
                    image_views: Vec::new(),
                    memories: Vec::new(),
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
        for buffer in frame.buffers.drain(..) {
            device.destroy_buffer(buffer, None);
        }
        for image in frame.images.drain(..) {
            device.destroy_image(image, None);
        }
        for image_view in frame.image_views.drain(..) {
            device.destroy_image_view(image_view, None);
        }
        for memory in frame.memories.drain(..) {
            device.free_memory(memory, None);
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

impl DeferredCleanup for vk::Buffer {
    fn inter_into(self, graveyard: &mut Graveyard) {
        graveyard.frames[graveyard.cursor].buffers.push(self);
    }
}

impl DeferredCleanup for vk::Image {
    fn inter_into(self, graveyard: &mut Graveyard) {
        graveyard.frames[graveyard.cursor].images.push(self);
    }
}

impl DeferredCleanup for vk::ImageView {
    fn inter_into(self, graveyard: &mut Graveyard) {
        graveyard.frames[graveyard.cursor].image_views.push(self);
    }
}

impl DeferredCleanup for vk::DeviceMemory {
    fn inter_into(self, graveyard: &mut Graveyard) {
        graveyard.frames[graveyard.cursor].memories.push(self);
    }
}

/// A collection of resources to be freed in the future
struct Frame {
    buffers: Vec<vk::Buffer>,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    memories: Vec<vk::DeviceMemory>,
}
