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
        self.cursor = self.cursor + 1 % self.frames.len();
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

    /// Free the resources in `resources` after `depth` frames
    pub fn inter(&mut self, resource: &impl DeferredCleanup) {
        resource.push_resources(&mut self.frames[self.cursor]);
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
    /// Record all owned resources in `frame`
    fn push_resources(&self, frame: &mut Frame);
}

impl DeferredCleanup for vk::Buffer {
    fn push_resources(&self, frame: &mut Frame) {
        frame.push_buffers(&[*self]);
    }
}

impl DeferredCleanup for vk::Image {
    fn push_resources(&self, frame: &mut Frame) {
        frame.push_images(&[*self]);
    }
}

impl DeferredCleanup for vk::ImageView {
    fn push_resources(&self, frame: &mut Frame) {
        frame.push_image_views(&[*self]);
    }
}

impl DeferredCleanup for vk::DeviceMemory {
    fn push_resources(&self, frame: &mut Frame) {
        frame.push_memories(&[*self]);
    }
}

/// A collection of resources to be freed in the future
pub struct Frame {
    buffers: Vec<vk::Buffer>,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    memories: Vec<vk::DeviceMemory>,
}

impl Frame {
    pub fn push_buffers(&mut self, buffers: &[vk::Buffer]) {
        self.buffers.extend_from_slice(buffers);
    }

    pub fn push_images(&mut self, images: &[vk::Image]) {
        self.images.extend_from_slice(images);
    }

    pub fn push_image_views(&mut self, image_views: &[vk::ImageView]) {
        self.image_views.extend_from_slice(image_views);
    }

    pub fn push_memories(&mut self, memories: &[vk::DeviceMemory]) {
        self.memories.extend_from_slice(memories);
    }
}
