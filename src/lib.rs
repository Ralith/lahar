//! Tools for asynchronous Vulkan

pub mod graveyard;
pub mod parallel_queue;
pub mod staging_ring;

mod memory;
mod region;
mod ring_state;
mod timeline_ring;
mod visit_handles;

pub use graveyard::{destroy_dynamic, Graveyard};
pub use memory::{
    align, alloc_bind, find_memory_type, AppendBuffer, DedicatedBuffer, DedicatedImage,
    DedicatedMapping, MemoryResource, ScratchBuffer, Staged,
};
pub use parallel_queue::ParallelQueue;
pub use region::{BufferRegion, BufferRegionAlloc, ImageRegion};
pub use staging_ring::StagingRing;
pub use timeline_ring::TimelineRing;
pub use visit_handles::{set_names, HandleVisitor, VisitHandles};

use ring_state::RingState;
