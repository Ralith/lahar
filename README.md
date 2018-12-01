Lahar provides:
- A futures-compatible threadpool that supplies spawned tasks with access to a per-thread allocator
  of Vulkan-visible memory
- FIFO memory allocator that allows Vulkan-visible host memory to be freed by the GPU
- A binding of Vulkan fences to Rust futures

These allow a Vulkan application to easily load, decompress, and/or generate assets such as textures
and geometry asynchronously, without loading screens or unnecessary hitching.
