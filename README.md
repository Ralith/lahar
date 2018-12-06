Lahar provides:
- FIFO memory allocator that allows Vulkan-visible host memory to be freed by the GPU with a fence
- A binding of Vulkan fences to Rust futures
- Helpers for performing asynchronous transfers

These allow a Vulkan application to easily load, decompress, and/or generate assets such as textures
and geometry asynchronously, without loading screens or unnecessary hitching.
