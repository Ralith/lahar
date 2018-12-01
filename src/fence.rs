use std::future::Future;
use std::mem;
use std::pin::Pin;
use std::sync::{Arc, Condvar, Mutex};
use std::task::{LocalWaker, Poll, Waker};

use ash::{version::DeviceV1_0, vk};

/// Adapter from Vulkan fences to futures.
///
/// Must be `poll`ed manually for blocked futures to make progress. Can be cloned cheaply, producing
/// multiple references to the same factory.
///
/// ```no_run
/// #![feature(await_macro, async_await)]
/// # extern crate ash;
/// # extern crate lahar;
/// # extern crate futures;
/// # fn do_something_with(x: ash::vk::Fence) {}
/// # fn stub(device: std::sync::Arc<ash::Device>) {
/// use futures::task::LocalSpawnExt;
/// let executor = futures::executor::LocalPool::new();
/// let factory = lahar::FenceFactory::new(device.clone());
/// let fence = factory.get();
/// do_something_with(fence.handle());
/// unsafe { fence.submitted(); }
/// executor.spawner().spawn_local(async {
///     await!(fence);
///     println!("fence signaled");
/// });
/// loop { factory.poll(); std::thread::yield_now(); }
/// # }
/// ```
#[derive(Clone)]
pub struct FenceFactory {
    device: Arc<ash::Device>,
    inner: Arc<Mutex<FenceFactoryInner>>,
}

impl FenceFactory {
    pub fn new(device: Arc<ash::Device>) -> Self {
        Self {
            device,
            inner: Arc::new(Mutex::new(FenceFactoryInner {
                fences: Vec::new(),
                tasks: Vec::new(),
            })),
        }
    }

    /// Create an unsignalled fence
    pub fn get(&self) -> (Fence, FenceSignaled) {
        let handle = unsafe {
            self.device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .unwrap()
        };
        let inner = Arc::new(FenceInner {
            device: Arc::clone(&self.device),
            factory: Arc::clone(&self.inner),
            submitted: Mutex::new(SubmitState::None),
            submitted_cv: Condvar::new(),
            handle,
        });
        (
            Fence {
                inner: inner.clone(),
            },
            FenceSignaled {
                inner: inner.clone(),
            },
        )
    }

    /// Wake tasks blocked on signalled fences.
    ///
    /// This should be called on a regular basis (e.g. per frame).
    pub fn poll(&self) {
        let mut inner = self.inner.lock().unwrap();
        debug_assert_eq!(inner.fences.len(), inner.tasks.len());
        if unsafe { self.device.wait_for_fences(&inner.fences, false, 0) }.is_err() {
            return;
        }
        let mut i = 0;
        loop {
            if i > inner.fences.len() {
                break;
            }
            if unsafe { self.device.get_fence_status(inner.fences[i]).is_ok() } {
                let fence = inner.fences.swap_remove(i);
                unsafe {
                    self.device.destroy_fence(fence, None);
                }
                let task = inner.tasks.swap_remove(i);
                task.wake();
            } else {
                i += 1;
            }
        }
    }
}

struct FenceFactoryInner {
    fences: Vec<vk::Fence>,
    tasks: Vec<Waker>,
}

/// A Vulkan fence associated with a future.
///
/// For the associated future to complete:
/// - the corresponding [`FenceFactory`] must be `poll`ed on a regular basis
/// - `submitted` must be invoked after the fence has been submitted to the Vulkan implementation
///
/// # Safety
/// The behavior is undefined if a `Fence` is dropped after the Vulkan device its factory was
/// created from is destroyed.
pub struct Fence {
    inner: Arc<FenceInner>,
}

impl Fence {
    /// Get the Vulkan handle to register this fence to be signaled.
    pub fn handle(&self) -> vk::Fence {
        self.inner.handle
    }

    /// Declare that this fence has been submitted.
    ///
    /// This must be called after and only after the fence has been submitted to the Vulkan
    /// implementation, such that calling `vkWaitForFences` on the handle is defined behavior.
    pub unsafe fn submitted(&self) {
        let old = {
            let mut submitted = self.inner.submitted.lock().unwrap();
            let x = mem::replace(&mut *submitted, SubmitState::Done);
            self.inner.submitted_cv.notify_one();
            x
        };
        if let SubmitState::Blocking(waker) = old {
            let mut factory = self.inner.factory.lock().unwrap();
            factory.fences.push(self.inner.handle);
            factory.tasks.push(waker);
        }
    }

    /// Check the current status of the fence.
    pub fn signaled(&self) -> bool {
        if !self.inner.submitted.lock().unwrap().is_done() {
            return false;
        }
        unsafe {
            self.inner
                .device
                .get_fence_status(self.inner.handle)
                .is_ok()
        }
    }

    /// Block the thread until the fence is signaled.
    pub fn block(&self) {
        let mut submitted = self.inner.submitted.lock().unwrap();
        while !submitted.is_done() {
            submitted = self.inner.submitted_cv.wait(submitted).unwrap();
        }
        let _ = unsafe {
            self.inner
                .device
                .wait_for_fences(&[self.inner.handle], true, !0)
        };
    }
}

/// A future that completes when the associated Vulkan fence is signaled.
///
/// For the associated future to complete:
/// - the corresponding [`FenceFactory`] must be `poll`ed on a regular basis
/// - `submitted` must be invoked after the fence has been submitted to the Vulkan implementation
///
/// # Safety
/// The behavior is undefined if this future is dropped after the Vulkan device its factory was
/// created from is destroyed.
pub struct FenceSignaled {
    inner: Arc<FenceInner>,
}

impl Future for FenceSignaled {
    type Output = ();
    fn poll(self: Pin<&mut Self>, lw: &LocalWaker) -> Poll<Self::Output> {
        let mut submitted = self.inner.submitted.lock().unwrap();
        if !submitted.is_done() {
            *submitted = SubmitState::Blocking(lw.clone().into_waker());
            Poll::Pending
        } else {
            mem::drop(submitted);
            if unsafe {
                self.inner
                    .device
                    .get_fence_status(self.inner.handle)
                    .is_ok()
            } {
                Poll::Ready(())
            } else {
                let mut factory = self.inner.factory.lock().unwrap();
                factory.fences.push(self.inner.handle);
                factory.tasks.push(lw.clone().into_waker());
                Poll::Pending
            }
        }
    }
}

struct FenceInner {
    device: Arc<ash::Device>,
    factory: Arc<Mutex<FenceFactoryInner>>,
    submitted: Mutex<SubmitState>,
    submitted_cv: Condvar,
    handle: vk::Fence,
}

impl Drop for FenceInner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_fence(self.handle, None);
        }
    }
}

enum SubmitState {
    None,
    Blocking(Waker),
    Done,
}

impl SubmitState {
    fn is_done(&self) -> bool {
        match *self {
            SubmitState::Done => true,
            _ => false,
        }
    }
}
