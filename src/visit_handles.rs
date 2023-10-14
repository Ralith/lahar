use std::ffi::CStr;

use ash::{ext::debug_utils, vk, vk::Handle};

pub trait HandleVisitor {
    fn visit<T: Handle>(&mut self, x: T);
}

pub trait VisitHandles {
    fn visit_handles<V: HandleVisitor>(&self, visitor: &mut V);
}

impl<T: VisitHandles> VisitHandles for Option<T> {
    fn visit_handles<V: HandleVisitor>(&self, visitor: &mut V) {
        if let Some(x) = self {
            x.visit_handles(visitor);
        }
    }
}

macro_rules! impl_handles {
    ( $($ty:ident,)* ) => {
        $(
            impl VisitHandles for vk::$ty {
                fn visit_handles<V: HandleVisitor>(&self, visitor: &mut V) {
                    visitor.visit(*self);
                }
            }
        )*
    };
}

impl_handles!(Buffer, Image, ImageView, DeviceMemory, Framebuffer,);

pub unsafe fn set_names<T: VisitHandles>(pfn: &debug_utils::Device, x: &T, name: &CStr) {
    struct Visitor<'a>(&'a debug_utils::Device, &'a CStr);
    impl HandleVisitor for Visitor<'_> {
        fn visit<T: Handle>(&mut self, x: T) {
            unsafe {
                self.0
                    .set_debug_utils_object_name(
                        &vk::DebugUtilsObjectNameInfoEXT::default()
                            .object_handle(x)
                            .object_name(self.1),
                    )
                    .unwrap();
            }
        }
    }
    x.visit_handles(&mut Visitor(pfn, name));
}
