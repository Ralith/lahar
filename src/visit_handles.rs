use std::ffi::CStr;

use ash::{extensions::ext::DebugUtils, vk, vk::Handle, Device};

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

pub unsafe fn set_names<T: VisitHandles>(device: &Device, pfn: &DebugUtils, x: &T, name: &CStr) {
    struct Visitor<'a>(&'a Device, &'a DebugUtils, &'a CStr);
    impl HandleVisitor for Visitor<'_> {
        fn visit<T: Handle>(&mut self, x: T) {
            unsafe {
                self.1
                    .set_debug_utils_object_name(
                        self.0.handle(),
                        &vk::DebugUtilsObjectNameInfoEXT::default()
                            .object_handle(x)
                            .object_name(self.2),
                    )
                    .unwrap();
            }
        }
    }
    x.visit_handles(&mut Visitor(device, pfn, name));
}
