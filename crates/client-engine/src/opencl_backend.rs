// Comments in English as requested.

//! OpenCL backend integration.
//!
//! The GPU plumbing can plan devices from multiple backends (CUDA/OpenCL).
//! Only CUDA execution is currently implemented. To keep the project buildable
//! on machines without an OpenCL SDK, this module is a stub.

use crate::gpu::GpuDeviceInfo;

/// Enumerate OpenCL devices.
///
/// Currently unimplemented: returns an empty list.
pub(crate) fn enumerate_opencl_devices() -> Vec<GpuDeviceInfo> {
    Vec::new()
}
