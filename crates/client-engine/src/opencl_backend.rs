// Comments in English as requested.

//! OpenCL backend integration.
//!
//! The engine supports OpenCL device enumeration in its planning layer, but the current
//! implementation focuses on CUDA execution. To keep the project buildable on machines
//! without OpenCL SDKs, we provide a minimal stub enumerator here.

use crate::gpu::GpuDeviceInfo;

/// Enumerate OpenCL devices.
///
/// Currently unimplemented: returns an empty list.
pub(crate) fn enumerate_opencl_devices() -> Vec<GpuDeviceInfo> {
    Vec::new()
}
