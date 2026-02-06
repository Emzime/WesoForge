/// GPU API type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuApi {
    /// NVIDIA CUDA
    Cuda,
    /// OpenCL
    OpenCl,
}

/// Minimal device descriptor.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Backend API.
    pub api: GpuApi,
    /// Human readable name.
    pub name: String,
    /// Optional vendor.
    pub vendor: Option<String>,
    /// Device index (backend-specific).
    pub device_index: usize,
}

/// Detect available devices for all compiled-in backends.
///
/// If features are disabled, returns an empty list for that backend.
pub fn detect_devices() -> Vec<GpuDevice> {
    // When no GPU features are enabled, this vector is never mutated and Rust warns on `mut`.
    // We keep `mut` to support appending devices when features are enabled.
    #[allow(unused_mut)]
    let mut out = Vec::new();

    #[cfg(feature = "cuda")]
    {
        if let Ok(mut list) = crate::cuda::detect_cuda_devices() {
            out.append(&mut list);
        }
    }

    #[cfg(feature = "opencl")]
    {
        if let Ok(mut list) = crate::opencl::detect_opencl_devices() {
            out.append(&mut list);
        }
    }

    out
}
