use std::sync::Arc;

use crate::{AccelError, AccelResult, VdfBackend};

use crate::backends::{CpuBackend};

#[cfg(feature = "cuda")]
use crate::backends::CudaBackend;

#[cfg(feature = "opencl")]
use crate::backends::OpenClBackend;

/// Backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// Force CPU.
    Cpu,
    /// Prefer CUDA then fallback.
    Cuda,
    /// Prefer OpenCL then fallback.
    OpenCl,
    /// Auto: try CUDA, then OpenCL, then CPU.
    Auto,
}

/// Options for selecting/initializing a backend.
#[derive(Debug, Clone)]
pub struct BackendOptions {
    /// Selection strategy.
    pub kind: BackendKind,
    /// Optional CUDA device ordinal.
    pub cuda_device: Option<usize>,
    /// Optional OpenCL platform/device filters.
    pub opencl_platform_substr: Option<String>,
    pub opencl_device_substr: Option<String>,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            kind: BackendKind::Auto,
            cuda_device: None,
            opencl_platform_substr: None,
            opencl_device_substr: None,
        }
    }
}

/// A backend chosen at runtime.
pub struct SelectedBackend {
    backend: Arc<dyn VdfBackend>,
}

impl SelectedBackend {
    /// Get backend handle.
    pub fn backend(&self) -> Arc<dyn VdfBackend> {
        self.backend.clone()
    }
}

/// Pick the best available backend for this machine.
pub fn select_backend(opts: &BackendOptions) -> AccelResult<SelectedBackend> {
    match opts.kind {
        BackendKind::Cpu => Ok(SelectedBackend { backend: Arc::new(CpuBackend::new()?) }),

        BackendKind::Cuda => {
            #[cfg(feature = "cuda")]
            {
                if let Ok(b) = CudaBackend::new(opts.cuda_device) {
                    return Ok(SelectedBackend { backend: Arc::new(b) });
                }
            }
            Ok(SelectedBackend { backend: Arc::new(CpuBackend::new()?) })
        }

        BackendKind::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                if let Ok(b) = OpenClBackend::new(
                    opts.opencl_platform_substr.as_deref(),
                    opts.opencl_device_substr.as_deref(),
                ) {
                    return Ok(SelectedBackend { backend: Arc::new(b) });
                }
            }
            Ok(SelectedBackend { backend: Arc::new(CpuBackend::new()?) })
        }

        BackendKind::Auto => {
            #[cfg(feature = "cuda")]
            {
                if let Ok(b) = CudaBackend::new(opts.cuda_device) {
                    return Ok(SelectedBackend { backend: Arc::new(b) });
                }
            }
            #[cfg(feature = "opencl")]
            {
                if let Ok(b) = OpenClBackend::new(
                    opts.opencl_platform_substr.as_deref(),
                    opts.opencl_device_substr.as_deref(),
                ) {
                    return Ok(SelectedBackend { backend: Arc::new(b) });
                }
            }
            Ok(SelectedBackend { backend: Arc::new(CpuBackend::new()?) })
        }
    }
}
