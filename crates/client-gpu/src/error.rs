use thiserror::Error;

/// Runtime preference for GPU usage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPreference {
    /// Prefer CUDA, fallback to OpenCL, then CPU.
    Auto,
    /// Force CUDA (fallback to CPU if not available).
    Cuda,
    /// Force OpenCL (fallback to CPU if not available).
    OpenCl,
    /// Disable GPU usage (CPU only).
    Off,
}

impl GpuPreference {
    /// Reads `WESOFORGE_GPU`:
    /// - "auto" (default)
    /// - "cuda"
    /// - "opencl"
    /// - "off"
    pub fn resolve_from_env(self) -> Self {
        if self != Self::Auto {
            return self;
        }

        let Ok(v) = std::env::var("WESOFORGE_GPU") else {
            return Self::Auto;
        };

        match v.trim().to_ascii_lowercase().as_str() {
            "cuda" => Self::Cuda,
            "opencl" => Self::OpenCl,
            "off" | "0" | "false" | "no" => Self::Off,
            "auto" | "1" | "true" | "yes" => Self::Auto,
            _unknown => Self::Auto,
        }
    }

    /// Whether this preference allows attempting a given API.
    pub fn allows(self, api: crate::GpuApi) -> bool {
        match self {
            Self::Auto => true,
            Self::Cuda => api == crate::GpuApi::Cuda,
            Self::OpenCl => api == crate::GpuApi::OpenCl,
            Self::Off => false,
        }
    }
}

/// Errors coming from GPU detection/initialization or CPU fallback.
#[derive(Debug, Error)]
pub enum ClientGpuError {
    /// CPU prover failed.
    #[error("cpu fallback failed: {0}")]
    CpuFallback(#[from] bbr_client_chiavdf_fast::ChiavdfFastError),

    /// CUDA context initialization failure.
    #[error("cuda init failed: {0}")]
    CudaInit(String),

    /// OpenCL context initialization failure.
    #[error("opencl init failed: {0}")]
    OpenClInit(String),
}
