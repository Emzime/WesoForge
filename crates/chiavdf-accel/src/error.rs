use thiserror::Error;

/// Result type for the accel layer.
pub type AccelResult<T> = Result<T, AccelError>;

/// Errors produced by CPU/GPU backends.
#[derive(Debug, Error)]
pub enum AccelError {
    /// Invalid input parameters.
    #[error("invalid input: {0}")]
    InvalidInput(&'static str),

    /// CPU backend failure.
    #[error("cpu backend failed: {0}")]
    CpuFailure(String),

    /// CUDA backend is not available or failed.
    #[cfg(feature = "cuda")]
    #[error("cuda backend failed: {0}")]
    CudaFailure(String),

    /// OpenCL backend is not available or failed.
    #[cfg(feature = "opencl")]
    #[error("opencl backend failed: {0}")]
    OpenClFailure(String),

    /// Backend not compiled in / not supported on this platform.
    #[error("backend not available: {0}")]
    NotAvailable(&'static str),
}
