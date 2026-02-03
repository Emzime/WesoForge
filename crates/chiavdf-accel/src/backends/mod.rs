//! Backend implementations.

mod cpu;

pub use cpu::CpuBackend;

#[cfg(feature = "cuda")]
mod cuda;
#[cfg(feature = "cuda")]
pub use cuda::CudaBackend;

#[cfg(feature = "opencl")]
mod opencl;
#[cfg(feature = "opencl")]
pub use opencl::OpenClBackend;
