//! GPU execution entrypoints.
//!
//! This module defines the GPU-side execution API. The CPU orchestration layer
//! calls into this module to execute GPU workloads. At this stage, the actual
//! CUDA/OpenCL compute backends are still under development, so these functions
//! mainly forward to backend-specific stubs.

use crate::GpuBackend;
use bbr_client_chiavdf_fast::api::ChiavdfBatchJob;

/// Execute a single proof on a GPU device.
pub fn prove_single(
    backend: GpuBackend,
    _device_ordinal: u32,
    _challenge: &[u8],
    _x: &[u8],
    _y_ref: &[u8],
    _discriminant_bits: usize,
    _num_iterations: u64,
) -> anyhow::Result<Vec<u8>> {
    match backend {
        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::compute::prove_single(
                    _device_ordinal,
                    _challenge,
                    _x,
                    _y_ref,
                    _discriminant_bits,
                    _num_iterations,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA backend not enabled");
            }
        }
        GpuBackend::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                crate::opencl::compute::prove_single(
                    _device_ordinal,
                    _challenge,
                    _x,
                    _y_ref,
                    _discriminant_bits,
                    _num_iterations,
                )
            }
            #[cfg(not(feature = "opencl"))]
            {
                anyhow::bail!("OpenCL backend not enabled");
            }
        }
    }
}

/// Execute a single proof with progress reporting.
pub fn prove_single_with_progress<F>(
    backend: GpuBackend,
    _device_ordinal: u32,
    _challenge: &[u8],
    _x: &[u8],
    _y_ref: &[u8],
    _discriminant_bits: usize,
    _num_iterations: u64,
    _progress_interval: u64,
    _progress: F,
) -> anyhow::Result<Vec<u8>>
where
    F: FnMut(u64) + Send + 'static,
{
    match backend {
        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::compute::prove_single_with_progress(
                    _device_ordinal,
                    _challenge,
                    _x,
                    _y_ref,
                    _discriminant_bits,
                    _num_iterations,
                    _progress_interval,
                    _progress,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA backend not enabled");
            }
        }
        GpuBackend::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                crate::opencl::compute::prove_single_with_progress(
                    _device_ordinal,
                    _challenge,
                    _x,
                    _y_ref,
                    _discriminant_bits,
                    _num_iterations,
                    _progress_interval,
                    _progress,
                )
            }
            #[cfg(not(feature = "opencl"))]
            {
                anyhow::bail!("OpenCL backend not enabled");
            }
        }
    }
}

/// Execute a batch proof on a GPU device.
pub fn prove_batch(
    backend: GpuBackend,
    _device_ordinal: u32,
    _challenge: &[u8],
    _x: &[u8],
    _discriminant_bits: usize,
    _jobs: &[ChiavdfBatchJob<'_>],
) -> anyhow::Result<Vec<Vec<u8>>> {
    match backend {
        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::compute::prove_batch(
                    _device_ordinal,
                    _challenge,
                    _x,
                    _discriminant_bits,
                    _jobs,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA backend not enabled");
            }
        }
        GpuBackend::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                crate::opencl::compute::prove_batch(
                    _device_ordinal,
                    _challenge,
                    _x,
                    _discriminant_bits,
                    _jobs,
                )
            }
            #[cfg(not(feature = "opencl"))]
            {
                anyhow::bail!("OpenCL backend not enabled");
            }
        }
    }
}

/// Execute a batch proof with progress reporting.
pub fn prove_batch_with_progress<F>(
    backend: GpuBackend,
    _device_ordinal: u32,
    _challenge: &[u8],
    _x: &[u8],
    _discriminant_bits: usize,
    _jobs: &[ChiavdfBatchJob<'_>],
    _progress_interval: u64,
    _progress: F,
) -> anyhow::Result<Vec<Vec<u8>>>
where
    F: FnMut(u64) + Send + 'static,
{
    match backend {
        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::compute::prove_batch_with_progress(
                    _device_ordinal,
                    _challenge,
                    _x,
                    _discriminant_bits,
                    _jobs,
                    _progress_interval,
                    _progress,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("CUDA backend not enabled");
            }
        }
        GpuBackend::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                crate::opencl::compute::prove_batch_with_progress(
                    _device_ordinal,
                    _challenge,
                    _x,
                    _discriminant_bits,
                    _jobs,
                    _progress_interval,
                    _progress,
                )
            }
            #[cfg(not(feature = "opencl"))]
            {
                anyhow::bail!("OpenCL backend not enabled");
            }
        }
    }
}
