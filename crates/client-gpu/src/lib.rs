#![deny(missing_docs)]
#![deny(unreachable_pub)]

//! GPU support crate (CUDA / OpenCL) for WesoForge.
//!
//! Phase 1 (CUDA): real CUDA execution plumbing (kernel launch) is implemented as a validation step.
//! The VDF prover math still runs on CPU for correctness.

mod detect;
mod error;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "opencl")]
mod opencl;

pub use detect::{detect_devices, GpuApi, GpuDevice};
pub use error::{ClientGpuError, GpuPreference};

fn log_backend(msg: &str) {
    eprintln!("[client-gpu] {msg}");
}

/// Auto-select GPU backend (CUDA/OpenCL) and run the chiavdf prover.
pub fn prove_one_weso_fast_streaming_auto(
    preference: GpuPreference,
    challenge_hash: &[u8],
    x_s: &[u8],
    y_ref_s: &[u8],
    discriminant_size_bits: usize,
    num_iterations: u64,
) -> Result<Vec<u8>, ClientGpuError> {
    let pref = preference.resolve_from_env();

    if pref.allows(GpuApi::Cuda) {
        #[cfg(feature = "cuda")]
        {
            if let Ok(true) = cuda::is_available() {
                log_backend("CUDA available; running phase-1 kernel (plumbing validation)");
                let _ctx = cuda::CudaContext::new().map_err(ClientGpuError::CudaInit)?;
                cuda::run_spin_kernel(num_iterations).map_err(ClientGpuError::CudaInit)?;
            }
        }
    }

    if pref.allows(GpuApi::OpenCl) {
        #[cfg(feature = "opencl")]
        {
            if let Ok(true) = opencl::is_available() {
                log_backend("OpenCL available; (compute path not implemented yet), using CPU");
                let _ctx = opencl::OpenClContext::new().map_err(ClientGpuError::OpenClInit)?;
            }
        }
    }

    log_backend("CPU prover (source of truth) running");
    bbr_client_chiavdf_fast::prove_one_weso_fast_streaming(
        challenge_hash,
        x_s,
        y_ref_s,
        discriminant_size_bits,
        num_iterations,
    )
    .map_err(ClientGpuError::CpuFallback)
}

/// Auto-select GPU backend with progress callback.
pub fn prove_one_weso_fast_streaming_auto_with_progress<F>(
    preference: GpuPreference,
    challenge_hash: &[u8],
    x_s: &[u8],
    y_ref_s: &[u8],
    discriminant_size_bits: usize,
    num_iterations: u64,
    progress_interval: u64,
    mut on_progress: F,
) -> Result<Vec<u8>, ClientGpuError>
where
    F: FnMut(u64) + Send + 'static,
{
    let pref = preference.resolve_from_env();

    if pref.allows(GpuApi::Cuda) {
        #[cfg(feature = "cuda")]
        {
            if let Ok(true) = cuda::is_available() {
                log_backend("CUDA available; running phase-1 kernel (plumbing validation) [with progress]");
                let _ctx = cuda::CudaContext::new().map_err(ClientGpuError::CudaInit)?;
                cuda::run_spin_kernel(num_iterations).map_err(ClientGpuError::CudaInit)?;
            }
        }
    }

    if pref.allows(GpuApi::OpenCl) {
        #[cfg(feature = "opencl")]
        {
            if let Ok(true) = opencl::is_available() {
                log_backend("OpenCL available; (compute path not implemented yet) [with progress], using CPU");
                let _ctx = opencl::OpenClContext::new().map_err(ClientGpuError::OpenClInit)?;
            }
        }
    }

    log_backend("CPU prover (source of truth) running [with progress]");
    bbr_client_chiavdf_fast::prove_one_weso_fast_streaming_with_progress(
        challenge_hash,
        x_s,
        y_ref_s,
        discriminant_size_bits,
        num_iterations,
        progress_interval,
        move |iters_done| on_progress(iters_done),
    )
    .map_err(ClientGpuError::CpuFallback)
}
