#![deny(missing_docs)]
#![deny(unreachable_pub)]

//! GPU support crate (CUDA / OpenCL) for WesoForge.
//!
//! This crate is intentionally self-contained and does not modify existing crates.
//! Integration is done by calling this crate from `client-engine`.

mod detect;
mod error;

#[cfg(feature = "cuda")]
mod cuda;

#[cfg(feature = "opencl")]
mod opencl;

pub use detect::{detect_devices, GpuApi, GpuDevice};
pub use error::{ClientGpuError, GpuPreference};

/// Auto-select GPU backend (CUDA/OpenCL) and run the chiavdf prover.
///
/// - If GPU is not available or not enabled at compile-time, it falls back to CPU.
/// - Current GPU compute path is a scaffold: it initializes the backend and then falls back to CPU for correctness.
///
/// Returns `y || proof` like `bbr_client_chiavdf_fast::prove_one_weso_fast_streaming`.
pub fn prove_one_weso_fast_streaming_auto(
    preference: GpuPreference,
    challenge_hash: &[u8],
    x_s: &[u8],
    y_ref_s: &[u8],
    discriminant_size_bits: usize,
    num_iterations: u64,
) -> Result<Vec<u8>, ClientGpuError> {
    // Decide runtime preference
    let pref = preference.resolve_from_env();

    // Try CUDA first if requested/auto and compiled-in.
    if pref.allows(GpuApi::Cuda) {
        #[cfg(feature = "cuda")]
        {
            if let Ok(true) = cuda::is_available() {
                // Scaffold execution: currently falls back to CPU after init,
                // but the CUDA context init is real.
                let _ctx = cuda::CudaContext::new().map_err(ClientGpuError::CudaInit)?;
                // TODO: implement real GPU prover kernel pipeline here.
            }
        }
    }

    // Try OpenCL if requested/auto and compiled-in.
    if pref.allows(GpuApi::OpenCl) {
        #[cfg(feature = "opencl")]
        {
            if let Ok(true) = opencl::is_available() {
                // Scaffold execution: currently falls back to CPU after init,
                // but platform/device discovery is real.
                let _ctx = opencl::OpenClContext::new().map_err(ClientGpuError::OpenClInit)?;
                // TODO: implement real GPU prover kernel pipeline here.
            }
        }
    }

    // CPU fallback (correctness path)
    bbr_client_chiavdf_fast::prove_one_weso_fast_streaming(
        challenge_hash,
        x_s,
        y_ref_s,
        discriminant_size_bits,
        num_iterations,
    )
    .map_err(ClientGpuError::CpuFallback)
}

/// Auto-select GPU backend (CUDA/OpenCL) and run the chiavdf prover with progress callback.
///
/// This keeps the exact same progress semantics as the CPU implementation:
/// - The callback receives `iters_done` in `[0..=num_iterations]`.
/// - The callback is invoked roughly every `progress_interval` iterations.
///
/// Current GPU compute path is a scaffold: it initializes the backend and then falls back to CPU for correctness.
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
    // If progress is disabled by caller, reuse the non-progress path.
    if progress_interval == 0 {
        return prove_one_weso_fast_streaming_auto(
            preference,
            challenge_hash,
            x_s,
            y_ref_s,
            discriminant_size_bits,
            num_iterations,
        );
    }

    // Decide runtime preference
    let pref = preference.resolve_from_env();

    // Try CUDA first if requested/auto and compiled-in.
    if pref.allows(GpuApi::Cuda) {
        #[cfg(feature = "cuda")]
        {
            if let Ok(true) = cuda::is_available() {
                let _ctx = cuda::CudaContext::new().map_err(ClientGpuError::CudaInit)?;
                // TODO: implement real GPU prover kernel pipeline here.
                // For now, fall back to CPU for correctness while keeping progress exact.
            }
        }
    }

    // Try OpenCL if requested/auto and compiled-in.
    if pref.allows(GpuApi::OpenCl) {
        #[cfg(feature = "opencl")]
        {
            if let Ok(true) = opencl::is_available() {
                let _ctx = opencl::OpenClContext::new().map_err(ClientGpuError::OpenClInit)?;
                // TODO: implement real GPU prover kernel pipeline here.
                // For now, fall back to CPU for correctness while keeping progress exact.
            }
        }
    }

    // CPU fallback with progress (correctness + exact progress)
    bbr_client_chiavdf_fast::prove_one_weso_fast_streaming_with_progress(
        challenge_hash,
        x_s,
        y_ref_s,
        discriminant_size_bits,
        num_iterations,
        progress_interval,
        move |iters_done| {
            on_progress(iters_done);
        },
    )
    .map_err(ClientGpuError::CpuFallback)
}
