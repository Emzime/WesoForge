//! OpenCL compute backend.
//!
//! NOTE: The VDF kernel implementation is not yet integrated. These entrypoints exist
//! so that the compute orchestration can live outside of the CPU engine and evolve
//! independently.

use bbr_client_chiavdf_fast::api::ChiavdfBatchJob;

/// Execute a single proof on the given OpenCL device index.
pub fn prove_single(
    device_index: u32,
    _challenge: &[u8],
    _x: &[u8],
    _y_ref: &[u8],
    _discriminant_bits: usize,
    _num_iterations: u64,
) -> anyhow::Result<Vec<u8>> {
    anyhow::bail!(
        "BBR_GPU_NOT_IMPLEMENTED: opencl compute path not implemented (device={})",
        device_index
    );
}

/// Execute a single proof with progress callbacks on the given OpenCL device index.
pub fn prove_single_with_progress<F>(
    device_index: u32,
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
    anyhow::bail!(
        "BBR_GPU_NOT_IMPLEMENTED: opencl compute path not implemented (device={})",
        device_index
    );
}

/// Execute a batch proof on the given OpenCL device index.
pub fn prove_batch(
    device_index: u32,
    _challenge: &[u8],
    _x: &[u8],
    _discriminant_bits: usize,
    _jobs: &[ChiavdfBatchJob<'_>],
) -> anyhow::Result<Vec<Vec<u8>>> {
    anyhow::bail!(
        "BBR_GPU_NOT_IMPLEMENTED: opencl compute path not implemented (device={})",
        device_index
    );
}

/// Execute a batch proof with progress callbacks on the given OpenCL device index.
pub fn prove_batch_with_progress<F>(
    device_index: u32,
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
    anyhow::bail!(
        "BBR_GPU_NOT_IMPLEMENTED: opencl compute path not implemented (device={})",
        device_index
    );
}
