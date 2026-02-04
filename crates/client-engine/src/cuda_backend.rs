// Comments in English as requested.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

//! CUDA backend glue for the client engine.
//!
//! This module is intentionally small and safe: all `unsafe` CUDA interaction is contained
//! in the `bbr-cuda-runtime` crate.

use anyhow::Context as _;

/// Run the CUDA smoketest on the specified device.
///
/// This is a lightweight health-check (H2D -> kernel -> D2H) and should not be run too often.
pub(crate) fn run_cuda_smoketest(device_index: usize, n: usize) -> anyhow::Result<()> {
    bbr_cuda_runtime::add1_smoketest(device_index, n)
        .with_context(|| format!("CUDA smoketest failed on device {device_index}"))
}

/// Execute the trivial add1 kernel for an entire packed batch.
///
/// This provides a real "pack -> H2D -> kernel -> D2H -> unpack" path, used as a scaffolding
/// until the real VDF CUDA kernels are implemented.
pub(crate) fn add1_batch(device_index: usize, input: &[u32]) -> anyhow::Result<Vec<u32>> {
    bbr_cuda_runtime::add1_execute(device_index, input)
        .with_context(|| format!("CUDA add1_batch failed on device {device_index}"))
}
