// Comments in English as requested.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

//! CUDA backend glue for the client engine.
//!
//! This module is intentionally small and safe: all `unsafe` CUDA interaction is contained
//! in the `bbr-cuda-runtime` crate.

use anyhow::Context as _;

use crate::gpu::{GpuBackendKind, GpuDeviceInfo};

/// Enumerate CUDA devices using the CUDA runtime.
///
/// This is best-effort: if CUDA is not available, it returns an empty list.
pub(crate) fn enumerate_cuda_devices() -> Vec<GpuDeviceInfo> {
    let count = match bbr_cuda_runtime::cuda_device_count() {
        Ok(n) => n,
        Err(_) => return Vec::new(),
    };

    let mut out: Vec<GpuDeviceInfo> = Vec::with_capacity(count);
    for index in 0..count {
        let name = bbr_cuda_runtime::cuda_device_name(index).unwrap_or_else(|_| "Unknown CUDA GPU".to_string());
        out.push(GpuDeviceInfo {
            backend: GpuBackendKind::Cuda,
            index,
            name,
            vendor: "NVIDIA".to_string(),
            total_mem_bytes: None,
            free_mem_bytes: None,
            vram_total_bytes: 0,
            vram_free_bytes: 0,
        });
    }
    out
}

/// Run the CUDA smoketest on the specified device.
///
/// This is a lightweight health-check (H2D -> kernel -> D2H) and should not be run too often.
pub(crate) fn run_cuda_smoketest(device_index: usize, n: usize) -> anyhow::Result<()> {
    bbr_cuda_runtime::add1_smoketest(device_index, n)
        .with_context(|| format!("CUDA smoketest failed on device {device_index}"))
}

/// Execute the trivial add1 kernel for an entire packed batch.
///
/// This provides a real "pack -> H2D -> kernel -> D2H -> unpack" path, used as scaffolding
/// until the real VDF CUDA kernels are implemented.
pub(crate) fn add1_batch(device_index: usize, input: &[u32]) -> anyhow::Result<Vec<u32>> {
    bbr_cuda_runtime::add1_execute(device_index, input)
        .with_context(|| format!("CUDA add1_batch failed on device {device_index}"))
}
