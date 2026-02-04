// Comments in English as requested.

use crate::gpu::{
    allowlist_matches, auto_batch_size_for_device, estimate_bytes_per_job, GpuBatchConfig,
    GpuDeviceInfo, GpuPlan, GpuPlannedDevice, GpuSelectConfig,
};

/// Enumerate devices available for CUDA and/or OpenCL.
///
/// Current implementation:
/// - CUDA uses `nvidia-smi` when available (works on Windows/Linux with NVIDIA drivers).
/// - OpenCL is still a stub (will be implemented later).
///
/// This keeps engine orchestration stable while the actual compute backends evolve.
pub fn enumerate_devices(select: &GpuSelectConfig) -> Vec<GpuDeviceInfo> {
    if !select.enabled {
        return Vec::new();
    }

    let mut devices: Vec<GpuDeviceInfo> = Vec::new();

    if select.allow_cuda {
        devices.extend(crate::cuda_backend::enumerate_cuda_devices());
    }

    if select.allow_opencl {
        devices.extend(crate::opencl_backend::enumerate_opencl_devices());
    }

    // Apply allowlist if present.
    if !select.allowlist.is_empty() {
        devices = devices
            .into_iter()
            .filter(|d| select.allowlist.iter().any(|a| allowlist_matches(d, a)))
            .collect();
    }

    // Apply max_devices cap.
    if let Some(max) = select.max_devices {
        devices.truncate(max);
    }

    devices
}

/// Build a GPU execution plan:
/// - select devices according to config
/// - compute per-device auto batch sizes
/// - compute min_batch_total used by engine reservations
pub fn build_plan(select: &GpuSelectConfig, batch_cfg: &GpuBatchConfig) -> GpuPlan {
    let devices = enumerate_devices(select);
    if devices.is_empty() {
        return GpuPlan {
            devices: Vec::new(),
            min_batch_total: 0,
        };
    }

    let bytes_per_job = estimate_bytes_per_job();

    let mut planned: Vec<GpuPlannedDevice> = Vec::with_capacity(devices.len());
    let mut min_total: usize = 0;

    for dev in devices {
        let (batch_size, min_batch) = auto_batch_size_for_device(&dev, batch_cfg, bytes_per_job);
        min_total = min_total.saturating_add(min_batch);
        planned.push(GpuPlannedDevice {
            info: dev,
            batch_size,
            min_batch,
        });
    }

    GpuPlan {
        devices: planned,
        min_batch_total: min_total,
    }
}
