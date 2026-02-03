// Comments in English as requested.

use crate::gpu::{
    allowlist_matches, auto_batch_size_for_device, estimate_bytes_per_job, GpuBatchConfig,
    GpuDeviceInfo, GpuPlan, GpuPlannedDevice, GpuSelectConfig, GpuBackendKind,
};

/// Enumerate devices available for CUDA and/or OpenCL.
///
/// Current implementation:
/// - returns "stub" devices when GPU is enabled, to validate orchestration end-to-end.
/// - once cuda_backend/opencl_backend are implemented, this should call into them.
///
/// This keeps the engine code stable while GPU compute is being implemented.
pub fn enumerate_devices(select: &GpuSelectConfig) -> Vec<GpuDeviceInfo> {
    if !select.enabled {
        return Vec::new();
    }

    let mut devices: Vec<GpuDeviceInfo> = Vec::new();

    // TODO: Replace stubs with real detection:
    // - CUDA: query device list (NVIDIA)
    // - OpenCL: query platforms/devices (AMD/NVIDIA)
    //
    // For now we expose predictable "virtual" devices so multi-GPU orchestration,
    // max_devices, allowlist, and batch sizing can be tested.

    // If allow_cuda is true, provide a CUDA stub.
    if select.allow_cuda {
        devices.push(GpuDeviceInfo {
            backend: GpuBackendKind::Cuda,
            index: 0,
            name: "CUDA-Stub".to_string(),
            vendor: "NVIDIA".to_string(),
            vram_total_bytes: 0,
            vram_free_bytes: 0,
        });
    }

    // If allow_opencl is true, provide an OpenCL stub.
    if select.allow_opencl {
        devices.push(GpuDeviceInfo {
            backend: GpuBackendKind::Opencl,
            index: 0,
            name: "OpenCL-Stub".to_string(),
            vendor: "AMD".to_string(),
            vram_total_bytes: 0,
            vram_free_bytes: 0,
        });
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
        devices.truncate(max.max(0));
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
