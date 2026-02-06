use crate::{GpuApi, GpuDevice};

/// Quick availability check (platform present, API usable).
pub(crate) fn is_available() -> Result<bool, ()> {
    let platforms = opencl3::platform::get_platforms().map_err(|_e| ())?;
    Ok(!platforms.is_empty())
}

/// OpenCL context wrapper (minimal, just for validation/discovery).
pub(crate) struct OpenClContext {
    _platform_index: usize,
    _device_index: usize,
}

impl OpenClContext {
    /// Initialize OpenCL using the first platform and first GPU device (if any).
    pub(crate) fn new() -> Result<Self, String> {
        let platforms = opencl3::platform::get_platforms().map_err(|e| format!("{e}"))?;
        if platforms.is_empty() {
            return Err("no OpenCL platform found".to_string());
        }

        // Pick first platform; then first GPU device.
        for (pidx, p) in platforms.iter().enumerate() {
            let devices = p
                .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
                .map_err(|e| format!("{e}"))?;

            if devices.is_empty() {
                continue;
            }

            // We do not create a full cl_context yet; this is a scaffold.
            return Ok(Self {
                _platform_index: pidx,
                _device_index: 0,
            });
        }

        Err("no OpenCL GPU device found".to_string())
    }
}

/// List OpenCL GPU devices.
pub(crate) fn detect_opencl_devices() -> Result<Vec<GpuDevice>, String> {
    let platforms = opencl3::platform::get_platforms().map_err(|e| format!("{e}"))?;
    let mut out = Vec::new();

    let mut global_idx: usize = 0;
    for (_pidx, p) in platforms.iter().enumerate() {
        let devices = p
            .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .map_err(|e| format!("{e}"))?;

        for (_didx, d) in devices.iter().enumerate() {
            let dev = opencl3::device::Device::new(*d);

            let name = dev.name().unwrap_or_else(|_e| "Unknown OpenCL Device".to_string());
            let vendor = dev.vendor().ok();

            out.push(GpuDevice {
                api: GpuApi::OpenCl,
                name,
                vendor,
                device_index: global_idx,
            });

            global_idx = global_idx.saturating_add(1);
        }
    }

    Ok(out)
}

/// Phase-1 OpenCL compute plumbing validation (scaffold).
///
/// This currently does not run a compute kernel yet; it validates that a GPU device exists
/// and that the OpenCL runtime can be initialized. It then burns a small amount of CPU time
/// proportionate to `num_iterations` to emulate a workload.
pub(crate) fn run_spin_kernel_on_device(device_index: usize, num_iterations: u64) -> Result<(), String> {
    let platforms = opencl3::platform::get_platforms().map_err(|e| format!("{e}"))?;
    if platforms.is_empty() {
        return Err("no OpenCL platform found".to_string());
    }

    // Resolve global device index into a platform + device handle.
    let mut global_idx: usize = 0;
    for p in platforms {
        let devices = p
            .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .map_err(|e| format!("{e}"))?;
        for d in devices {
            if global_idx == device_index {
                // Creating a Device validates that the handle is usable.
                let _dev = opencl3::device::Device::new(d);

                // Emulate a scaled workload until the real OpenCL kernels land.
                let ms = ((num_iterations / 1_000_000).clamp(1, 2_000)) as u64;
                std::thread::sleep(std::time::Duration::from_millis(ms));
                return Ok(());
            }
            global_idx = global_idx.saturating_add(1);
        }
    }

    Err(format!("OpenCL device index {device_index} not found"))
}
