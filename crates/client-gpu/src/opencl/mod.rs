use crate::{GpuApi, GpuDevice};

/// Quick availability check (platform present, API usable).
pub fn is_available() -> Result<bool, ()> {
    let platforms = opencl3::platform::get_platforms().map_err(|_e| ())?;
    Ok(!platforms.is_empty())
}

/// OpenCL context wrapper (minimal, just for validation/discovery).
pub struct OpenClContext {
    _platform_index: usize,
    _device_index: usize,
}

impl OpenClContext {
    /// Initialize OpenCL using the first platform and first GPU device (if any).
    pub fn new() -> Result<Self, String> {
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
pub fn detect_opencl_devices() -> Result<Vec<GpuDevice>, String> {
    let platforms = opencl3::platform::get_platforms().map_err(|e| format!("{e}"))?;
    let mut out = Vec::new();

    for (pidx, p) in platforms.iter().enumerate() {
        let devices = p
            .get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)
            .map_err(|e| format!("{e}"))?;

        for (didx, d) in devices.iter().enumerate() {
            let dev = opencl3::device::Device::new(*d);

            let name = dev.name().unwrap_or_else(|_e| "Unknown OpenCL Device".to_string());
            let vendor = dev.vendor().ok();

            // Keep indices stable for later selection if needed.
            let _platform_index = pidx;
            let _device_index = didx;

            out.push(GpuDevice {
                api: GpuApi::OpenCl,
                name,
                vendor,
                device_index: didx,
            });
        }
    }

    Ok(out)
}
