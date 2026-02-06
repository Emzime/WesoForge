use crate::{GpuApi, GpuDevice};

/// Quick availability check (driver present, API usable).
pub(crate) fn is_available() -> Result<bool, ()> {
    // If CUDA is compiled in, try initializing the driver.
    // Any error means not available.
    match cust::CudaFlags::empty() {
        _ => {}
    }

    cust::quick_init().map(|_ctx| true).map_err(|_e| ())
}

/// CUDA context wrapper.
pub(crate) struct CudaContext {
    _context: cust::context::Context,
}

impl CudaContext {
    /// Initialize CUDA (select device 0 by default).
    pub(crate) fn new() -> Result<Self, String> {
        cust::quick_init()
            .map(|ctx| Self { _context: ctx })
            .map_err(|e| format!("{e}"))
    }
}

/// List CUDA devices.
///
/// Note: uses driver API via `cust`.
pub(crate) fn detect_cuda_devices() -> Result<Vec<GpuDevice>, String> {
    let _ctx = cust::quick_init().map_err(|e| format!("{e}"))?;

    let count = cust::device::Device::num_devices().map_err(|e| format!("{e}"))?;
    let mut out = Vec::with_capacity(count as usize);

    for i in 0..count {
        let dev = cust::device::Device::get_device(i).map_err(|e| format!("{e}"))?;
        let name = dev.name().map_err(|e| format!("{e}"))?;

        out.push(GpuDevice {
            api: GpuApi::Cuda,
            name,
            vendor: Some("NVIDIA".to_string()),
            device_index: i as usize,
        });
    }

    Ok(out)
}
