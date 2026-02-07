use cust::prelude::*;

use crate::{GpuApi, GpuDevice};

const KERNELS_PTX: &str = include_str!("kernels.ptx");

/// Quick availability check (CUDA driver present and at least one device).
pub(crate) fn is_available() -> Result<bool, ()> {
    cust::init(cust::CudaFlags::empty()).map_err(|_e| ())?;
    let count = Device::num_devices().map_err(|_e| ())?;
    Ok(count > 0)
}

pub(crate) struct CudaContext {
    _context: cust::context::Context,
}

impl CudaContext {
    pub(crate) fn new() -> Result<Self, String> {
        let ctx = cust::quick_init().map_err(|e| format!("{e}"))?;
        Ok(Self { _context: ctx })
    }

    pub(crate) fn new_for_device(device_index: usize) -> Result<Self, String> {
        cust::init(cust::CudaFlags::empty()).map_err(|e| format!("{e}"))?;
        let dev = Device::get_device(device_index as u32).map_err(|e| format!("{e}"))?;

        // cust 0.3.x: Context::new(device) is the supported API (no create_and_push).
        let ctx = cust::context::Context::new(dev).map_err(|e| format!("{e}"))?;
        Ok(Self { _context: ctx })
    }
}

/// List CUDA devices as `GpuDevice`.
pub(crate) fn detect_cuda_devices() -> Result<Vec<GpuDevice>, String> {
    cust::init(cust::CudaFlags::empty()).map_err(|e| format!("{e}"))?;
    let count = Device::num_devices().map_err(|e| format!("{e}"))?;

    let mut out = Vec::new();
    for idx in 0..count {
        let dev = Device::get_device(idx).map_err(|e| format!("{e}"))?;
        let name = dev.name().map_err(|e| format!("{e}"))?;
        out.push(GpuDevice {
            api: GpuApi::Cuda,
            name,
            vendor: Some("NVIDIA".to_string()),
            device_index: idx as usize,
        });
    }
    Ok(out)
}

pub(crate) fn run_spin_kernel(num_iterations: u64) -> Result<(), String> {
    run_spin_kernel_on_device(0, num_iterations)
}

pub(crate) fn run_spin_kernel_on_device(device_index: usize, num_iterations: u64) -> Result<(), String> {
    let _ctx = CudaContext::new_for_device(device_index)?;
    run_spin_kernel_on_current_device(device_index, num_iterations)
}

fn run_spin_kernel_on_current_device(device_index: usize, num_iterations: u64) -> Result<(), String> {
    let module = Module::from_ptx(KERNELS_PTX, &[]).map_err(|e| format!("{e}"))?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| format!("{e}"))?;
    let function = module.get_function("spin_kernel").map_err(|e| format!("{e}"))?;

    let len: usize = 1 << 18;
    let iters: u32 = ((num_iterations / len as u64).max(1).min(u32::MAX as u64)) as u32;

    eprintln!(
        "[client-gpu][cuda] spin_kernel launch: device={} len={} iters={} (scaled from num_iterations={})",
        device_index,
        len,
        iters,
        num_iterations
    );

    let buf = DeviceBuffer::<u32>::zeroed(len).map_err(|e| format!("{e}"))?;

    // Pass the raw device pointer as u64 (cust param packing compatibility).
    let buf_ptr: u64 = buf.as_device_ptr().as_raw();

    let block: u32 = 256;
    let grid: u32 = ((len as u32) + block - 1) / block;

    unsafe {
        launch!(function<<<grid, block, 0, stream>>>(buf_ptr, len as u32, iters))
            .map_err(|e| format!("{e}"))?;
    }

    stream.synchronize().map_err(|e| format!("{e}"))?;
    Ok(())
}
