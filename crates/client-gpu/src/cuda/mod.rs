use cust::prelude::*;

use crate::{GpuApi, GpuDevice};

const PTX: &str = include_str!("kernels.ptx");

/// Quick availability check (driver present, API usable).
pub(crate) fn is_available() -> Result<bool, ()> {
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

/// Phase-1 CUDA compute: run a deterministic "spin" kernel whose cost scales with `num_iterations`.
///
/// This is real CUDA execution plumbing (module load, device alloc, kernel launch, sync).
/// It does not yet implement the VDF prover math; the CPU prover remains the source of truth.
pub(crate) fn run_spin_kernel(num_iterations: u64) -> Result<(), String> {
    let _ctx = cust::quick_init().map_err(|e| format!("{e}"))?;

    let len: u32 = 1 << 20; // ~4MB
    let iters: u32 = ((num_iterations / 1024).clamp(1, 50_000)) as u32;

    // Initialize host buffer, then upload in a single safe call.
    let host_init: Vec<u32> = (0..len).map(|i| i ^ 0xA5A5_5A5A).collect();
    let d_buf = DeviceBuffer::<u32>::from_slice(&host_init).map_err(|e| format!("{e}"))?;

    let module = Module::from_ptx(PTX, &[]).map_err(|e| format!("{e}"))?;
    let func = module.get_function("spin_kernel").map_err(|e| format!("{e}"))?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| format!("{e}"))?;

    let block = 256u32;
    let grid = (len + block - 1) / block;

    unsafe {
        launch!(
            func<<<grid, block, 0, stream>>>(
                d_buf.as_device_ptr(),
                iters,
                len
            )
        )
        .map_err(|e| format!("{e}"))?;
    }

    stream.synchronize().map_err(|e| format!("{e}"))?;

    // Read back one value so the buffer is used.
    let mut one = [0u32; 1];
    d_buf.copy_to(&mut one).map_err(|e| format!("{e}"))?;
    let _sanity = one[0];

    Ok(())
}
