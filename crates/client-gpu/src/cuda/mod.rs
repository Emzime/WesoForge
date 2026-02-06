use cust::prelude::*;

use crate::{GpuApi, GpuDevice};

const PTX: &str = include_str!("kernels.ptx");

pub(crate) fn is_available() -> Result<bool, ()> {
    cust::quick_init().map(|_ctx| true).map_err(|_e| ())
}

pub(crate) struct CudaContext {
    _context: cust::context::Context,
}

impl CudaContext {
    pub(crate) fn new() -> Result<Self, String> {
        cust::quick_init()
            .map(|ctx| Self { _context: ctx })
            .map_err(|e| format!("{e}"))
    }
}

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

/// Phase-1 CUDA compute: real kernel execution plumbing without any host<->device slice copies.
///
/// We avoid `DeviceBuffer::from_slice` / `copy_to` here because `cust` panics on length mismatch.
/// Instead we:
/// 1) allocate a device buffer
/// 2) run an `init_kernel` that deterministically fills it
/// 3) run `spin_kernel` to spend time proportional to `num_iterations`
/// 4) synchronize
pub(crate) fn run_spin_kernel(num_iterations: u64) -> Result<(), String> {
    let _ctx = cust::quick_init().map_err(|e| format!("{e}"))?;

    let len: u32 = 1 << 18; // 262,144 elements (~1MB)
    let iters: u32 = ((num_iterations / 1_000_000).clamp(1, 2_000)) as u32;

    eprintln!(
        "[client-gpu][cuda] spin_kernel launch: len={len} iters={iters} (scaled from num_iterations={num_iterations})"
    );

    // SAFETY: buffer is uninitialized but will be fully written by init_kernel before any read.
    let d_buf = unsafe { DeviceBuffer::<u32>::uninitialized(len as usize) }
        .map_err(|e| format!("{e}"))?;

    // IMPORTANT: pass raw device pointers to kernels.
    //
    // On some `cust` versions / platforms, `DevicePointer<T>` can have a host-side size
    // that differs from 8 bytes (e.g. it may include an extra marker field). The `launch!`
    // macro packs kernel arguments by copying their raw bytes into an internal parameter
    // buffer; if the argument type size doesn't match what the packer expects for a
    // pointer-sized value, it can panic with:
    // "destination and source slices have different lengths".
    //
    // Our PTX expects `.param .u64 buf_ptr`, so we pass the underlying `u64` explicitly.
    let d_ptr: u64 = d_buf.as_device_ptr().as_raw();

    let module = Module::from_ptx(PTX, &[]).map_err(|e| format!("{e}"))?;
    let init = module.get_function("init_kernel").map_err(|e| format!("{e}"))?;
    let spin = module.get_function("spin_kernel").map_err(|e| format!("{e}"))?;

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(|e| format!("{e}"))?;

    let block = 256u32;
    let grid = (len + block - 1) / block;

    unsafe {
        launch!(init<<<grid, block, 0, stream>>>(d_ptr, len)).map_err(|e| format!("{e}"))?;

        launch!(spin<<<grid, block, 0, stream>>>(d_ptr, iters, len)).map_err(|e| format!("{e}"))?;
    }

    stream.synchronize().map_err(|e| format!("{e}"))?;

    // Keep buffer alive until after sync.
    std::mem::drop(d_buf);

    eprintln!("[client-gpu][cuda] spin_kernel done");

    Ok(())
}
