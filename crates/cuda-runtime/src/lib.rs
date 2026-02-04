// Comments in English as requested.

#![allow(clippy::needless_return)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
// This crate intentionally permits unsafe code internally to perform CUDA kernel launches.
// The public API is safe.

use anyhow::Context as _;

use cust::prelude::*;

/// Run a minimal CUDA end-to-end kernel test on a selected device.
///
/// This function validates:
/// - CUDA driver is present
/// - device selection works
/// - context creation works
/// - H2D and D2H copies work
/// - kernel launch works
///
/// The kernel is a trivial "add1" which computes: `out[i] = in[i] + 1`.
pub fn add1_smoketest(device_index: usize, n: usize) -> anyhow::Result<()> {
    init_cuda().context("CUDA init")?;

    let dev = Device::get_device(device_index)
        .with_context(|| format!("CUDA device {device_index} not available"))?;

    // Context must be current on the calling thread.
    let _ctx = Context::create_and_push(ContextFlags::SCHED_AUTO, dev)
        .context("failed to create CUDA context")?;

    let module = unsafe { Module::from_ptx(ADD1_PTX, &[]) }.context("failed to load PTX module")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).context("failed to create stream")?;

    let n = n.max(1);
    let n_u32: u32 = u32::try_from(n).context("n too large for u32")?;

    let input: Vec<u32> = (0..n_u32).collect();
    let mut output: Vec<u32> = vec![0; input.len()];

    let in_dev = DeviceBuffer::from_slice(&input).context("failed to allocate input buffer")?;
    let mut out_dev: DeviceBuffer<u32> =
        DeviceBuffer::uninitialized(input.len()).context("failed to allocate output buffer")?;

    // Launch config: 256 threads per block.
    let threads_per_block: u32 = 256;
    let blocks: u32 = (n_u32 + threads_per_block - 1) / threads_per_block;

    let func = module.get_function("add1").context("failed to get function add1")?;

    unsafe {
        launch!(
            func<<<blocks, threads_per_block, 0, stream>>>(
                in_dev.as_device_ptr(),
                out_dev.as_device_ptr(),
                n_u32
            )
        )
        .context("kernel launch failed")?;
    }

    stream.synchronize().context("stream synchronize failed")?;

    out_dev
        .copy_to(&mut output)
        .context("failed to copy output back")?;

    // Validate a few samples (lightweight guard).
    for i in 0..input.len().min(16) {
        let expected = input[i].wrapping_add(1);
        if output[i] != expected {
            anyhow::bail!(
                "CUDA add1 smoketest mismatch at {i}: expected {expected}, got {}",
                output[i]
            );
        }
    }

    Ok(())
}

/// Execute the trivial `add1` kernel on the provided input slice and return the output.
///
/// Kernel semantics: `out[i] = input[i] + 1`.
pub fn add1_execute(device_index: usize, input: &[u32]) -> anyhow::Result<Vec<u32>> {
    init_cuda().context("CUDA init")?;

    let dev = Device::get_device(device_index)
        .with_context(|| format!("CUDA device {device_index} not available"))?;

    // Context must be current on the calling thread.
    let _ctx = Context::create_and_push(ContextFlags::SCHED_AUTO, dev)
        .context("failed to create CUDA context")?;

    let module = unsafe { Module::from_ptx(ADD1_PTX, &[]) }.context("failed to load PTX module")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).context("failed to create stream")?;

    // If input is empty, keep the path uniform: run on a single dummy element.
    let host_in: Vec<u32> = if input.is_empty() { vec![0u32] } else { input.to_vec() };
    let mut host_out: Vec<u32> = vec![0u32; host_in.len()];

    let n_u32: u32 = u32::try_from(host_in.len()).context("input too large for u32")?;

    let in_dev = DeviceBuffer::from_slice(&host_in).context("failed to allocate input buffer")?;
    let mut out_dev: DeviceBuffer<u32> =
        DeviceBuffer::uninitialized(host_in.len()).context("failed to allocate output buffer")?;

    let threads_per_block: u32 = 256;
    let blocks: u32 = (n_u32 + threads_per_block - 1) / threads_per_block;

    let func = module.get_function("add1").context("failed to get function add1")?;

    unsafe {
        launch!(
            func<<<blocks, threads_per_block, 0, stream>>>(
                in_dev.as_device_ptr(),
                out_dev.as_device_ptr(),
                n_u32
            )
        )
        .context("kernel launch failed")?;
    }

    stream.synchronize().context("stream synchronize failed")?;

    out_dev
        .copy_to(&mut host_out)
        .context("failed to copy output back")?;

    if input.is_empty() {
        return Ok(Vec::new());
    }

    Ok(host_out)
}

fn init_cuda() -> anyhow::Result<()> {
    cust::init(CudaFlags::empty()).context("cust::init failed")?;
    Ok(())
}

// PTX for a trivial add1 kernel.
// Target is set to a low SM to be broadly compatible; the driver will JIT as needed.
//
// Kernel signature:
//   extern "C" __global__ void add1(const unsigned int* in, unsigned int* out, unsigned int n)
const ADD1_PTX: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry add1(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
)
{
    .reg .pred  %p<2>;
    .reg .b32   %r<6>;
    .reg .b64   %rd<6>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    ld.param.u32 %r1,  [n];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;

    mad.lo.s32 %r5, %r2, %r3, %r4;     // idx = blockIdx.x * blockDim.x + threadIdx.x

    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;

    mul.wide.u32 %rd4, %r5, 4;

    add.u64 %rd5, %rd1, %rd4;
    ld.global.u32 %r2, [%rd5];

    add.u32 %r2, %r2, 1;

    add.u64 %rd5, %rd2, %rd4;
    st.global.u32 [%rd5], %r2;

DONE:
    ret;
}
"#;
