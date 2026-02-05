// Comments in English as requested.

#![allow(clippy::needless_return)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
// This crate intentionally permits unsafe code internally to perform CUDA kernel launches.
// The public API is safe.

use anyhow::Context as _;
use cust::prelude::*;

fn create_context(device_index: usize) -> anyhow::Result<Context> {
    // cust::Device::get_device expects a u32 ordinal.
    let ordinal: u32 = u32::try_from(device_index).context("device_index too large for u32")?;

    let dev = Device::get_device(ordinal)
        .with_context(|| format!("CUDA device {device_index} not available"))?;

    // In cust 0.3.x, Context::new retains the device's primary context and makes it current
    // on the calling thread.
    let ctx = Context::new(dev).context("failed to create CUDA context")?;
    Ok(ctx)
}

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

    let _ctx = create_context(device_index)?;

    // Module::from_ptx is safe in cust 0.3.x (no need for `unsafe`).
    let module = Module::from_ptx(ADD1_PTX, &[]).context("failed to load PTX module")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).context("failed to create stream")?;

    let n = n.max(1);
    let n_u32: u32 = u32::try_from(n).context("n too large for u32")?;

    let input: Vec<u32> = (0..n_u32).collect();
    let mut output: Vec<u32> = vec![0; input.len()];

    let in_dev = DeviceBuffer::from_slice(&input).context("failed to allocate input buffer")?;

    // DeviceBuffer::uninitialized is unsafe: the device memory is uninitialized.
    // Safety: the kernel writes every element of `out_dev` before we copy it back to host.
    let out_dev: DeviceBuffer<u32> = unsafe {
        DeviceBuffer::uninitialized(input.len()).context("failed to allocate output buffer")?
    };

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

    let _ctx = create_context(device_index)?;

    let module = Module::from_ptx(ADD1_PTX, &[]).context("failed to load PTX module")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).context("failed to create stream")?;

    // If input is empty, keep the path uniform: run on a single dummy element.
    let host_in: Vec<u32> = if input.is_empty() { vec![0u32] } else { input.to_vec() };
    let mut host_out: Vec<u32> = vec![0u32; host_in.len()];

    let n_u32: u32 = u32::try_from(host_in.len()).context("input too large for u32")?;

    let in_dev = DeviceBuffer::from_slice(&host_in).context("failed to allocate input buffer")?;

    // Safety: the kernel writes every element of `out_dev` before the D2H copy.
    let out_dev: DeviceBuffer<u32> = unsafe {
        DeviceBuffer::uninitialized(host_in.len()).context("failed to allocate output buffer")?
    };

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

/// Execute a shape-correct stub "proof" batch on the GPU.
///
/// Input layout:
/// - `challenges_words` must contain `jobs * 8` u32 words (32 bytes per job).
///
/// Output layout:
/// - returns `jobs * 50` u32 words (200 bytes per job):
///   - first 25 words (100 bytes) are "y"
///   - second 25 words (100 bytes) are "witness"
///
/// This does NOT implement the real VDF; it is a shape-correct CUDA execution path intended
/// to validate packing, device routing, and submission plumbing.
pub fn prove_stub_execute(
    device_index: usize,
    challenges_words: &[u32],
    jobs: usize,
) -> anyhow::Result<Vec<u32>> {
    init_cuda().context("CUDA init")?;

    let expected_words = jobs.checked_mul(8).context("jobs overflow")?;
    anyhow::ensure!(
        challenges_words.len() == expected_words,
        "invalid challenges_words length: expected {expected_words}, got {}",
        challenges_words.len()
    );

    // Uniform handling: if jobs == 0, return empty.
    if jobs == 0 {
        return Ok(Vec::new());
    }

    let _ctx = create_context(device_index)?;

    let module = Module::from_ptx(PROVE_STUB_PTX, &[]).context("failed to load PTX module")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).context("failed to create stream")?;

    let total_out_words = jobs.checked_mul(50).context("jobs overflow")?;

    let jobs_u32: u32 = u32::try_from(jobs).context("jobs too large for u32")?;
    let n_u32: u32 = u32::try_from(total_out_words).context("output too large for u32")?;

    let in_dev = DeviceBuffer::from_slice(challenges_words).context("failed to allocate input buffer")?;

    // Safety: the kernel writes every element of the output buffer.
    let out_dev: DeviceBuffer<u32> = unsafe {
        DeviceBuffer::uninitialized(total_out_words).context("failed to allocate output buffer")?
    };

    let threads_per_block: u32 = 256;
    let blocks: u32 = (n_u32 + threads_per_block - 1) / threads_per_block;

    let func = module
        .get_function("prove_stub")
        .context("failed to get function prove_stub")?;

    unsafe {
        launch!(
            func<<<blocks, threads_per_block, 0, stream>>>(
                in_dev.as_device_ptr(),
                out_dev.as_device_ptr(),
                jobs_u32
            )
        )
        .context("kernel launch failed")?;
    }

    stream.synchronize().context("stream synchronize failed")?;

    let mut host_out: Vec<u32> = vec![0u32; total_out_words];
    out_dev
        .copy_to(&mut host_out)
        .context("failed to copy output back")?;

    Ok(host_out)
}

/// Execute the real VDF batch kernel on the GPU.
///
/// Input layout:
/// - `challenges_words` must contain `jobs * 8` u32 words (32 bytes per job).
///
/// Output layout:
/// - returns `jobs * 50` u32 words (200 bytes per job):
///   - first 25 words (100 bytes) are `y`
///   - second 25 words (100 bytes) are `witness`
///
/// Notes:
/// - This function is wired to the `vdf_prove` PTX kernel embedded in this crate.
/// - The surrounding plumbing (packing/unpacking, worker submission) assumes these shapes.
/// - Replace `src/vdf.ptx` with your actual kernel PTX without changing any Rust code.
pub fn prove_vdf_execute(
    device_index: usize,
    challenges_words: &[u32],
    jobs: usize,
) -> anyhow::Result<Vec<u32>> {
    init_cuda().context("CUDA init")?;

    let expected_words = jobs.checked_mul(8).context("jobs overflow")?;
    anyhow::ensure!(
        challenges_words.len() == expected_words,
        "invalid challenges_words length: expected {expected_words}, got {}",
        challenges_words.len()
    );

    // Uniform handling: if jobs == 0, return empty.
    if jobs == 0 {
        return Ok(Vec::new());
    }

    let _ctx = create_context(device_index)?;

    let module = Module::from_ptx(VDF_PTX, &[]).context("failed to load VDF PTX module")?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).context("failed to create stream")?;

    let total_out_words = jobs.checked_mul(50).context("jobs overflow")?;
    let jobs_u32: u32 = u32::try_from(jobs).context("jobs too large for u32")?;
    let n_u32: u32 = u32::try_from(total_out_words).context("output too large for u32")?;

    let in_dev = DeviceBuffer::from_slice(challenges_words).context("failed to allocate input buffer")?;

    // Safety: the kernel writes every element of the output buffer.
    let out_dev: DeviceBuffer<u32> = unsafe {
        DeviceBuffer::uninitialized(total_out_words).context("failed to allocate output buffer")?
    };

    let threads_per_block: u32 = 256;
    let blocks: u32 = (n_u32 + threads_per_block - 1) / threads_per_block;

    let func = module
        .get_function("vdf_prove")
        .context("failed to get function vdf_prove")?;

    unsafe {
        launch!(
            func<<<blocks, threads_per_block, 0, stream>>>(
                in_dev.as_device_ptr(),
                out_dev.as_device_ptr(),
                jobs_u32
            )
        )
        .context("kernel launch failed")?;
    }

    stream.synchronize().context("stream synchronize failed")?;

    let mut host_out: Vec<u32> = vec![0u32; total_out_words];
    out_dev
        .copy_to(&mut host_out)
        .context("failed to copy output back")?;

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

// PTX for the actual VDF batch kernel.
//
// IMPORTANT:
// - The engine expects the output layout to be `jobs * 200 bytes`.
// - `y` must be written to the first 100 bytes and `witness` to the next 100 bytes.
// - The worker validates `y` against the backend-provided output before using the GPU witness.
//
// You can replace `src/vdf.ptx` with a real kernel PTX without touching Rust plumbing.
const VDF_PTX: &str = include_str!("vdf.ptx");

// PTX for a shape-correct stub proof kernel.
const PROVE_STUB_PTX: &str = r#"
.version 6.0
.target sm_30
.address_size 64

.visible .entry prove_stub(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 jobs
)
{
    .reg .pred  %p<6>;
    .reg .b32   %r<24>;
    .reg .b64   %rd<10>;

    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    ld.param.u32 %r1,  [jobs];

    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;

    mul.lo.u32 %r6, %r1, 50;

    setp.ge.u32 %p1, %r5, %r6;
    @%p1 bra DONE;

    div.u32 %r7, %r5, 50;
    mad.lo.s32 %r8, %r7, -50, %r5;

    mul.wide.u32 %rd3, %r5, 4;
    add.u64 %rd4, %rd2, %rd3;

    setp.lt.u32 %p2, %r8, 8;
    @%p2 bra LANE_LT_8;

    setp.lt.u32 %p3, %r8, 25;
    @%p3 bra LANE_LT_25;

    setp.lt.u32 %p4, %r8, 33;
    @%p4 bra LANE_LT_33;

    shl.b32 %r9, %r7, 16;
    xor.b32 %r10, %r9, %r8;
    xor.b32 %r10, %r10, 0xA5A5;
    st.global.u32 [%rd4], %r10;
    bra DONE;

LANE_LT_8:
    mad.lo.s32 %r11, %r7, 8, %r8;
    mul.wide.u32 %rd5, %r11, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u32 %r12, [%rd6];
    add.u32 %r12, %r12, 1;
    st.global.u32 [%rd4], %r12;
    bra DONE;

LANE_LT_25:
    shl.b32 %r9, %r7, 16;
    xor.b32 %r10, %r9, %r8;
    st.global.u32 [%rd4], %r10;
    bra DONE;

LANE_LT_33:
    add.u32 %r13, %r8, -25;
    mad.lo.s32 %r11, %r7, 8, %r13;
    mul.wide.u32 %rd5, %r11, 4;
    add.u64 %rd6, %rd1, %rd5;
    ld.global.u32 %r12, [%rd6];
    add.u32 %r12, %r12, 2;
    st.global.u32 [%rd4], %r12;
    bra DONE;

DONE:
    ret;
}
"#;
