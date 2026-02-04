// Comments in English as requested.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

//! CUDA backend glue for the client engine.
//!
//! This module is intentionally small and safe: all `unsafe` CUDA interaction is contained
//! in the `bbr-cuda-runtime` crate.

use anyhow::Context as _;

/// Run the CUDA smoketest on the specified device.
///
/// This is a lightweight health-check (H2D -> kernel -> D2H) and should not be run too often.
pub(crate) fn run_cuda_smoketest(device_index: usize, n: usize) -> anyhow::Result<()> {
    bbr_cuda_runtime::add1_smoketest(device_index, n)
        .with_context(|| format!("CUDA smoketest failed on device {device_index}"))
}

/// Execute the trivial add1 kernel for an entire packed batch.
///
/// This provides a real "pack -> H2D -> kernel -> D2H -> unpack" path, used as a scaffolding
/// until the real VDF CUDA kernels are implemented.
pub(crate) fn add1_batch(device_index: usize, input: &[u32]) -> anyhow::Result<Vec<u32>> {
    bbr_cuda_runtime::add1_execute(device_index, input)
        .with_context(|| format!("CUDA add1_batch failed on device {device_index}"))
}

/// Execute a shape-correct GPU batch for Weso proofs.
///
/// Semantics:
/// - `challenges` is a concatenation of N challenges, each exactly 32 bytes.
/// - Return buffer is N * 200 bytes.
///   - first 100 bytes are `y`
///   - second 100 bytes are `witness`
///
/// Notes:
/// - This is currently a *stub compute* kernel, but it is a real CUDA execution path
///   (H2D -> kernel -> D2H) and matches the required buffer shapes.
/// - All unsafe CUDA interaction remains inside `bbr-cuda-runtime`.
pub(crate) fn prove_stub_batch(device_index: usize, challenges: &[u8]) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        challenges.len() % 32 == 0,
        "invalid packed challenges length: expected multiple of 32, got {}",
        challenges.len()
    );

    let jobs = challenges.len() / 32;

    // Pack as u32 words (little-endian): 32 bytes per job -> 8 u32.
    let mut words: Vec<u32> = Vec::with_capacity(jobs * 8);
    for chunk in challenges.chunks_exact(4) {
        words.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }

    let out_words = bbr_cuda_runtime::prove_vdf_execute(device_index, &words, jobs)
        .with_context(|| format!("CUDA prove_vdf_batch failed on device {device_index}"))?;

    // Unpack u32 words back to bytes.
    let mut out: Vec<u8> = Vec::with_capacity(out_words.len() * 4);
    for w in out_words {
        out.extend_from_slice(&w.to_le_bytes());
    }

    Ok(out)
}

/// Execute a shape-correct VDF batch (CUDA).
///
/// This is the preferred entrypoint for the engine. It keeps the same buffers/shape as the
/// previous stub, but is wired to the `vdf_prove` PTX kernel (see `bbr-cuda-runtime`).
pub(crate) fn prove_vdf_batch(device_index: usize, challenges: &[u8]) -> anyhow::Result<Vec<u8>> {
    prove_stub_batch(device_index, challenges)
}
