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
/// - This calls the `vdf_prove` kernel in `bbr-cuda-runtime` and provides a real CUDA execution path
///   (H2D -> kernel -> D2H). The current CUDA kernel may still be a placeholder; tests are structured
///   accordingly.
/// - All unsafe CUDA interaction remains inside `bbr-cuda-runtime`.
pub(crate) fn prove_vdf_batch(device_index: usize, challenges: &[u8]) -> anyhow::Result<Vec<u8>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    const DISCRIMINANT_BITS: usize = 1024;

    fn default_x_s() -> [u8; 100] {
        // Must match worker.rs default classgroup element.
        let mut el = [0u8; 100];
        el[0] = 0x08;
        el
    }

    fn fixed_challenge(seed: u8) -> [u8; 32] {
        // Deterministic but non-trivial challenges for golden inputs.
        let mut c = [0u8; 32];
        for (i, b) in c.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8).wrapping_mul(31).wrapping_add(7);
        }
        c
    }

    fn cpu_reference_y_witness(challenge: &[u8; 32], iters: u64) -> [u8; 200] {
        // Uses the fast chiavdf CPU oracle to compute y||witness (200 bytes for 1024-bit).
        let x = default_x_s();
        let out = bbr_client_chiavdf_fast::prove_one_weso_fast(challenge, &x, DISCRIMINANT_BITS, iters)
            .expect("cpu reference prove_one_weso_fast failed");
        assert_eq!(
            out.len(),
            200,
            "expected 200 bytes (y||witness) for 1024-bit discriminant"
        );
        let mut buf = [0u8; 200];
        buf.copy_from_slice(&out);
        buf
    }

    fn cuda_available(device_index: usize) -> bool {
        // A lightweight runtime check to decide whether to run CUDA-dependent tests.
        // This returns false if the driver/device is not available on the current machine.
        run_cuda_smoketest(device_index, 32).is_ok()
    }

    #[test]
    fn cpu_golden_inputs_are_deterministic() {
        // Golden *inputs* are deterministic by construction; this test ensures the CPU oracle
        // is deterministic for those inputs as well.
        let c0 = fixed_challenge(1);
        let c1 = fixed_challenge(2);

        let a = cpu_reference_y_witness(&c0, 250);
        let b = cpu_reference_y_witness(&c0, 250);
        assert_eq!(a, b, "CPU oracle must be deterministic for fixed inputs");

        let x = cpu_reference_y_witness(&c1, 250);
        assert_ne!(a, x, "Different challenges should yield different outputs");
    }

    #[test]
    fn gpu_batch_shape_is_correct_when_cuda_is_available() {
        // This test is intentionally shape-only. It verifies packing + kernel + unpacking returns
        // the expected buffer length (N*200), without assuming cryptographic correctness.
        if !cuda_available(0) {
            eprintln!("skipping: CUDA device 0 not available");
            return;
        }

        let challenges = [fixed_challenge(10), fixed_challenge(11), fixed_challenge(12)];
        let mut packed = Vec::with_capacity(challenges.len() * 32);
        for c in &challenges {
            packed.extend_from_slice(c);
        }

        let out = prove_vdf_batch(0, &packed).expect("prove_vdf_batch failed");
        assert_eq!(out.len(), challenges.len() * 200, "output must be N*200 bytes");
    }

    #[test]
    #[ignore = "Enable once vdf_prove implements real chiavdf (bit-for-bit parity)."]
    fn gpu_matches_cpu_golden_vectors() {
        // This is the key parity harness: CPU is treated as the oracle, and GPU must match exactly.
        // Keep this ignored until the CUDA kernel is cryptographically correct.
        if !cuda_available(0) {
            eprintln!("skipping: CUDA device 0 not available");
            return;
        }

        // Keep iterations modest so the CPU oracle is fast enough for dev machines.
        // Increase/add vectors as the CUDA implementation matures.
        let vectors = [
            (fixed_challenge(1), 250u64),
            (fixed_challenge(2), 250u64),
            (fixed_challenge(3), 500u64),
        ];

        let mut packed = Vec::with_capacity(vectors.len() * 32);
        let mut cpu = Vec::with_capacity(vectors.len());
        for (c, iters) in vectors {
            packed.extend_from_slice(&c);
            cpu.push(cpu_reference_y_witness(&c, iters));
        }

        let gpu = prove_vdf_batch(0, &packed).expect("prove_vdf_batch failed");
        assert_eq!(gpu.len(), cpu.len() * 200);

        for (i, cpu_out) in cpu.iter().enumerate() {
            let gpu_out = &gpu[i * 200..(i + 1) * 200];
            assert_eq!(gpu_out, cpu_out, "GPU output mismatch for vector index {i}");
        }
    }
}
