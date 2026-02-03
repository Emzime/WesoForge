#![forbid(unsafe_code)]
#![deny(unreachable_pub)]
#![deny(missing_docs)]

//! GPU/CPU abstraction layer for Bluebox compaction (Wesolowski witness computation).
//!
//! This crate provides a single interface (`VdfBackend`) with multiple implementations:
//! - CPU: uses the existing chiavdf fast wrapper
//! - CUDA: NVIDIA backend (optional feature)
//! - OpenCL: multi-vendor backend (optional feature)

mod dispatch;
mod error;

pub mod backends;

pub use dispatch::{BackendKind, BackendOptions, SelectedBackend};
pub use error::{AccelError, AccelResult};

/// Progress callback invoked with `iters_done`.
pub type ProgressFn = dyn Fn(u64) + Send + Sync + 'static;

/// Unified backend interface (CPU/GPU).
pub trait VdfBackend: Send + Sync {
    /// Human readable name (e.g. "cpu", "cuda:0", "opencl:amd").
    fn name(&self) -> &str;

    /// Returns true if this backend is actually hardware-accelerated.
    fn is_accelerated(&self) -> bool;

    /// Compute witness for a single job (may internally batch or stream).
    ///
    /// Returns `(witness, output_mismatch)`.
    fn prove_weso_witness(
        &self,
        challenge: &[u8],
        x: &[u8],
        expected_y: &[u8],
        discriminant_bits: usize,
        iters: u64,
        progress_interval: u64,
        progress: Option<&ProgressFn>,
    ) -> AccelResult<(Vec<u8>, bool)>;
}
