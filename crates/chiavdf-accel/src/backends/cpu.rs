use crate::{AccelError, AccelResult, ProgressFn, VdfBackend};

/// CPU backend using the existing chiavdf fast wrapper.
pub struct CpuBackend {
    name: String,
}

impl CpuBackend {
    /// Create CPU backend.
    pub fn new() -> AccelResult<Self> {
        Ok(Self { name: "cpu".to_string() })
    }
}

impl VdfBackend for CpuBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn is_accelerated(&self) -> bool {
        false
    }

    fn prove_weso_witness(
        &self,
        challenge: &[u8],
        x: &[u8],
        expected_y: &[u8],
        discriminant_bits: usize,
        iters: u64,
        progress_interval: u64,
        progress: Option<&ProgressFn>,
    ) -> AccelResult<(Vec<u8>, bool)> {
        if challenge.is_empty() {
            return Err(AccelError::InvalidInput("challenge must not be empty"));
        }
        if x.is_empty() {
            return Err(AccelError::InvalidInput("x must not be empty"));
        }
        if expected_y.is_empty() {
            return Err(AccelError::InvalidInput("expected_y must not be empty"));
        }
        if discriminant_bits == 0 {
            return Err(AccelError::InvalidInput("discriminant_bits must be > 0"));
        }
        if iters == 0 {
            return Err(AccelError::InvalidInput("iters must be > 0"));
        }

        // NOTE: This assumes your existing dependency exposes the streaming API you already use
        // in WesoForge worker code.
        let out = if progress_interval == 0 || progress.is_none() {
            bbr_client_chiavdf_fast::prove_one_weso_fast_streaming(
                challenge,
                x,
                expected_y,
                discriminant_bits,
                iters,
            ).map_err(|e| AccelError::CpuFailure(format!("{e:?}")))?
        } else {
            let cb = progress.unwrap();
            bbr_client_chiavdf_fast::prove_one_weso_fast_streaming_with_progress(
                challenge,
                x,
                expected_y,
                discriminant_bits,
                iters,
                progress_interval,
                move |done| (cb)(done),
            ).map_err(|e| AccelError::CpuFailure(format!("{e:?}")))?
        };

        if out.len() < 2 || out.len() % 2 != 0 {
            return Err(AccelError::CpuFailure(format!("unexpected output length: {}", out.len())));
        }

        let half = out.len() / 2;
        let y = &out[..half];
        let witness = out[half..].to_vec();
        Ok((witness, y != expected_y))
    }
}
