// Comments in English as requested.

use std::fmt;

/// GPU backend selection used by the engine at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackendKind {
    Cuda,
    Opencl,
}

/// Basic device identification + sizing hints.
/// This is intentionally lightweight and backend-agnostic.
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    /// Backend kind (CUDA or OpenCL).
    pub backend: GpuBackendKind,
    /// Stable index within that backend enumeration (0..N).
    pub index: usize,
    /// Human-readable name (e.g. "RTX 4090", "RX 7900 XTX").
    pub name: String,
    /// Vendor string (e.g. "NVIDIA", "AMD") if known.
    pub vendor: String,
    /// Total VRAM in bytes if known (0 if unknown).
    pub vram_total_bytes: u64,
    /// Free VRAM in bytes if known (0 if unknown).
    pub vram_free_bytes: u64,
}

impl GpuDeviceInfo {
    pub fn label(&self) -> String {
        format!(
            "{}:{} {} ({})",
            match self.backend {
                GpuBackendKind::Cuda => "CUDA",
                GpuBackendKind::Opencl => "OpenCL",
            },
            self.index,
            self.name,
            self.vendor
        )
    }
}

/// Device selection configuration derived from EngineConfig.
#[derive(Debug, Clone)]
pub struct GpuSelectConfig {
    /// Whether GPU is enabled at all.
    pub enabled: bool,
    /// Backend strategy (Auto/Cuda/Opencl/Off) is handled outside;
    /// here we only receive the already-resolved allowed backends.
    pub allow_cuda: bool,
    pub allow_opencl: bool,

    /// Use at most this many devices across all backends.
    pub max_devices: Option<usize>,

    /// Allowlist items:
    /// - numeric indexes: "0","1"
    /// - substrings matched against device label/name/vendor (case-insensitive)
    pub allowlist: Vec<String>,
}

/// Batch sizing config used by the auto-tuner.
#[derive(Debug, Clone)]
pub struct GpuBatchConfig {
    /// Minimum batch size per launch (per device).
    pub min_batch: usize,
    /// Maximum batch size per launch (per device).
    pub max_batch: usize,
    /// VRAM usage ratio (0..=0.95) used for auto sizing.
    pub vram_ratio: f32,
    /// Batch builder timeout in ms (engine uses this).
    pub batch_timeout_ms: u32,
    /// Pipelining: max in-flight batches per device (engine uses this).
    pub inflight_batches: usize,
}

/// Result of device selection + computed batch sizes.
#[derive(Debug, Clone)]
pub struct GpuPlan {
    /// Active devices.
    pub devices: Vec<GpuPlannedDevice>,
    /// Total minimum batch across all active devices (used by engine reservations).
    pub min_batch_total: usize,
}

/// A selected device with computed batch sizing.
#[derive(Debug, Clone)]
pub struct GpuPlannedDevice {
    pub info: GpuDeviceInfo,
    /// Auto computed batch size for this device (clamped).
    pub batch_size: usize,
    /// Min batch for this device (clamped).
    pub min_batch: usize,
}

impl fmt::Display for GpuPlannedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} batch_size={} (min={})",
            self.info.label(),
            self.batch_size,
            self.min_batch
        )
    }
}

/// A very rough bytes/job estimate for the current CPU-stub pipeline.
///
/// Once CUDA/OpenCL kernels are implemented, replace with accurate packing size.
/// For now we assume:
/// - challenge: 32 bytes
/// - output (y): 100 bytes (classgroup element-ish; actual may vary)
/// - witness: ~100 bytes (varies)
/// - plus overhead/alignment
pub fn estimate_bytes_per_job() -> u64 {
    32 + 100 + 100 + 128
}

/// Compute an auto batch size for a device.
/// - If VRAM free is known, use vram_free * ratio
/// - Otherwise use a conservative fixed heuristic.
pub fn auto_batch_size_for_device(
    device: &GpuDeviceInfo,
    cfg: &GpuBatchConfig,
    bytes_per_job: u64,
) -> (usize, usize) {
    let min_b = cfg.min_batch.max(1);
    let max_b = cfg.max_batch.max(min_b);

    // If VRAM is unknown, pick a conservative batch.
    if device.vram_free_bytes == 0 || bytes_per_job == 0 {
        let batch = (min_b * 2).min(max_b);
        return (batch, min_b);
    }

    let ratio = cfg.vram_ratio.clamp(0.0, 0.95);
    let usable = (device.vram_free_bytes as f64 * ratio as f64).floor() as u64;
    let raw = (usable / bytes_per_job).max(1) as usize;

    // Round to a multiple of 32 for nicer GPU occupancy alignment.
    let rounded = (raw / 32).max(1) * 32;

    let batch = rounded.clamp(min_b, max_b);
    (batch, min_b)
}

/// Case-insensitive check: allowlist item matches this device.
pub fn allowlist_matches(device: &GpuDeviceInfo, allow_item: &str) -> bool {
    let item = allow_item.trim();
    if item.is_empty() {
        return false;
    }

    // Numeric index match
    if let Ok(idx) = item.parse::<usize>() {
        return device.index == idx;
    }

    let needle = item.to_ascii_lowercase();
    let hay = format!("{} {} {}", device.label(), device.name, device.vendor).to_ascii_lowercase();
    hay.contains(&needle)
}
