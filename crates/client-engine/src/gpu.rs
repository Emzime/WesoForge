// Comments in English as requested.

use std::fmt;

/// GPU backend selection used by the engine at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum GpuBackendKind {
    Cuda,
    Opencl,
}

/// Structured GPU device identifier.
///
/// This is used across the engine/worker boundary to ensure stable routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct GpuDeviceId {
    pub(crate) backend: GpuBackendKind,
    pub(crate) index: usize,
}

impl GpuDeviceId {
    pub(crate) fn new(backend: GpuBackendKind, index: usize) -> Self {
        Self { backend, index }
    }
}

/// Basic device identification + sizing hints.
/// This is intentionally lightweight and backend-agnostic.
#[derive(Debug, Clone)]
pub(crate) struct GpuDeviceInfo {
    /// Backend kind (CUDA or OpenCL).
    pub(crate) backend: GpuBackendKind,
    /// Stable index within that backend enumeration (0..N).
    pub(crate) index: usize,
    /// Human-readable name (e.g. "RTX 4090", "RX 7900 XTX").
    pub(crate) name: String,
    /// Vendor string (e.g. "NVIDIA", "AMD") if known.
    pub(crate) vendor: String,

    /// Total memory in bytes if known.
    ///
    /// This field exists to keep compatibility with earlier iterations of the GPU plumbing.
    pub(crate) total_mem_bytes: Option<u64>,
    /// Free memory in bytes if known.
    pub(crate) free_mem_bytes: Option<u64>,

    /// Total VRAM in bytes if known (0 if unknown).
    pub(crate) vram_total_bytes: u64,
    /// Free VRAM in bytes if known (0 if unknown).
    pub(crate) vram_free_bytes: u64,
}

impl GpuDeviceInfo {
    pub(crate) fn id(&self) -> GpuDeviceId {
        GpuDeviceId::new(self.backend, self.index)
    }

    pub(crate) fn label(&self) -> String {
        // Read memory hint fields to avoid dead-code warnings and provide better diagnostics.
        let free = if self.vram_free_bytes > 0 {
            Some(self.vram_free_bytes)
        } else {
            self.free_mem_bytes
        };

        match free {
            Some(free) => format!(
                "{}:{} {} ({}) free={}B",
                match self.backend {
                    GpuBackendKind::Cuda => "CUDA",
                    GpuBackendKind::Opencl => "OpenCL",
                },
                self.index,
                self.name,
                self.vendor,
                free
            ),
            None => format!(
                "{}:{} {} ({})",
                match self.backend {
                    GpuBackendKind::Cuda => "CUDA",
                    GpuBackendKind::Opencl => "OpenCL",
                },
                self.index,
                self.name,
                self.vendor
            ),
        }
    }
}

/// Match a single allowlist token against a device.
///
/// Supported tokens:
/// - numeric index: "0", "1" (matches device index within backend)
/// - substring: matched case-insensitively against label/name/vendor
pub(crate) fn allowlist_matches(dev: &GpuDeviceInfo, token: &str) -> bool {
    let t = token.trim();
    if t.is_empty() {
        return false;
    }

    if let Ok(idx) = t.parse::<usize>() {
        return dev.index == idx;
    }

    let needle = t.to_ascii_lowercase();
    let hay_label = dev.label().to_ascii_lowercase();
    let hay_name = dev.name.to_ascii_lowercase();
    let hay_vendor = dev.vendor.to_ascii_lowercase();

    hay_label.contains(&needle) || hay_name.contains(&needle) || hay_vendor.contains(&needle)
}

/// Device selection configuration derived from EngineConfig.
#[derive(Debug, Clone)]
pub(crate) struct GpuSelectConfig {
    /// Whether GPU is enabled at all.
    pub enabled: bool,
    /// Backend strategy (Auto/Cuda/Opencl/Off) is handled outside;
    /// here we only receive the already-resolved allowed backends.
    pub(crate) allow_cuda: bool,
    pub(crate) allow_opencl: bool,

    /// Use at most this many devices across all backends.
    pub(crate) max_devices: Option<usize>,

    /// Allowlist items:
    /// - numeric indexes: "0","1"
    /// - substrings matched against device label/name/vendor (case-insensitive)
    pub(crate) allowlist: Vec<String>,
}

/// Batch sizing config used by the auto-tuner.
#[derive(Debug, Clone)]
pub(crate) struct GpuBatchConfig {
    /// Minimum batch size per launch (per device).
    pub(crate) min_batch: usize,
    /// Maximum batch size per launch (per device).
    pub(crate) max_batch: usize,
    /// VRAM usage ratio (0..=0.95) used for auto sizing.
    pub(crate) vram_ratio: f32,
    /// Batch builder timeout in ms (engine uses this).
    pub(crate) batch_timeout_ms: u32,
    /// Pipelining: max in-flight batches per device (engine uses this).
    pub(crate) inflight_batches: usize,
}

/// Result of device selection + computed batch sizes.
#[derive(Debug, Clone)]
pub(crate) struct GpuPlan {
    /// Active devices.
    pub(crate) devices: Vec<GpuPlannedDevice>,
    /// Total minimum batch across all active devices (used by engine reservations).
    pub(crate) min_batch_total: usize,
}

/// A selected device with computed batch sizing.
#[derive(Debug, Clone)]
pub(crate) struct GpuPlannedDevice {
    pub(crate) info: GpuDeviceInfo,
    /// Auto computed batch size for this device (clamped).
    pub(crate) batch_size: usize,
    /// Min batch for this device (clamped).
    pub(crate) min_batch: usize,
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

/// A very rough bytes/job estimate for the current pipeline.
///
/// For now we assume:
/// - challenge: 32 bytes
/// - output (y): 100 bytes
/// - witness: 100 bytes
/// - plus overhead/alignment
pub(crate) fn estimate_bytes_per_job() -> u64 {
    32 + 100 + 100 + 128
}

/// Compute a per-device batch size from VRAM and user limits.
///
/// This is intentionally conservative until the real GPU kernels and precise memory
/// accounting are in place.
pub(crate) fn auto_batch_size_for_device(
    dev: &GpuDeviceInfo,
    cfg: &GpuBatchConfig,
    bytes_per_job: u64,
) -> (usize, usize) {
    let bytes_per_job = bytes_per_job.max(1);

    let total_bytes: u64 = if dev.vram_total_bytes > 0 {
        dev.vram_total_bytes
    } else if let Some(v) = dev.total_mem_bytes {
        v
    } else {
        0
    };

    let max_from_mem: usize = if total_bytes == 0 {
        cfg.max_batch
    } else {
        let ratio = cfg.vram_ratio.clamp(0.05, 0.95) as f64;
        let usable = (total_bytes as f64 * ratio) as u64;
        usize::try_from(usable / bytes_per_job).unwrap_or(cfg.max_batch)
    };

    let mut batch_size = max_from_mem.clamp(cfg.min_batch.max(1), cfg.max_batch.max(1));
    if batch_size == 0 {
        batch_size = 1;
    }

    let min_batch = cfg.min_batch.min(batch_size).max(1);
    (batch_size, min_batch)
}
