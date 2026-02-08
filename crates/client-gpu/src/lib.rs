//! GPU backend facade.
//!
//! This crate is intentionally isolated from the CPU engine. The CPU engine should only
//! depend on `bbr-client-compute`, which may delegate to this crate for GPU selection.

pub mod auto;

/// GPU execution entrypoints.
///
/// This module intentionally contains only GPU-side logic. The CPU engine should remain
/// separate and only call into this crate via `bbr-client-compute`.
pub mod execute;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "opencl")]
pub mod opencl;

/// GPU backend kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA CUDA (driver API).
    Cuda,
    /// OpenCL (AMD/Intel/NVIDIA).
    OpenCl,
}

/// Result of a GPU probe.
#[derive(Debug, Clone)]
pub struct GpuProbe {
    /// Whether the backend appears usable.
    pub available: bool,
    /// Optional detail message.
    pub detail: Option<String>,
}

impl GpuProbe {
    /// Convenience constructor for an unavailable backend.
    pub fn unavailable(detail: impl Into<String>) -> Self {
        Self {
            available: false,
            detail: Some(detail.into()),
        }
    }

    /// Convenience constructor for an available backend.
    pub fn available(detail: Option<String>) -> Self {
        Self {
            available: true,
            detail,
        }
    }
}

/// GPU device selection policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuDeviceSelection {
    /// Use all usable devices detected by the backend.
    All,
    /// Disable GPU usage entirely.
    None,
    /// Use a specific list of device ordinals.
    List(Vec<u32>),
}

impl GpuDeviceSelection {
    /// Parse a device selection string.
    ///
    /// Accepted values:
    /// - `all`
    /// - `none`
    /// - comma-separated list of integers, e.g. `0,2,3`
    pub fn parse(input: &str) -> Option<Self> {
        let s = input.trim();
        if s.is_empty() {
            return None;
        }
        let lower = s.to_ascii_lowercase();
        if lower == "all" {
            return Some(Self::All);
        }
        if lower == "none" {
            return Some(Self::None);
        }

        let mut out: Vec<u32> = Vec::new();
        for part in s.split(',') {
            let p = part.trim();
            if p.is_empty() {
                continue;
            }
            let v: u32 = p.parse().ok()?;
            if !out.contains(&v) {
                out.push(v);
            }
        }
        if out.is_empty() {
            None
        } else {
            Some(Self::List(out))
        }
    }

    pub fn matches(&self, index: u32) -> bool {
        match self {
            Self::All => true,
            Self::None => false,
            Self::List(list) => list.contains(&index),
        }
    }

    /// Render selection to a compact string form.
    pub fn to_env_string(&self) -> String {
        match self {
            Self::All => "all".to_string(),
            Self::None => "none".to_string(),
            Self::List(list) => list.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
        }
    }
}

/// GPU runtime configuration (read from env or CLI).
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub devices: GpuDeviceSelection,
    pub streams: Option<u32>,
    pub batch_size: Option<u32>,
    pub mem_budget_bytes: Option<u64>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            devices: GpuDeviceSelection::All,
            streams: None,
            batch_size: None,
            mem_budget_bytes: None,
        }
    }
}

impl GpuConfig {
    /// Read GPU configuration from environment variables.
    ///
    /// Environment variables:
    /// - `BBR_GPU_DEVICES` (`all`, `none`, or `0,2,3`)
    /// - `BBR_GPU_DEVICE` (legacy single device; used only if `BBR_GPU_DEVICES` is not set)
    /// - `BBR_GPU_STREAMS`
    /// - `BBR_GPU_BATCH_SIZE`
    /// - `BBR_GPU_MEM_BUDGET` (bytes)
    pub fn from_env() -> Self {
        let devices = if let Ok(v) = std::env::var("BBR_GPU_DEVICES") {
            GpuDeviceSelection::parse(&v).unwrap_or(GpuDeviceSelection::All)
        } else if let Ok(v) = std::env::var("BBR_GPU_DEVICE") {
            match v.trim().parse::<u32>() {
                Ok(idx) => GpuDeviceSelection::List(vec![idx]),
                Err(_) => GpuDeviceSelection::All,
            }
        } else {
            GpuDeviceSelection::All
        };

        Self {
            devices,
            streams: parse_env_u32("BBR_GPU_STREAMS"),
            batch_size: parse_env_u32("BBR_GPU_BATCH_SIZE"),
            mem_budget_bytes: parse_env_u64("BBR_GPU_MEM_BUDGET"),
        }
    }

    /// Returns true if GPU usage is disabled by configuration.
    pub fn is_disabled(&self) -> bool {
        matches!(self.devices, GpuDeviceSelection::None)
    }
}

fn parse_env_u32(name: &str) -> Option<u32> {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<u32>().ok())
}

fn parse_env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok())
}
