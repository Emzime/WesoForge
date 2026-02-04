use clap::{Parser, ValueEnum};
use reqwest::Url;

#[cfg(feature = "prod-backend")]
const DEFAULT_BACKEND_URL: &str = "https://weso.forgeros.fr/";

#[cfg(not(feature = "prod-backend"))]
const DEFAULT_BACKEND_URL: &str = "http://127.0.0.1:8080";

fn default_backend_url() -> Url {
    Url::parse(DEFAULT_BACKEND_URL).expect("DEFAULT_BACKEND_URL must be a valid URL")
}

pub fn default_parallel_proofs() -> u16 {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(512) as u16
}

fn parse_mem_budget_bytes(input: &str) -> Result<u64, String> {
    let s = input.trim();
    if s.is_empty() {
        return Err("mem budget must not be empty".to_string());
    }

    let lower = s.to_ascii_lowercase();
    let (num, scale) = if let Some(raw) = lower.strip_suffix("kib") {
        (raw, 1024u64)
    } else if let Some(raw) = lower.strip_suffix("mib") {
        (raw, 1024u64 * 1024)
    } else if let Some(raw) = lower.strip_suffix("gib") {
        (raw, 1024u64 * 1024 * 1024)
    } else if let Some(raw) = lower.strip_suffix("kb") {
        (raw, 1000u64)
    } else if let Some(raw) = lower.strip_suffix("mb") {
        (raw, 1000u64 * 1000)
    } else if let Some(raw) = lower.strip_suffix("gb") {
        (raw, 1000u64 * 1000 * 1000)
    } else if let Some(raw) = lower.strip_suffix('b') {
        (raw, 1u64)
    } else {
        // Default unit is MiB to match typical user expectations (e.g. "128").
        (lower.as_str(), 1024u64 * 1024)
    };

    let num = num.trim();
    if num.is_empty() {
        return Err(format!("invalid mem budget: {input:?}"));
    }

    let value: u64 = num
        .parse()
        .map_err(|_| format!("invalid mem budget number: {input:?}"))?;

    value
        .checked_mul(scale)
        .ok_or_else(|| format!("mem budget too large: {input:?}"))
}

/// GPU backend selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum GpuBackendArg {
    /// Try CUDA first (NVIDIA), then OpenCL (AMD/NVIDIA), else disable.
    Auto,
    /// Force CUDA (NVIDIA only). If unavailable, GPU is disabled.
    Cuda,
    /// Force OpenCL (AMD/NVIDIA). If unavailable, GPU is disabled.
    Opencl,
    /// Disable GPU completely.
    Off,
}

fn default_gpu_backend() -> GpuBackendArg {
    GpuBackendArg::Auto
}

fn default_gpu_enabled() -> bool {
    true
}

fn default_cpu_pin_threads() -> bool {
    true
}

fn default_cpu_reserve_core0() -> bool {
    true
}

fn default_cpu_reverse_cores() -> bool {
    true
}

fn default_gpu_inflight_batches() -> u16 {
    2
}

fn default_gpu_batch_min() -> u32 {
    128
}

fn default_gpu_batch_max() -> u32 {
    4096
}

fn default_gpu_batch_timeout_ms() -> u32 {
    250
}

fn default_gpu_vram_ratio() -> f32 {
    0.60
}

/// Parse a comma-separated list into Vec<String>.
fn parse_csv_list(input: &str) -> Result<Vec<String>, String> {
    let items: Vec<String> = input
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();

    if items.is_empty() {
        return Err("list must not be empty".to_string());
    }
    Ok(items)
}

#[derive(Debug, Clone, Parser)]
#[command(name = "wesoforge", version, about = "WesoForge compact proof worker")]
pub struct Cli {
    #[arg(long, env = "BBR_BACKEND_URL", default_value_t = default_backend_url())]
    pub backend_url: Url,

    /// Number of proof workers to run in parallel (CPU workers).
    ///
    /// Note: GPU workers are configured separately and run concurrently.
    #[arg(
        short = 'p',
        long,
        env = "BBR_PARALLEL_PROOFS",
        default_value_t = default_parallel_proofs(),
        value_parser = clap::value_parser!(u16).range(1..=512)
    )]
    pub parallel: u16,

    #[arg(long, env = "BBR_NO_TUI", default_value_t = false)]
    pub no_tui: bool,

    /// Memory budget per worker for streaming proof generation (e.g. `128MB`).
    ///
    /// This is used by the `(k,l)` parameter tuner in the native prover.
    #[arg(
        short = 'm',
        long = "mem",
        env = "BBR_MEM_BUDGET",
        default_value = "128MB",
        value_parser = parse_mem_budget_bytes
    )]
    pub mem_budget_bytes: u64,

    // -----------------------------
    // CPU pinning / core policy
    // -----------------------------

    /// Enable CPU thread pinning (best effort on non-Windows/Linux).
    #[arg(long, env = "BBR_CPU_PIN", default_value_t = default_cpu_pin_threads())]
    pub cpu_pin_threads: bool,

    /// Reserve logical core 0 for the OS (never schedule CPU workers on core 0).
    #[arg(long, env = "BBR_CPU_RESERVE_CORE0", default_value_t = default_cpu_reserve_core0())]
    pub cpu_reserve_core0: bool,

    /// Assign CPU workers on cores in reverse order (last -> ... -> 1).
    #[arg(long, env = "BBR_CPU_REVERSE_CORES", default_value_t = default_cpu_reverse_cores())]
    pub cpu_reverse_cores: bool,

    /// Optional CPU core allowlist (comma-separated, supports ranges).
    ///
    /// Example: "2,3,6,7,10-15"
    #[arg(long = "cpu-cores", env = "BBR_CPU_CORES")]
    pub cpu_cores: Option<String>,

    /// Optional CPU core blocklist (comma-separated, supports ranges).
    ///
    /// Example: "0,1"
    #[arg(long = "cpu-cores-exclude", env = "BBR_CPU_CORES_EXCLUDE")]
    pub cpu_cores_exclude: Option<String>,

    // -----------------------------
    // GPU configuration
    // -----------------------------

    /// Enable GPU acceleration (CUDA/OpenCL). If false, CPU-only mode.
    #[arg(long, env = "BBR_GPU_ENABLED", default_value_t = default_gpu_enabled())]
    pub gpu_enabled: bool,

    /// GPU backend selection (prepares for GUI toggle later).
    #[arg(long, env = "BBR_GPU_BACKEND", default_value_t = default_gpu_backend(), value_enum)]
    pub gpu_backend: GpuBackendArg,

    /// Use at most N GPUs out of those installed/detected.
    ///
    /// Example: if you have 4 GPUs installed and set this to 2, only 2 will be used.
    #[arg(long, env = "BBR_GPU_MAX_DEVICES")]
    pub gpu_max_devices: Option<u16>,

    /// Optional GPU device allowlist (comma-separated).
    ///
    /// Items can be indexes ("0", "1") or substrings matched against device names ("4090", "7900").
    #[arg(long, env = "BBR_GPU_DEVICE_ALLOWLIST", value_parser = parse_csv_list)]
    pub gpu_device_allowlist: Option<Vec<String>>,

    /// Maximum number of in-flight batches per GPU device (pipelining).
    #[arg(long, env = "BBR_GPU_INFLIGHT_BATCHES", default_value_t = default_gpu_inflight_batches(), value_parser = clap::value_parser!(u16).range(1..=8))]
    pub gpu_inflight_batches: u16,

    /// Minimum batch size per GPU launch.
    #[arg(long, env = "BBR_GPU_BATCH_MIN", default_value_t = default_gpu_batch_min(), value_parser = clap::value_parser!(u32).range(1..=1_000_000))]
    pub gpu_batch_min: u32,

    /// Maximum batch size per GPU launch (auto-tuner will clamp to this).
    #[arg(long, env = "BBR_GPU_BATCH_MAX", default_value_t = default_gpu_batch_max(), value_parser = clap::value_parser!(u32).range(1..=1_000_000))]
    pub gpu_batch_max: u32,

    /// Batch builder timeout in milliseconds (launch when >= min batch, even if not full).
    #[arg(long, env = "BBR_GPU_BATCH_TIMEOUT_MS", default_value_t = default_gpu_batch_timeout_ms(), value_parser = clap::value_parser!(u32).range(1..=60_000))]
    pub gpu_batch_timeout_ms: u32,

    /// VRAM utilization ratio for auto batch sizing (0.0..=0.95 recommended).
    ///
    /// The auto-tuner will try to allocate buffers within vram_free * ratio.
    #[arg(long, env = "BBR_GPU_VRAM_RATIO", default_value_t = default_gpu_vram_ratio(), value_parser = clap::value_parser!(f32).range(0.0..=0.95))]
    pub gpu_vram_ratio: f32,

    /// Run a local benchmark (e.g. `--bench 0`) and exit.
    #[arg(long, value_name = "ALGO")]
    pub bench: Option<u32>,
}
