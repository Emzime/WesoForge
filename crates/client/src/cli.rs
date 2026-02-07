use clap::Parser;
use clap::ValueEnum;
use reqwest::Url;

fn parse_csv_list(input: &str) -> Result<Vec<String>, String> {
    let items = input
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    Ok(items)
}

#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum GpuModeCli {
    Off,
    Auto,
    Cuda,
    Opencl,
}

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

#[derive(Debug, Clone, Parser)]
#[command(name = "wesoforge", version, about = "WesoForge compact proof worker")]
pub struct Cli {
    #[arg(long, env = "BBR_BACKEND_URL", default_value_t = default_backend_url())]
    pub backend_url: Url,

    /// Number of CPU proof workers to run in parallel.
    ///
    /// Use 0 to disable CPU workers entirely (GPU-only run).
    #[arg(
        long = "cpu-workers",
        env = "BBR_CPU_WORKERS",
        default_value_t = default_parallel_proofs(),
        value_parser = clap::value_parser!(u16).range(0..=512)
    )]
    pub cpu_workers: u16,

    /// Backward-compatible alias for cpu workers.
    ///
    /// If provided, it overrides --cpu-workers.
    #[arg(
        short = 'p',
        long,
        env = "BBR_PARALLEL_PROOFS",
        value_parser = clap::value_parser!(u16).range(0..=512)
    )]
    pub parallel: Option<u16>,

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

    /// GPU worker mode.
    #[arg(long, env = "WESOFORGE_GPU_MODE", value_enum, default_value_t = GpuModeCli::Off)]
    pub gpu_mode: GpuModeCli,

    /// GPU workers per detected device.
    #[arg(
        long,
        env = "WESOFORGE_GPU_WORKERS_PER_DEVICE",
        default_value_t = 1,
        value_parser = clap::value_parser!(u16).range(1..=64)
    )]
    pub gpu_workers_per_device: u16,

    /// Comma-separated allow list of GPU device keys (e.g. `cuda:0,opencl:1`). If set, only these devices are used.
    #[arg(long, env = "WESOFORGE_GPU_ALLOW", value_parser = parse_csv_list)]
    pub gpu_allow: Option<Vec<String>>,

    /// Comma-separated deny list of GPU device keys.
    #[arg(long, env = "WESOFORGE_GPU_DENY", value_parser = parse_csv_list)]
    pub gpu_deny: Option<Vec<String>>,

    /// Comma-separated list of device keys that should start disabled.
    #[arg(long, env = "WESOFORGE_GPU_START_DISABLED", value_parser = parse_csv_list)]
    pub gpu_start_disabled: Option<Vec<String>>,

    /// Run a local benchmark (e.g. `--bench 0`) and exit.
    #[arg(long, value_name = "ALGO")]
    pub bench: Option<u32>,
}

impl Cli {
    pub fn effective_cpu_workers(&self) -> u16 {
        self.parallel.unwrap_or(self.cpu_workers)
    }
}
