//! Public API types for the in-process `bbr-client` engine.

use std::time::Duration;

use bbr_client_core::submitter::SubmitterConfig;
use reqwest::Url;
use serde::{Deserialize, Serialize};

/// GPU backend selection (CUDA/OpenCL).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum GpuBackend {
    /// Try CUDA first (NVIDIA), then OpenCL (AMD/NVIDIA), else disable.
    Auto,
    /// Force CUDA (NVIDIA only). If unavailable, GPU will be effectively disabled.
    Cuda,
    /// Force OpenCL (AMD/NVIDIA). If unavailable, GPU will be effectively disabled.
    Opencl,
    /// Disable GPU completely.
    Off,
}

/// Configuration for the in-process engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Backend base URL (e.g. `http://127.0.0.1:8080`).
    pub backend_url: Url,

    /// Number of proof workers to run concurrently (CPU workers).
    ///
    /// GPU workers are configured separately and run concurrently.
    pub parallel: usize,

    /// Memory budget (bytes) for the native streaming prover parameter tuner.
    ///
    /// Note: the chiavdf fast wrapper currently treats this as a *process-wide*
    /// setting, so all workers share the same configured budget.
    pub mem_budget_bytes: u64,

    /// Submitter metadata attached to job submissions.
    pub submitter: SubmitterConfig,

    /// How long to sleep after an empty work fetch / error.
    pub idle_sleep: Duration,

    /// Target number of progress updates per job.
    ///
    /// This is used to derive the chiavdf progress callback cadence
    /// (`progress_interval`).
    pub progress_steps: u64,

    /// How often the engine samples worker progress to emit progress events.
    pub progress_tick: Duration,

    /// Maximum number of completed jobs retained in the snapshot.
    pub recent_jobs_max: usize,

    // -----------------------------
    // CPU scheduling / pinning policy
    // -----------------------------

    /// Enable CPU thread pinning (best effort on non-Windows/Linux).
    pub cpu_pin_threads: bool,

    /// Reserve logical core 0 for the OS (never schedule CPU workers on core 0).
    pub cpu_reserve_core0: bool,

    /// Assign CPU workers on cores in reverse order (last -> ... -> 1).
    pub cpu_reverse_cores: bool,
    /// Optional CPU core allowlist (e.g. \"2,3,6,7,10-15\").
    ///
    /// When set, only those logical cores are used for CPU workers.
    /// Note: this has priority over `cpu_reserve_core0`.
    pub cpu_core_allowlist: Option<String>,

    /// Optional CPU core blocklist (e.g. \"0,1\").
    ///
    /// When set, those logical cores are excluded (even if present in allowlist).
    pub cpu_core_blocklist: Option<String>,
    // -----------------------------
    // GPU orchestration (prepared for GUI)
    // -----------------------------

    /// Enable GPU acceleration. If false, CPU-only mode.
    pub gpu_enabled: bool,

    /// Selected GPU backend strategy.
    pub gpu_backend: GpuBackend,

    /// Use at most N GPUs out of those installed/detected.
    pub gpu_max_devices: Option<usize>,

    /// Optional allowlist (indexes like "0","1" or name substrings like "4090","7900").
    pub gpu_device_allowlist: Vec<String>,

    /// Max number of in-flight batches per GPU device (pipelining).
    pub gpu_inflight_batches: usize,

    /// Minimum batch size per GPU launch.
    pub gpu_batch_min: usize,

    /// Maximum batch size per GPU launch (auto-tuner clamps to this).
    pub gpu_batch_max: usize,

    /// Batch builder timeout in milliseconds (launch when >= min batch even if not full).
    pub gpu_batch_timeout_ms: u32,

    /// VRAM utilization ratio for auto batch sizing.
    pub gpu_vram_ratio: f32,
}

impl EngineConfig {
    /// Default idle backoff used by the CLI worker.
    pub const DEFAULT_IDLE_SLEEP: Duration = Duration::from_secs(10);

    /// Default number of progress steps (matches the current CLI progress bars).
    pub const DEFAULT_PROGRESS_STEPS: u64 = 20;

    /// Default progress sampling tick.
    pub const DEFAULT_PROGRESS_TICK: Duration = Duration::from_millis(200);

    /// Default size of the recent-jobs ring buffer.
    pub const DEFAULT_RECENT_JOBS_MAX: usize = 100;
}

/// A lightweight summary of a leased proof job.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobSummary {
    /// Backend job identifier.
    pub job_id: u64,
    /// Block height.
    pub height: u32,
    /// Compressible VDF field identifier (1..=4).
    pub field_vdf: i32,
    /// VDF iteration count.
    pub number_of_iterations: u64,
}

/// Stage of a worker in the job lifecycle.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WorkerStage {
    /// No job assigned (idle).
    Idle,
    /// Computing the proof witness.
    Computing,
    /// Submitting the witness to the backend.
    Submitting,
}

/// Snapshot of a single worker’s current state.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerSnapshot {
    /// Worker index (0-based).
    pub worker_idx: usize,
    /// Current stage.
    pub stage: WorkerStage,
    /// Current job, if any.
    pub job: Option<JobSummary>,
    /// Iterations completed for the current job.
    pub iters_done: u64,
    /// Total iterations for the current job.
    pub iters_total: u64,
    /// Estimated speed in iterations/second.
    pub iters_per_sec: u64,
}

/// Result of a completed job (submitted or failed).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JobOutcome {
    /// Worker index (0-based).
    pub worker_idx: usize,
    /// Job metadata.
    pub job: JobSummary,
    /// Whether the computed output mismatched the expected `y_ref`.
    pub output_mismatch: bool,
    /// Backend submission reason (e.g. `accepted`, `already_compact`), if submission happened.
    pub submit_reason: Option<String>,
    /// Backend submission detail string, if submission happened.
    pub submit_detail: Option<String>,
    /// Remove this job from the local in-flight store (resume file) even on failure.
    ///
    /// This is used for terminal submission rejections where retrying would be useless
    /// (e.g. `job_not_found`, lease conflicts).
    #[serde(default)]
    pub drop_inflight: bool,
    /// Human-readable failure message, for compute/submit errors.
    pub error: Option<String>,
    /// Total compute time (milliseconds).
    pub compute_ms: u64,
    /// Total submission time (milliseconds).
    pub submit_ms: u64,
    /// Total job time (milliseconds).
    pub total_ms: u64,
}

/// Engine event stream payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type")]
pub enum EngineEvent {
    /// Engine started.
    Started,
    /// Engine is stopping (graceful shutdown requested).
    StopRequested,
    /// Worker has been assigned a new job.
    WorkerJobStarted {
        /// Worker index (0-based).
        worker_idx: usize,
        /// Job summary.
        job: JobSummary,
    },
    /// Worker progress update.
    WorkerProgress {
        /// Worker index (0-based).
        worker_idx: usize,
        /// Iterations completed.
        iters_done: u64,
        /// Iterations total.
        iters_total: u64,
        /// Speed estimate in iterations/second.
        iters_per_sec: u64,
    },
    /// Worker stage transition.
    WorkerStage {
        /// Worker index (0-based).
        worker_idx: usize,
        /// New stage.
        stage: WorkerStage,
    },
    /// Global speed update (sum of all workers), used by CLI/TUI.
    Speed {
        /// Total speed (iterations/second).
        iters_per_sec: u64,
        /// Number of workers currently busy.
        busy: usize,
        /// Total number of workers.
        total: usize,
    },
    /// Worker completed a job (success or failure).
    JobFinished {
        /// Job outcome.
        outcome: JobOutcome,
    },
    /// A warning from the engine.
    Warning {
        /// Warning message.
        message: String,
    },
    /// A non-fatal error from the engine.
    Error {
        /// Error message.
        message: String,
    },
    /// Engine stopped (no more workers running).
    Stopped,
}

/// Current engine state snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StatusSnapshot {
    /// Whether the engine has been asked to stop.
    pub stop_requested: bool,
    /// Per-worker snapshots.
    pub workers: Vec<WorkerSnapshot>,
    /// Recently completed jobs (newest last).
    pub recent_jobs: Vec<JobOutcome>,
}

/// Handle to a running in-process engine instance.
pub struct EngineHandle {
    pub(crate) inner: std::sync::Arc<crate::engine::EngineInner>,
    pub(crate) join: tokio::task::JoinHandle<anyhow::Result<()>>,
}


impl EngineHandle {
    /// Subscribe to engine events.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<EngineEvent> {
        self.inner.event_tx.subscribe()
    }

    /// Request a graceful stop of the engine.
    pub fn request_stop(&self) {
        self.inner.request_stop();
    }

    /// Get a receiver that yields status snapshots.
    pub fn snapshot_receiver(&self) -> tokio::sync::watch::Receiver<StatusSnapshot> {
        self.inner.snapshot_rx.clone()
    }


    /// Get the latest snapshot (cloned from the watch receiver).
    pub fn snapshot(&self) -> StatusSnapshot {
        self.inner.snapshot_rx.borrow().clone()
    }
}
/// Start a new in-process engine instance.
pub fn start_engine(config: EngineConfig) -> EngineHandle {
    crate::engine::start_engine(config)
}
