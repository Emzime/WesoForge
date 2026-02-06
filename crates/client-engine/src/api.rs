//! Public API types for the in-process `bbr-client` engine.

use std::sync::Arc;
use std::time::Duration;

use bbr_client_core::submitter::SubmitterConfig;
use reqwest::Url;
use serde::{Deserialize, Serialize};

/// GPU backend selection mode.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum GpuMode {
    /// Do not start any GPU workers.
    Off,
    /// Start GPU workers for all detected devices (subject to allow/deny lists).
    Auto,
    /// Start only CUDA GPU workers.
    Cuda,
    /// Start only OpenCL GPU workers.
    OpenCl,
}

impl Default for GpuMode {
    fn default() -> Self {
        Self::Off
    }
}

/// GPU configuration for the engine.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// GPU mode.
    pub mode: GpuMode,
    /// Number of worker tasks to spawn per detected GPU device.
    pub workers_per_device: usize,
    /// Allow-list of device keys (e.g. `cuda:0`). If non-empty, only listed devices are used.
    pub allow: Vec<String>,
    /// Deny-list of device keys.
    pub deny: Vec<String>,
    /// Device keys that should start disabled.
    pub start_disabled: Vec<String>,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            mode: GpuMode::Off,
            workers_per_device: 1,
            allow: Vec::new(),
            deny: Vec::new(),
            start_disabled: Vec::new(),
        }
    }
}

/// Worker type (CPU or a specific GPU device).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum WorkerKind {
    /// CPU prover.
    Cpu,
    /// GPU prover on a specific device.
    Gpu {
        /// Stable device key (e.g. `cuda:0`, `opencl:3`).
        device_key: String,
        /// Human readable device name.
        device_name: String,
    },
}

/// Configuration for the in-process engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Backend base URL (e.g. `http://127.0.0.1:8080`).
    pub backend_url: Url,

    /// Number of proof workers to run concurrently.
    pub parallel: usize,

    /// GPU worker configuration.
    pub gpu: GpuConfig,

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
    /// Worker kind (CPU / GPU).
    pub worker_kind: WorkerKind,
    /// Whether this worker is enabled.
    pub enabled: bool,
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
    /// Worker kind (CPU / GPU).
    pub worker_kind: WorkerKind,
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
    /// Worker enabled/disabled state changed.
    WorkerEnabled {
        /// Worker index.
        worker_idx: usize,
        /// Whether enabled.
        enabled: bool,
    },
    /// GPU device enabled/disabled state changed.
    GpuEnabled {
        /// Device key (e.g. `cuda:0`).
        device_key: String,
        /// Whether enabled.
        enabled: bool,
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
    pub(crate) inner: Arc<crate::engine::EngineInner>,
    pub(crate) join: tokio::task::JoinHandle<anyhow::Result<()>>,
}

/// Start a new in-process engine instance.
pub fn start_engine(config: EngineConfig) -> EngineHandle {
    crate::engine::start_engine(config)
}

impl EngineHandle {
    /// Subscribe to engine events.
    pub fn subscribe(&self) -> tokio::sync::broadcast::Receiver<EngineEvent> {
        self.inner.event_tx.subscribe()
    }

    /// Get the latest snapshot.
    pub fn latest_snapshot(&self) -> StatusSnapshot {
        self.inner.snapshot_rx.borrow().clone()
    }

    /// Request stop.
    pub fn request_stop(&self) {
        self.inner.request_stop();
    }

    /// Enable/disable a specific worker by index.
    pub fn set_worker_enabled(&self, worker_idx: usize, enabled: bool) {
        self.inner.set_worker_enabled(worker_idx, enabled);
    }

    /// Enable/disable all workers belonging to a given GPU device key.
    pub fn set_gpu_enabled(&self, device_key: &str, enabled: bool) {
        self.inner.set_gpu_enabled(device_key, enabled);
    }

    /// Wait for engine termination.
    pub async fn wait(self) -> anyhow::Result<()> {
        self.join.await??;
        Ok(())
    }
}
