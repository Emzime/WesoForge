use std::sync::{Arc, OnceLock, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::time::{Duration, Instant};

use anyhow::Context;
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine as _;
use chrono::Utc;
use reqwest::Url;
use tokio::sync::mpsc;

use bbr_client_core::submitter::SubmitterConfig;

use crate::api::{JobOutcome, JobSummary, WorkerStage};
use crate::gpu::GpuDeviceId;
use crate::backend::{BackendError, BackendJobDto, SubmitResponse, submit_job};
use crate::cpu_affinity::{CpuPinPolicy, pin_current_thread_with_lists};

const DISCRIMINANT_BITS: usize = 1024;


// Global, process-wide switch to enable/disable CPU thread pinning.
//
// The engine can configure this from EngineConfig.cpu_pin_threads.
static CPU_PINNING_ENABLED: AtomicBool = AtomicBool::new(true);

// CPU pin policy toggles (process-wide). Default matches previous behaviour.
// Bit 0: reserve_core0
// Bit 1: reversed
static CPU_PIN_POLICY_BITS: AtomicU8 = AtomicU8::new(0b11);

// Optional CPU core allow/block lists configured by the user.
// These are stored as parsed core IDs, and applied by the worker when pinning.
static CPU_CORE_ALLOWLIST: OnceLock<RwLock<Option<Vec<usize>>>> = OnceLock::new();
static CPU_CORE_BLOCKLIST: OnceLock<RwLock<Option<Vec<usize>>>> = OnceLock::new();

fn allowlist_cell() -> &'static RwLock<Option<Vec<usize>>> {
    CPU_CORE_ALLOWLIST.get_or_init(|| RwLock::new(None))
}

fn blocklist_cell() -> &'static RwLock<Option<Vec<usize>>> {
    CPU_CORE_BLOCKLIST.get_or_init(|| RwLock::new(None))
}

/// Enable/disable CPU thread pinning for CPU workers.
///
/// Notes:
/// - This is process-wide and should be configured once at startup.
/// - GPU workers are never pinned by `worker.rs`.
pub(crate) fn set_cpu_pinning_enabled(enabled: bool) {
    CPU_PINNING_ENABLED.store(enabled, Ordering::SeqCst);
}

/// Configure the CPU pinning policy (process-wide).
///
/// Notes:
/// - This should be called once at startup (from the engine or CLI).
/// - This affects only CPU workers (GPU workers are never pinned here).
pub(crate) fn set_cpu_pin_policy(reserve_core0: bool, reversed: bool) {
    let mut bits: u8 = 0;
    if reserve_core0 {
        bits |= 0b01;
    }
    if reversed {
        bits |= 0b10;
    }
    CPU_PIN_POLICY_BITS.store(bits, Ordering::SeqCst);
}

/// Set the optional CPU core allowlist (process-wide).
///
/// When set, only these cores are eligible for CPU workers.
pub(crate) fn set_cpu_core_allowlist(cores: Option<Vec<usize>>) {
    if let Ok(mut guard) = allowlist_cell().write() {
        *guard = cores;
    }
}

/// Set the optional CPU core blocklist (process-wide).
///
/// When set, these cores are excluded from CPU worker scheduling.
pub(crate) fn set_cpu_core_blocklist(cores: Option<Vec<usize>>) {
    if let Ok(mut guard) = blocklist_cell().write() {
        *guard = cores;
    }
}

fn current_cpu_pin_policy() -> CpuPinPolicy {
    let bits = CPU_PIN_POLICY_BITS.load(Ordering::SeqCst);
    CpuPinPolicy {
        reserve_core0: (bits & 0b01) != 0,
        reversed: (bits & 0b10) != 0,
    }
}

fn default_classgroup_element() -> [u8; 100] {
    let mut el = [0u8; 100];
    el[0] = 0x08;
    el
}

#[derive(Debug)]
struct SubmitFailure {
    message: String,
    drop_inflight: bool,
}

pub(crate) enum WorkerCommand {
    Job {
        worker_idx: usize,
        backend_url: Url,
        lease_id: String,
        lease_expires_at: i64,
        progress_steps: u64,
        job: BackendJobDto,
    },
    /// GPU batch execution command.
    ///
    /// Notes:
    /// - `worker_idx` should be unique across CPU + GPU workers (engine decides the numbering).
    /// - The current implementation computes the batch sequentially on CPU as a functional stub.
    ///   The CUDA/OpenCL backends will plug into this command in the next step.
    GpuBatch {
        worker_idx: usize,
        backend_url: Url,
        lease_id: String,
        lease_expires_at: i64,
        progress_steps: u64,
        jobs: Vec<BackendJobDto>,
        /// Structured GPU device identifier.
        device_id: GpuDeviceId,
        /// Human-readable device label (e.g. "CUDA:0 RTX 4090").
        device_label: String,
    },
    Stop,
}

pub(crate) enum WorkerInternalEvent {
    StageChanged { worker_idx: usize, stage: WorkerStage },
    Finished { outcome: JobOutcome },
    Warning { message: String },
    Error { message: String },
}

pub(crate) async fn run_worker_task(
    worker_idx: usize,
    mut rx: mpsc::Receiver<WorkerCommand>,
    internal_tx: mpsc::UnboundedSender<WorkerInternalEvent>,
    progress: Arc<AtomicU64>,
    http: reqwest::Client,
    submitter: Arc<tokio::sync::RwLock<SubmitterConfig>>,
    warned_invalid_reward_address: Arc<AtomicBool>,
) {
    // Best-effort CPU pinning.
    // This will be fully effective once the engine runs 1 dedicated OS thread per CPU worker.
    // On Windows/Linux, this pins the current thread. On macOS, it's a no-op.
    match pin_current_thread(worker_idx, CpuPinPolicy::default()) {
        Ok(Some(core_id)) => {
            let _ = internal_tx.send(WorkerInternalEvent::Warning {
                message: format!(
                    "info: cpu worker {} pinned to core {} (core 0 reserved, reverse order)",
                    worker_idx + 1,
                    core_id
                ),
            });
        }
        Ok(None) => {
            // No warning needed: could be unsupported OS or no core list available.
        }
        Err(err) => {
            let _ = internal_tx.send(WorkerInternalEvent::Warning {
                message: format!(
                    "warning: cpu worker {} pinning failed: {}",
                    worker_idx + 1,
                    err
                ),
            });
        }
    }

    while let Some(cmd) = rx.recv().await {
        match cmd {
            WorkerCommand::Stop => break,
            WorkerCommand::Job {
                worker_idx,
                backend_url,
                lease_id,
                lease_expires_at,
                progress_steps,
                job,
            } => {
if !cpu_pinned {
    cpu_pinned = true;

    if CPU_PINNING_ENABLED.load(Ordering::SeqCst) {
        let policy = current_cpu_pin_policy();

        // Snapshot allow/block lists (avoid holding locks across the pin call).
        let allowlist_snapshot: Option<Vec<usize>> =
            allowlist_cell().read().ok().and_then(|g| g.clone());
        let blocklist_snapshot: Option<Vec<usize>> =
            blocklist_cell().read().ok().and_then(|g| g.clone());

        let allow_ref = allowlist_snapshot.as_deref();
        let block_ref = blocklist_snapshot.as_deref();

        match pin_current_thread_with_lists(worker_idx, policy, allow_ref, block_ref) {
            Ok(Some(core_id)) => {
                let _ = internal_tx.send(WorkerInternalEvent::Warning {
                    message: format!(
                        "info: cpu worker {} pinned to core {}",
                        worker_idx + 1,
                        core_id
                    ),
                });
            }
            Ok(None) => {}
            Err(err) => {
                let _ = internal_tx.send(WorkerInternalEvent::Warning {
                    message: format!(
                        "warning: cpu worker {} pinning failed: {}",
                        worker_idx + 1,
                        err
                    ),
                });
            }
        }
    }
}

                let outcome = run_job(
                    worker_idx,
                    &internal_tx,
                    progress.clone(),
                    &http,
                    &submitter,
                    warned_invalid_reward_address.clone(),
                    backend_url,
                    lease_id,
                    lease_expires_at,
                    progress_steps,
                    job,
                )
                .await;
                let _ = internal_tx.send(WorkerInternalEvent::Finished { outcome });
            }
            WorkerCommand::GpuBatch {
                worker_idx,
                backend_url,
                lease_id,
                lease_expires_at,
                progress_steps,
                jobs,
                device_id,
                device_label,
            } => {
                let outcomes = run_gpu_batch(
                    worker_idx,
                    device_id,
                    &device_label,
                    &internal_tx,
                    progress.clone(),
                    &http,
                    &submitter,
                    warned_invalid_reward_address.clone(),
                    backend_url,
                    lease_id,
                    lease_expires_at,
                    progress_steps,
                    jobs,
                )
                .await;

                for outcome in outcomes {
                    let _ = internal_tx.send(WorkerInternalEvent::Finished { outcome });
                }
            }
        }
    }
}

async fn run_gpu_batch(
    worker_idx: usize,
    device_id: GpuDeviceId,
    device_label: &str,
    internal_tx: &mpsc::UnboundedSender<WorkerInternalEvent>,
    progress: Arc<AtomicU64>,
    http: &reqwest::Client,
    submitter: &tokio::sync::RwLock<SubmitterConfig>,
    warned_invalid_reward_address: Arc<AtomicBool>,
    backend_url: Url,
    lease_id: String,
    lease_expires_at: i64,
    progress_steps: u64,
    jobs: Vec<BackendJobDto>,
) -> Vec<JobOutcome> {
    // Comments in English as requested.
    //
    // Production "light guards" strategy:
    // - Always keep CPU path available (this project *adds* GPU, it does not remove CPU).
    // - GPU is opportunistic: if CUDA batch execution fails, we log and continue with CPU.
    // - We only do cheap validation (shape + a few samples) to avoid burning CPU time.

    let _ = internal_tx.send(WorkerInternalEvent::Warning {
        message: format!(
            "info: gpu worker {} received batch={} on {} (device_id={:?}:{})",
            worker_idx + 1,
            jobs.len(),
            device_label,
            device_id.backend,
            device_id.index
        ),
    });

    // Real GPU batch hook (scaffolding):
    // - Pack one u32 per job (first 4 bytes of decoded challenge, little-endian).
    // - Run CUDA add1 batch.
    // - Unpack only for diagnostics.
    //
    // This validates the end-to-end pipeline: pack -> H2D -> kernel -> D2H -> unpack.
    if device_id.backend == crate::gpu::GpuBackendKind::Cuda && !jobs.is_empty() {
        let mut packed: Vec<u32> = Vec::with_capacity(jobs.len());
        for job in &jobs {
            match B64.decode(job.challenge_b64.as_bytes()) {
                Ok(bytes) if bytes.len() >= 4 => {
                    packed.push(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]));
                }
                _ => packed.push(0),
            }
        }

        match crate::cuda_backend::add1_batch(device_id.index, &packed) {
            Ok(out) => {
                // Lightweight guards: shape + a few samples (constant-time-ish).
                if out.len() != packed.len() {
                    let _ = internal_tx.send(WorkerInternalEvent::Warning {
                        message: format!(
                            "warning: gpu worker {} CUDA batch shape mismatch on device {}: in={}, out={}",
                            worker_idx + 1,
                            device_id.index,
                            packed.len(),
                            out.len()
                        ),
                    });
                } else {
                    let sample_n = out.len().min(3);
                    let mut ok = true;
                    for i in 0..sample_n {
                        if out[i] != packed[i].wrapping_add(1) {
                            ok = false;
                            break;
                        }
                    }

                    if ok {
                        let mut samples = Vec::with_capacity(sample_n);
                        for i in 0..sample_n {
                            samples.push(format!("{}->{}", packed[i], out[i]));
                        }
                        let _ = internal_tx.send(WorkerInternalEvent::Warning {
                            message: format!(
                                "info: gpu worker {} CUDA batch OK on device {} (samples: {})",
                                worker_idx + 1,
                                device_id.index,
                                samples.join(", ")
                            ),
                        });
                    } else {
                        let _ = internal_tx.send(WorkerInternalEvent::Warning {
                            message: format!(
                                "warning: gpu worker {} CUDA batch sample mismatch on device {} (falling back to CPU path for VDF)",
                                worker_idx + 1,
                                device_id.index
                            ),
                        });
                    }
                }
            }
            Err(err) => {
                let _ = internal_tx.send(WorkerInternalEvent::Warning {
                    message: format!(
                        "warning: gpu worker {} CUDA batch failed on device {}: {} (continuing with CPU path)",
                        worker_idx + 1,
                        device_id.index,
                        err
                    ),
                });
            }
        }
    }

    // CPU remains the reference compute path for VDF until real VDF kernels exist.
    let mut outcomes = Vec::with_capacity(jobs.len());

    for job in jobs {
        // Reset progress per job to keep UI consistent.
        progress.store(0, Ordering::Relaxed);

        let outcome = run_job(
            worker_idx,
            internal_tx,
            progress.clone(),
            http,
            submitter,
            warned_invalid_reward_address.clone(),
            backend_url.clone(),
            lease_id.clone(),
            lease_expires_at,
            progress_steps,
            job,
        )
        .await;

        outcomes.push(outcome);
    }

    outcomes
}


async fn run_job(
    worker_idx: usize,
    internal_tx: &mpsc::UnboundedSender<WorkerInternalEvent>,
    progress: Arc<AtomicU64>,
    http: &reqwest::Client,
    submitter: &tokio::sync::RwLock<SubmitterConfig>,
    warned_invalid_reward_address: Arc<AtomicBool>,
    backend_url: Url,
    lease_id: String,
    lease_expires_at: i64,
    progress_steps: u64,
    job: BackendJobDto,
) -> JobOutcome {
    let started_at = Instant::now();

    let job_summary = JobSummary {
        job_id: job.job_id,
        height: job.height,
        field_vdf: job.field_vdf,
        number_of_iterations: job.number_of_iterations,
    };

    let output = match B64.decode(job.output_b64.as_bytes()) {
        Ok(v) => v,
        Err(err) => {
            return JobOutcome {
                worker_idx,
                job: job_summary,
                output_mismatch: false,
                submit_reason: None,
                submit_detail: None,
                drop_inflight: false,
                error: Some(format!("Error (bad output_b64: {err:#})")),
                compute_ms: 0,
                submit_ms: 0,
                total_ms: started_at.elapsed().as_millis() as u64,
            };
        }
    };
    let challenge = match B64.decode(job.challenge_b64.as_bytes()) {
        Ok(v) => v,
        Err(err) => {
            return JobOutcome {
                worker_idx,
                job: job_summary,
                output_mismatch: false,
                submit_reason: None,
                submit_detail: None,
                drop_inflight: false,
                error: Some(format!("Error (bad challenge_b64: {err:#})")),
                compute_ms: 0,
                submit_ms: 0,
                total_ms: started_at.elapsed().as_millis() as u64,
            };
        }
    };

    let _ = internal_tx.send(WorkerInternalEvent::StageChanged {
        worker_idx,
        stage: WorkerStage::Computing,
    });

    let compute_started_at = Instant::now();
    let (witness, output_mismatch) = match compute_witness(
        worker_idx,
        internal_tx,
        progress.clone(),
        job.number_of_iterations,
        progress_steps,
        challenge,
        output.clone(),
    )
    .await
    {
        Ok(v) => v,
        Err(status) => {
            return JobOutcome {
                worker_idx,
                job: job_summary,
                output_mismatch: false,
                submit_reason: None,
                submit_detail: None,
                drop_inflight: false,
                error: Some(status),
                compute_ms: compute_started_at.elapsed().as_millis() as u64,
                submit_ms: 0,
                total_ms: started_at.elapsed().as_millis() as u64,
            };
        }
    };
    let compute_ms = compute_started_at.elapsed().as_millis() as u64;

    let _ = internal_tx.send(WorkerInternalEvent::StageChanged {
        worker_idx,
        stage: WorkerStage::Submitting,
    });

    let submit_started_at = Instant::now();
    let submit_res = submit_witness(
        http,
        submitter,
        warned_invalid_reward_address,
        internal_tx,
        &backend_url,
        job.job_id,
        &lease_id,
        lease_expires_at,
        &witness,
    )
    .await;
    let submit_ms = submit_started_at.elapsed().as_millis() as u64;

    match submit_res {
        Ok(res) => JobOutcome {
            worker_idx,
            job: job_summary,
            output_mismatch,
            submit_reason: Some(res.reason),
            submit_detail: Some(res.detail),
            drop_inflight: false,
            error: None,
            compute_ms,
            submit_ms,
            total_ms: started_at.elapsed().as_millis() as u64,
        },
        Err(err) => JobOutcome {
            worker_idx,
            job: job_summary,
            output_mismatch,
            submit_reason: None,
            submit_detail: None,
            drop_inflight: err.drop_inflight,
            error: Some(err.message),
            compute_ms,
            submit_ms,
            total_ms: started_at.elapsed().as_millis() as u64,
        },
    }
}

pub(crate) async fn compute_witness(
    worker_idx: usize,
    internal_tx: &mpsc::UnboundedSender<WorkerInternalEvent>,
    progress: Arc<AtomicU64>,
    total_iters: u64,
    progress_steps: u64,
    challenge: Vec<u8>,
    output: Vec<u8>,
) -> Result<(Vec<u8>, bool), String> {
    let mut last_compute_err: Option<String> = None;
    let mut last_log_at = Instant::now()
        .checked_sub(Duration::from_secs(3600))
        .unwrap_or_else(Instant::now);
    let mut attempts: u32 = 0;

    loop {
        let total_iters = total_iters.max(1);
        let progress_interval = progress_interval(total_iters, progress_steps);
        let challenge = challenge.clone();
        let output = output.clone();
        let progress_clone = progress.clone();

        let compute = tokio::task::spawn_blocking(move || -> anyhow::Result<(Vec<u8>, bool)> {
            let x = default_classgroup_element();
            let out = if progress_steps == 0 {
                bbr_client_chiavdf_fast::prove_one_weso_fast_streaming(
                    &challenge,
                    &x,
                    &output,
                    DISCRIMINANT_BITS,
                    total_iters,
                )
                .context("chiavdf prove_one_weso_fast_streaming")?
            } else {
                let progress_for_cb = progress_clone.clone();
                bbr_client_chiavdf_fast::prove_one_weso_fast_streaming_with_progress(
                    &challenge,
                    &x,
                    &output,
                    DISCRIMINANT_BITS,
                    total_iters,
                    progress_interval,
                    move |iters_done| {
                        progress_for_cb.store(iters_done, Ordering::Relaxed);
                    },
                )
                .context("chiavdf prove_one_weso_fast_streaming_with_progress")?
            };

            progress_clone.store(total_iters, Ordering::Relaxed);

            let half = out.len() / 2;
            let y = &out[..half];
            let witness = out[half..].to_vec();
            Ok((witness, y != output))
        })
        .await;

        match compute {
            Ok(Ok((witness, output_mismatch))) => return Ok((witness, output_mismatch)),
            Ok(Err(err)) => {
                attempts = attempts.saturating_add(1);
                let err_msg = format!("{err:#}");
                let should_log = last_compute_err.as_deref() != Some(&err_msg)
                    || last_log_at.elapsed() >= Duration::from_secs(30);
                if should_log {
                    last_compute_err = Some(err_msg.clone());
                    last_log_at = Instant::now();
                    let _ = internal_tx.send(WorkerInternalEvent::Error {
                        message: format!(
                            "error: worker {} compute failed (attempt {}): {}; retrying in 5s",
                            worker_idx + 1,
                            attempts,
                            err_msg
                        ),
                    });
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
            Err(err) => {
                attempts = attempts.saturating_add(1);
                let err_msg = format!("{err:#}");
                let should_log = last_compute_err.as_deref() != Some(&err_msg)
                    || last_log_at.elapsed() >= Duration::from_secs(30);
                if should_log {
                    last_compute_err = Some(err_msg.clone());
                    last_log_at = Instant::now();
                    let _ = internal_tx.send(WorkerInternalEvent::Error {
                        message: format!(
                            "error: worker {} compute join failed (attempt {}): {}; retrying in 5s",
                            worker_idx + 1,
                            attempts,
                            err_msg
                        ),
                    });
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        };
    }
}

fn progress_interval(total_iters: u64, progress_steps: u64) -> u64 {
    if progress_steps == 0 {
        return 0;
    }
    if total_iters == 0 {
        return 1;
    }
    (total_iters.saturating_add(progress_steps - 1) / progress_steps).max(1)
}

async fn submit_witness(
    http: &reqwest::Client,
    submitter: &tokio::sync::RwLock<SubmitterConfig>,
    warned_invalid_reward_address: Arc<AtomicBool>,
    internal_tx: &mpsc::UnboundedSender<WorkerInternalEvent>,
    backend: &Url,
    job_id: u64,
    lease_id: &str,
    lease_expires_at: i64,
    witness: &[u8],
) -> Result<SubmitResponse, SubmitFailure> {
    let mut last_submit_err: Option<String> = None;
    let mut attempts: u32 = 0;
    let mut last_log_at =
        Instant::now().checked_sub(Duration::from_secs(3600)).unwrap_or_else(Instant::now);

    loop {
        let now = Utc::now().timestamp();

        let (reward_address, name) = {
            let cfg = submitter.read().await;
            (cfg.reward_address.clone(), cfg.name.clone())
        };

        match submit_job(
            http,
            backend,
            job_id,
            lease_id,
            witness,
            reward_address.as_deref(),
            name.as_deref(),
        )
        .await
        {
            Ok(res) => return Ok(res),
            Err(err) => {
                attempts = attempts.saturating_add(1);
                if matches!(
                    err.downcast_ref::<BackendError>(),
                    Some(BackendError::LeaseInvalid)
                ) {
                    let _ = internal_tx.send(WorkerInternalEvent::Error {
                        message: format!(
                            "error: submit rejected for job {job_id}: lease invalid/expired"
                        ),
                    });
                    return Err(SubmitFailure {
                        message: "Error (lease invalid/expired)".to_string(),
                        drop_inflight: true,
                    });
                }
                if matches!(
                    err.downcast_ref::<BackendError>(),
                    Some(BackendError::LeaseConflict)
                ) {
                    let _ = internal_tx.send(WorkerInternalEvent::Error {
                        message: format!(
                            "error: submit rejected for job {job_id}: lease conflict (already leased by someone else)"
                        ),
                    });
                    return Err(SubmitFailure {
                        message: "Error (lease conflict)".to_string(),
                        drop_inflight: true,
                    });
                }
                if matches!(
                    err.downcast_ref::<BackendError>(),
                    Some(BackendError::JobNotFound)
                ) {
                    let _ = internal_tx.send(WorkerInternalEvent::Error {
                        message: format!("error: submit rejected for job {job_id}: job not found"),
                    });
                    return Err(SubmitFailure {
                        message: "Error (job not found)".to_string(),
                        drop_inflight: true,
                    });
                }
                if matches!(
                    err.downcast_ref::<BackendError>(),
                    Some(BackendError::InvalidRewardAddress)
                ) && reward_address.is_some()
                {
                    {
                        let mut cfg = submitter.write().await;
                        cfg.reward_address = None;
                    }

                    if !warned_invalid_reward_address.swap(true, Ordering::SeqCst) {
                        let _ = internal_tx.send(WorkerInternalEvent::Warning {
                            message: "warning: backend rejected configured reward address; submitting without reward metadata"
                                .to_string(),
                        });
                    }

                    continue;
                }

                let err_msg = format!("{err:#}");
                let should_log = last_submit_err.as_deref() != Some(&err_msg)
                    || last_log_at.elapsed() >= Duration::from_secs(30);
                if should_log {
                    last_submit_err = Some(err_msg.clone());
                    last_log_at = Instant::now();
                    let expires_in = (lease_expires_at - now).max(0);
                    let _ = internal_tx.send(WorkerInternalEvent::Error {
                        message: format!(
                            "error: submit failed for job {job_id} (attempt {attempts}, lease expires in {expires_in}s): {err_msg}; retrying in 5s"
                        ),
                    });
                }
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
        }
    }
}
