use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use chrono::Utc;
use tokio::sync::{broadcast, mpsc, watch};

use crate::api::{
    EngineConfig, EngineEvent, EngineHandle, JobOutcome, JobSummary, StatusSnapshot, WorkerSnapshot,
    WorkerStage,
};
\1use crate::cpu_affinity::parse_core_list;
use crate::inflight::InflightStore;
use crate::worker::{WorkerCommand, WorkerInternalEvent};

pub(crate) struct EngineInner {
    pub(crate) event_tx: broadcast::Sender<EngineEvent>,
    pub(crate) snapshot_rx: watch::Receiver<StatusSnapshot>,
    stop_requested: AtomicBool,
    notify: tokio::sync::Notify,
}

impl EngineInner {
    pub(crate) fn request_stop(&self) {
        if !self.stop_requested.swap(true, Ordering::SeqCst) {
            let _ = self.event_tx.send(EngineEvent::StopRequested);
            self.notify.notify_waiters();
        }
    }

    fn should_stop(&self) -> bool {
        self.stop_requested.load(Ordering::SeqCst)
    }
}

#[derive(Debug)]
struct WorkJobItem {
    lease_id: String,
    lease_expires_at: i64,
    job: BackendJobDto,
}

#[derive(Debug)]
struct WorkerRuntime {
    stage: WorkerStage,
    job: Option<JobSummary>,
    iters_total: u64,
    last_speed_sample_at: Option<Instant>,
    prev_speed_interval: Option<(u64, Duration)>,
    speed_its_per_sec: u64,
    last_reported_iters_done: u64,
    last_emitted_iters_done: u64,
}

impl WorkerRuntime {
    fn new() -> Self {
        Self {
            stage: WorkerStage::Idle,
            job: None,
            iters_total: 0,
            last_speed_sample_at: None,
            prev_speed_interval: None,
            speed_its_per_sec: 0,
            last_reported_iters_done: 0,
            last_emitted_iters_done: 0,
        }
    }

    fn is_idle(&self) -> bool {
        self.stage == WorkerStage::Idle
    }

    fn is_busy(&self) -> bool {
        self.stage != WorkerStage::Idle
    }

    fn start_job(&mut self, job: JobSummary) {
        self.stage = WorkerStage::Computing;
        self.job = Some(job.clone());
        self.iters_total = job.number_of_iterations;
        self.last_speed_sample_at = Some(Instant::now());
        self.prev_speed_interval = None;
        self.speed_its_per_sec = 0;
        self.last_reported_iters_done = 0;
        self.last_emitted_iters_done = 0;
    }

    fn set_stage(&mut self, stage: WorkerStage) {
        self.stage = stage;
    }

    fn finish_job(&mut self) {
        self.stage = WorkerStage::Idle;
        self.job = None;
        self.iters_total = 0;
        self.last_speed_sample_at = None;
        self.prev_speed_interval = None;
        self.speed_its_per_sec = 0;
        self.last_reported_iters_done = 0;
        self.last_emitted_iters_done = 0;
    }

    fn apply_progress(&mut self, iters_done: u64) -> Option<u64> {
        let total_iters = self.iters_total;
        if total_iters == 0 {
            return None;
        }

        let now = Instant::now();
        let iters_done = iters_done.min(total_iters);
        let delta = iters_done.saturating_sub(self.last_reported_iters_done);
        if delta == 0 {
            return None;
        }

        if let Some(last_at) = self.last_speed_sample_at {
            let dt = now.duration_since(last_at);
            let (sum_iters, sum_dt) = if let Some((prev_iters, prev_dt)) = self.prev_speed_interval {
                (prev_iters.saturating_add(delta), prev_dt + dt)
            } else {
                (delta, dt)
            };

            if sum_dt.as_secs_f64() > 0.0 {
                self.speed_its_per_sec = (sum_iters as f64 / sum_dt.as_secs_f64()).round() as u64;
            }

            self.prev_speed_interval = Some((sum_iters, sum_dt));
        }

        self.last_speed_sample_at = Some(now);
        self.last_reported_iters_done = iters_done;
        Some(iters_done)
    }
}

struct EngineRuntime {
    http: reqwest::Client,
    cfg: EngineConfig,

    workers: Vec<WorkerRuntime>,
    worker_cmds: Vec<mpsc::Sender<WorkerCommand>>,
    worker_progress: Vec<Arc<std::sync::atomic::AtomicU64>>,
    internal_rx: mpsc::UnboundedReceiver<WorkerInternalEvent>,

    // Dedicated OS threads for CPU workers (each runs a current-thread Tokio runtime).
    worker_threads: Vec<std::thread::JoinHandle<()>>,

    pending: VecDeque<WorkJobItem>,
    fetch_task: Option<tokio::task::JoinHandle<anyhow::Result<Vec<WorkJobItem>>>>,
    fetch_backoff: Option<Pin<Box<tokio::time::Sleep>>>,
    inflight: Option<InflightStore>,

    recent_jobs: VecDeque<JobOutcome>,
    snapshot_tx: watch::Sender<StatusSnapshot>,
    inner: Arc<EngineInner>,

    // Global speed emission
    last_speed_emitted: u64,
}

impl EngineRuntime {
    fn build_snapshot(&self) -> StatusSnapshot {
        let workers = self
            .workers
            .iter()
            .enumerate()
            .map(|(idx, w)| WorkerSnapshot {
                worker_idx: idx,
                stage: w.stage,
                job: w.job.clone(),
                iters_done: self
                    .worker_progress
                    .get(idx)
                    .map(|a| a.load(std::sync::atomic::Ordering::Relaxed))
                    .unwrap_or(0),
                iters_total: w.iters_total,
                iters_per_sec: w.speed_its_per_sec,
            })
            .collect();

        StatusSnapshot {
            stop_requested: self.inner.should_stop(),
            workers,
            recent_jobs: self.recent_jobs.iter().cloned().collect(),
        }
    }

    fn push_snapshot(&self) {
        let snap = self.build_snapshot();
        let _ = self.snapshot_tx.send(snap);
    }

    fn emit(&self, event: EngineEvent) {
        let _ = self.inner.event_tx.send(event);
    }

    fn idle_count(&self) -> usize {
        self.workers.iter().filter(|w| w.is_idle()).count()
    }

    fn all_idle(&self) -> bool {
        !self.workers.iter().any(|w| w.is_busy())
    }

    fn maybe_warn_gpu_not_ready(&self) {
        if !self.cfg.gpu_enabled {
            return;
        }
        if self.cfg.gpu_backend == crate::api::GpuBackend::Off {
            return;
        }

        // GPU orchestration is prepared (config + plumbing),
        // but compute/dispatch is implemented in the next patch.
        self.emit(EngineEvent::Warning {
            message: "warning: GPU is enabled in config but GPU workers are not wired yet in this build; running CPU-only (CUDA/OpenCL batch worker will be added next).".to_string(),
        });
    }

    fn maybe_start_fetch(&mut self) {
        if self.inner.should_stop() {
            return;
        }

        let cpu_idle = self.idle_count();
        if cpu_idle == 0 {
            return;
        }

        if !self.pending.is_empty() || self.fetch_task.is_some() || self.fetch_backoff.is_some() {
            return;
        }

        let http = self.http.clone();
        let backend = self.cfg.backend_url.clone();

        // For now, fetch based on CPU needs only.
        // Next patch: include GPU batch demand in count.
        let count = cpu_idle;

        self.fetch_task = Some(tokio::spawn(async move {
            let count = count.min(u32::MAX as usize) as u32;
            let batch: BackendWorkBatch = fetch_work(&http, &backend, count).await?;
            let items = batch
                .jobs
                .into_iter()
                .map(|job| WorkJobItem {
                    lease_id: batch.lease_id.clone(),
                    lease_expires_at: batch.lease_expires_at,
                    job,
                })
                .collect();
            Ok(items)
        }));
    }

    async fn assign_jobs_cpu(&mut self) -> anyhow::Result<()> {
        if self.inner.should_stop() {
            self.pending.clear();
            return Ok(());
        }

        let mut snapshot_dirty = false;

        for idx in 0..self.workers.len() {
            if !self.workers[idx].is_idle() {
                continue;
            }

            let Some(item) = self.pending.pop_front() else {
                break;
            };

            let job_summary = JobSummary {
                job_id: item.job.job_id,
                height: item.job.height,
                field_vdf: item.job.field_vdf,
                number_of_iterations: item.job.number_of_iterations,
            };

            {
                let worker = &mut self.workers[idx];
                worker.start_job(job_summary.clone());
            }

            self.worker_progress
                .get(idx)
                .map(|p| p.store(0, std::sync::atomic::Ordering::Relaxed));

            let _ = self.worker_cmds[idx]
                .send(WorkerCommand::Job {
                    worker_idx: idx,
                    backend_url: self.cfg.backend_url.clone(),
                    lease_id: item.lease_id,
                    lease_expires_at: item.lease_expires_at,
                    progress_steps: self.cfg.progress_steps,
                    job: item.job,
                })
                .await;

            self.emit(EngineEvent::WorkerJobStarted {
                worker_idx: idx,
                job: job_summary,
            });

            snapshot_dirty = true;
        }

        if snapshot_dirty {
            self.push_snapshot();
        }

        Ok(())
    }

    async fn shutdown_workers(&mut self) {
        for tx in &self.worker_cmds {
            let _ = tx.send(WorkerCommand::Stop).await;
        }

        let mut handles = Vec::new();
        std::mem::swap(&mut handles, &mut self.worker_threads);

        // Join threads without blocking the async reactor.
        let join_res = tokio::task::spawn_blocking(move || {
            for h in handles {
                let _ = h.join();
            }
        })
        .await;

        if let Err(err) = join_res {
            self.emit(EngineEvent::Warning {
                message: format!("warning: join cpu worker threads task failed: {err:#}"),
            });
        }
    }

    fn update_speeds_and_emit(&mut self) {
        let mut total_speed: u64 = 0;
        let mut busy: usize = 0;

        for (idx, w) in self.workers.iter_mut().enumerate() {
            if w.is_busy() {
                busy += 1;
            }

            if w.iters_total == 0 {
                continue;
            }

            let iters_done = self
                .worker_progress
                .get(idx)
                .map(|p| p.load(std::sync::atomic::Ordering::Relaxed))
                .unwrap_or(0);

            if let Some(iters_done) = w.apply_progress(iters_done) {
                if iters_done != w.last_emitted_iters_done {
                    w.last_emitted_iters_done = iters_done;
                    self.emit(EngineEvent::WorkerProgress {
                        worker_idx: idx,
                        iters_done,
                        iters_total: w.iters_total,
                        iters_per_sec: w.speed_its_per_sec,
                    });
                }
            }

            total_speed = total_speed.saturating_add(w.speed_its_per_sec);
        }

        if total_speed != self.last_speed_emitted {
            self.last_speed_emitted = total_speed;
            self.emit(EngineEvent::Speed {
                iters_per_sec: total_speed,
                busy,
                total: self.workers.len(),
            });
        }
    }

    async fn run(mut self) -> anyhow::Result<()> {
        self.emit(EngineEvent::Started);

        // Warn once if GPU enabled but not wired yet.
        self.maybe_warn_gpu_not_ready();

        let mut tick = tokio::time::interval(self.cfg.progress_tick);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        let mut last_snapshot_at = Instant::now();
        let mut result: anyhow::Result<()> = Ok(());

        loop {
            if self.inner.should_stop() && self.all_idle() {
                break;
            }

            self.maybe_start_fetch();

            tokio::select! {
                _ = self.inner.notify.notified() => {}
                _ = tick.tick() => {}
                Some(ev) = self.internal_rx.recv() => {
                    match ev {
                        WorkerInternalEvent::StageChanged { worker_idx, stage } => {
                            if let Some(w) = self.workers.get_mut(worker_idx) {
                                w.set_stage(stage);
                            }
                            self.push_snapshot();
                            self.emit(EngineEvent::WorkerStage { worker_idx, stage });
                        }
                        WorkerInternalEvent::Finished { outcome } => {
                            let worker_idx = outcome.worker_idx;
                            if let Some(w) = self.workers.get_mut(worker_idx) {
                                w.finish_job();
                            }
                            self.recent_jobs.push_back(outcome.clone());
                            while self.recent_jobs.len() > self.cfg.recent_jobs_max {
                                self.recent_jobs.pop_front();
                            }
                            self.push_snapshot();
                            self.emit(EngineEvent::JobFinished { outcome });
                        }
                        WorkerInternalEvent::Warning { message } => {
                            self.emit(EngineEvent::Warning { message });
                        }
                        WorkerInternalEvent::Error { message } => {
                            self.emit(EngineEvent::Error { message });
                        }
                    }
                }
                Some(res) = async { self.fetch_task.as_mut().map(|h| h.await).transpose() } => {
                    self.fetch_task = None;
                    match res {
                        Ok(Ok(items)) => {
                            for item in items {
                                self.pending.push_back(item);
                            }
                        }
                        Ok(Err(err)) => {
                            self.fetch_backoff = Some(Box::pin(tokio::time::sleep(Duration::from_secs(5))));
                            self.emit(EngineEvent::Warning {
                                message: format!("warning: fetch work failed: {err:#}"),
                            });
                        }
                        Err(err) => {
                            self.fetch_backoff = Some(Box::pin(tokio::time::sleep(Duration::from_secs(5))));
                            self.emit(EngineEvent::Warning {
                                message: format!("warning: fetch join failed: {err:#}"),
                            });
                        }
                    }
                }
                _ = async {
                    if let Some(s) = self.fetch_backoff.as_mut() {
                        s.as_mut().await;
                    }
                }, if self.fetch_backoff.is_some() => {
                    self.fetch_backoff = None;
                }
                _ = tokio::time::sleep(self.cfg.idle_sleep), if self.pending.is_empty() => {}
            }

            if let Err(err) = self.assign_jobs_cpu().await {
                let message = format!("assign jobs failed: {err:#}");
                self.emit(EngineEvent::Error { message: message.clone() });
                result = Err(anyhow::anyhow!("{message}"));
                break;
            }

            let now = Instant::now();
            if now.duration_since(last_snapshot_at) >= Duration::from_secs(1) {
                last_snapshot_at = now;
                self.update_speeds_and_emit();
                self.push_snapshot();
            }
        }

        self.pending.clear();
        self.shutdown_workers().await;

        self.emit(EngineEvent::Stopped);
        self.push_snapshot();

        result
    }
}

pub(crate) fn start_engine(cfg: EngineConfig) -> EngineHandle {
    let (event_tx, _) = broadcast::channel::<EngineEvent>(1024);
    let (snapshot_tx, snapshot_rx) = watch::channel(StatusSnapshot {
        stop_requested: false,
        workers: Vec::new(),
        recent_jobs: Vec::new(),
    });

    let inner = Arc::new(EngineInner {
        event_tx,
        snapshot_rx,
        stop_requested: AtomicBool::new(false),
        notify: tokio::sync::Notify::new(),
    });

    let join = tokio::spawn(run_engine(inner.clone(), snapshot_tx, cfg));
    EngineHandle { inner, join }
}

async fn run_engine(
    inner: Arc<EngineInner>,
    snapshot_tx: watch::Sender<StatusSnapshot>,
    mut cfg: EngineConfig,
) -> anyhow::Result<()> {
    if cfg.parallel == 0 {
        cfg.parallel = 1;
    }
    if cfg.idle_sleep == Duration::ZERO {
        cfg.idle_sleep = EngineConfig::DEFAULT_IDLE_SLEEP;
    }
    if cfg.progress_steps == 0 {
        cfg.progress_steps = EngineConfig::DEFAULT_PROGRESS_STEPS;
    }
    if cfg.progress_tick == Duration::ZERO {
        cfg.progress_tick = EngineConfig::DEFAULT_PROGRESS_TICK;
    }
    if cfg.recent_jobs_max == 0 {
        cfg.recent_jobs_max = EngineConfig::DEFAULT_RECENT_JOBS_MAX;
    }

    // chiavdf fast wrapper uses a process-wide memory budget.
    bbr_client_chiavdf_fast::set_bucket_memory_budget_bytes(cfg.mem_budget_bytes);

    let http = reqwest::Client::builder()
        .timeout(Duration::from_secs(60))
        .build()?;

    // Spawn CPU worker threads (each owns its own current-thread Tokio runtime).
    let (internal_tx, internal_rx) = mpsc::unbounded_channel::<WorkerInternalEvent>();

    let mut worker_cmds = Vec::with_capacity(cfg.parallel);
    let mut worker_progress = Vec::with_capacity(cfg.parallel);
    let mut worker_threads: Vec<std::thread::JoinHandle<()>> = Vec::with_capacity(cfg.parallel);

    let submitter = Arc::new(tokio::sync::RwLock::new(cfg.submitter.clone()));
    let warned_invalid_reward_address = Arc::new(AtomicBool::new(false));
// ----------------------------------------
// CPU pinning configuration (process-wide)
// ----------------------------------------
// Comments in English as requested.
crate::worker::set_cpu_pinning_enabled(cfg.cpu_pin_threads);
crate::worker::set_cpu_pin_policy(cfg.cpu_reserve_core0, cfg.cpu_reverse_cores);

// Optional core allow/block lists, parsed once here and stored process-wide.
// This avoids changing the worker command enum and keeps the project structure intact.
let allowlist = match cfg.cpu_core_allowlist.as_deref() {
    Some(spec) if !spec.trim().is_empty() => match parse_core_list(spec) {
        Ok(v) => Some(v),
        Err(err) => {
            let _ = inner.event_tx.send(EngineEvent::Warning {
                message: format!(
                    "warning: invalid --cpu-cores specification {spec:?}: {err}; ignoring allowlist"
                ),
            });
            None
        }
    },
    _ => None,
};

let blocklist = match cfg.cpu_core_blocklist.as_deref() {
    Some(spec) if !spec.trim().is_empty() => match parse_core_list(spec) {
        Ok(v) => Some(v),
        Err(err) => {
            let _ = inner.event_tx.send(EngineEvent::Warning {
                message: format!(
                    "warning: invalid --cpu-cores-exclude specification {spec:?}: {err}; ignoring blocklist"
                ),
            });
            None
        }
    },
    _ => None,
};

crate::worker::set_cpu_core_allowlist(allowlist);
crate::worker::set_cpu_core_blocklist(blocklist);


    for worker_idx in 0..cfg.parallel {
        let (tx, rx) = mpsc::channel::<WorkerCommand>(1);
        worker_cmds.push(tx);

        let progress = Arc::new(std::sync::atomic::AtomicU64::new(0));
        worker_progress.push(progress.clone());

        let internal_tx = internal_tx.clone();
        let http = http.clone();
        let submitter = submitter.clone();
        let warned = warned_invalid_reward_address.clone();

        // Dedicated OS thread to make CPU affinity deterministic.
        let handle = std::thread::Builder::new()
            .name(format!("bbr-cpu-worker-{}", worker_idx + 1))
            .spawn(move || {
                // Each worker thread runs its own single-threaded Tokio runtime.
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build();

                let rt = match rt {
                    Ok(v) => v,
                    Err(err) => {
                        let _ = internal_tx.send(WorkerInternalEvent::Error {
                            message: format!(
                                "error: failed to build worker tokio runtime (worker {}): {err:#}",
                                worker_idx + 1
                            ),
                        });
                        return;
                    }
                };

                rt.block_on(async move {
                    crate::worker::run_worker_task(
                        worker_idx,
                        rx,
                        internal_tx,
                        progress,
                        http,
                        submitter,
                        warned,
                    )
                    .await;
                });
            });

        match handle {
            Ok(h) => worker_threads.push(h),
            Err(err) => {
                let _ = inner.event_tx.send(EngineEvent::Error {
                    message: format!("error: failed to spawn cpu worker thread {}: {err:#}", worker_idx + 1),
                });
                return Err(anyhow::anyhow!("failed to spawn cpu worker thread"));
            }
        }
    }

    // Load inflight leases (resume).
    let mut inflight = match InflightStore::load() {
        Ok(Some(store)) => Some(store),
        Ok(None) => None,
        Err(err) => {
            let message = format!("warning: failed to load inflight leases (resume disabled): {err:#}");
            let _ = inner.event_tx.send(EngineEvent::Warning { message });
            None
        }
    };

    // Drop expired inflight leases.
    if let Some(store) = inflight.as_mut() {
        let now = Utc::now().timestamp();
        let expired_job_ids: Vec<u64> = store
            .entries()
            .filter(|entry| entry.lease_expires_at <= now)
            .map(|entry| entry.job.job_id)
            .collect();

        if !expired_job_ids.is_empty() {
            for job_id in expired_job_ids {
                store.remove_job(job_id);
            }

            if let Err(err) = store.persist().await {
                let _ = inner.event_tx.send(EngineEvent::Warning {
                    message: format!("warning: failed to persist expired inflight lease cleanup: {err:#}"),
                });
            }
        }
    }

    // Seed pending from inflight.
    let mut pending = VecDeque::new();
    if let Some(store) = inflight.as_ref() {
        for entry in store.entries() {
            pending.push_back(WorkJobItem {
                lease_id: entry.lease_id.clone(),
                lease_expires_at: entry.lease_expires_at,
                job: entry.job.clone(),
            });
        }
        if !pending.is_empty() {
            let _ = inner.event_tx.send(EngineEvent::Warning {
                message: format!(
                    "Loaded {} inflight lease(s) from previous run; processing them before leasing new work.",
                    pending.len()
                ),
            });
        }
    }

    let workers = (0..cfg.parallel).map(|_| WorkerRuntime::new()).collect();

    let runtime = EngineRuntime {
        http,
        cfg,
        workers,
        worker_cmds,
        worker_progress,
        internal_rx,
        worker_threads,
        pending,
        fetch_task: None,
        fetch_backoff: None,
        inflight: inflight.take(),
        recent_jobs: VecDeque::new(),
        snapshot_tx,
        inner,
        last_speed_emitted: 0,
    };

    runtime.push_snapshot();
    runtime.run().await
}
