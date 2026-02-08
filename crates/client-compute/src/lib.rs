//! Compute backend selection wrapper.
//!
//! The original engine is CPU-only and calls directly into `bbr-client-chiavdf-fast`.
//! This crate provides an external abstraction layer so a future GPU backend can be
//! integrated with minimal changes to the existing engine.

use anyhow::Context;

use std::sync::{Arc, OnceLock};

use bbr_client_gpu::{auto as gpu_auto, auto::MultiGpuScheduler, execute, GpuBackend};
use bbr_client_core::logging::Logger;

use bbr_client_chiavdf_fast::{
    ChiavdfBatchJob,
    prove_one_weso_fast_streaming_getblock_opt,
    prove_one_weso_fast_streaming_getblock_opt_with_progress,
    prove_one_weso_fast_streaming_getblock_opt_batch,
    prove_one_weso_fast_streaming_getblock_opt_batch_with_progress,
};

fn log_line(line: &str) {
    Logger::try_init("compute");
    if let Some(l) = Logger::global() {
        l.line(line);
    }
}

/// User-requested compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackendRequest {
    /// Force CPU compute.
    Cpu,
    /// Force GPU compute (with CPU fallback on failure).
    Gpu,
    /// Prefer GPU when available, otherwise CPU (default).
    Auto,
}

impl ComputeBackendRequest {
    /// Parse a backend request from a string (`cpu`, `gpu`, `auto`). Case-insensitive.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Self::Cpu),
            "gpu" => Some(Self::Gpu),
            "auto" => Some(Self::Auto),
            _ => None,
        }
    }
}

/// Effective backend used for a given compute call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackendSelected {
    Cpu,
    Cuda,
    OpenCl,
}

static CUDA_SCHEDULER: OnceLock<Arc<MultiGpuScheduler>> = OnceLock::new();
static OPENCL_SCHEDULER: OnceLock<Arc<MultiGpuScheduler>> = OnceLock::new();

/// Read requested backend from `BBR_COMPUTE_BACKEND` (defaults to `auto`).
pub fn requested_backend_from_env() -> ComputeBackendRequest {
    std::env::var("BBR_COMPUTE_BACKEND")
        .ok()
        .as_deref()
        .and_then(ComputeBackendRequest::parse)
        .unwrap_or(ComputeBackendRequest::Auto)
}

/// Read fallback behavior from `BBR_COMPUTE_FALLBACK` (defaults to true).
pub fn allow_cpu_fallback_from_env() -> bool {
    match std::env::var("BBR_COMPUTE_FALLBACK").ok().as_deref() {
        Some("0") | Some("false") | Some("no") | Some("off") => false,
        _ => true,
    }
}

/// Best-effort GPU availability probe.
pub fn gpu_available() -> bool {
    gpu_auto::pick_best(gpu_auto::GpuPreference::CudaFirst).is_some()
}

fn pick_backend(req: ComputeBackendRequest) -> ComputeBackendSelected {
    match req {
        ComputeBackendRequest::Cpu => ComputeBackendSelected::Cpu,
        ComputeBackendRequest::Gpu | ComputeBackendRequest::Auto => match gpu_auto::pick_best(
            gpu_auto::GpuPreference::CudaFirst,
        ) {
            Some(GpuBackend::Cuda) => ComputeBackendSelected::Cuda,
            Some(GpuBackend::OpenCl) => ComputeBackendSelected::OpenCl,
            None => ComputeBackendSelected::Cpu,
        },
    }
}

fn selected_to_gpu_backend(selected: ComputeBackendSelected) -> GpuBackend {
    match selected {
        ComputeBackendSelected::Cuda => GpuBackend::Cuda,
        ComputeBackendSelected::OpenCl => GpuBackend::OpenCl,
        ComputeBackendSelected::Cpu => unreachable!("CPU selected has no GPU backend"),
    }
}

fn is_gpu_not_implemented(err: &anyhow::Error) -> bool {
    err.chain()
        .any(|e| e.to_string().contains("BBR_GPU_NOT_IMPLEMENTED"))
}

fn init_scheduler_once(
    slot: &'static OnceLock<Arc<MultiGpuScheduler>>,
    backend: GpuBackend,
) -> anyhow::Result<&'static Arc<MultiGpuScheduler>> {
    if let Some(v) = slot.get() {
        log_line(&format!("[compute] scheduler already initialized for {backend:?}"));
        return Ok(v);
    }

    let cfg = gpu_auto::config_from_env();
    log_line(&format!("[compute] creating scheduler for {backend:?} cfg={cfg:?}"));

    let created = Arc::new(
        MultiGpuScheduler::new(backend, &cfg)
            .with_context(|| format!("failed to create scheduler for {backend:?}"))?,
    );

    let _ = slot.set(created);
    slot.get()
        .context("scheduler OnceLock set/get failed unexpectedly")
}

fn scheduler_for_backend(
    backend: ComputeBackendSelected,
) -> anyhow::Result<Option<&'static Arc<MultiGpuScheduler>>> {
    match backend {
        ComputeBackendSelected::Cuda => Ok(Some(init_scheduler_once(&CUDA_SCHEDULER, GpuBackend::Cuda)?)),
        ComputeBackendSelected::OpenCl => Ok(Some(init_scheduler_once(&OPENCL_SCHEDULER, GpuBackend::OpenCl)?)),
        ComputeBackendSelected::Cpu => Ok(None),
    }
}

fn cpu_prove_single(
    challenge: &[u8],
    x: &[u8],
    y_ref: &[u8],
    discriminant_bits: usize,
    num_iterations: u64,
) -> anyhow::Result<Vec<u8>> {
    prove_one_weso_fast_streaming_getblock_opt(
        challenge,
        x,
        y_ref,
        discriminant_bits,
        num_iterations,
    )
    .context("chiavdf prove_one_weso_fast_streaming_getblock_opt")
    .map_err(Into::into)
}

fn cpu_prove_single_with_progress<F>(
    challenge: &[u8],
    x: &[u8],
    y_ref: &[u8],
    discriminant_bits: usize,
    num_iterations: u64,
    progress_interval: u64,
    progress: F,
) -> anyhow::Result<Vec<u8>>
where
    F: FnMut(u64) + Send + 'static,
{
    prove_one_weso_fast_streaming_getblock_opt_with_progress(
        challenge,
        x,
        y_ref,
        discriminant_bits,
        num_iterations,
        progress_interval,
        progress,
    )
    .context("chiavdf prove_one_weso_fast_streaming_getblock_opt_with_progress")
    .map_err(Into::into)
}

fn cpu_prove_batch(
    challenge: &[u8],
    x: &[u8],
    discriminant_bits: usize,
    jobs: &[ChiavdfBatchJob<'_>],
) -> anyhow::Result<Vec<Vec<u8>>> {
    prove_one_weso_fast_streaming_getblock_opt_batch(challenge, x, discriminant_bits, jobs)
        .context("chiavdf prove_one_weso_fast_streaming_getblock_opt_batch")
        .map_err(Into::into)
}

fn cpu_prove_batch_with_progress<F>(
    challenge: &[u8],
    x: &[u8],
    discriminant_bits: usize,
    jobs: &[ChiavdfBatchJob<'_>],
    progress_interval: u64,
    progress: F,
) -> anyhow::Result<Vec<Vec<u8>>>
where
    F: FnMut(u64) + Send + 'static,
{
    prove_one_weso_fast_streaming_getblock_opt_batch_with_progress(
        challenge,
        x,
        discriminant_bits,
        jobs,
        progress_interval,
        progress,
    )
    .context("chiavdf prove_one_weso_fast_streaming_getblock_opt_batch_with_progress")
    .map_err(Into::into)
}

pub fn prove_single(
    challenge: &[u8],
    x: &[u8],
    y_ref: &[u8],
    discriminant_bits: usize,
    num_iterations: u64,
) -> anyhow::Result<(ComputeBackendSelected, Vec<u8>)> {
    let req = requested_backend_from_env();
    let allow_fallback = allow_cpu_fallback_from_env();

    let selected = pick_backend(req);
    log_line(&format!(
        "[compute] prove_single req={req:?} selected={selected:?} fallback={allow_fallback}",
    ));

    match selected {
        ComputeBackendSelected::Cpu => Ok((ComputeBackendSelected::Cpu, cpu_prove_single(challenge, x, y_ref, discriminant_bits, num_iterations)?)),
        ComputeBackendSelected::Cuda | ComputeBackendSelected::OpenCl => {
            let sched = scheduler_for_backend(selected)?
                .context("GPU backend selected but no scheduler could be created")?;

            let backend = selected_to_gpu_backend(selected);
            let gpu_attempt = sched.run_with_cpu_fallback(
                |dev| execute::prove_single(backend, dev, challenge, x, y_ref, discriminant_bits, num_iterations),
                || cpu_prove_single(challenge, x, y_ref, discriminant_bits, num_iterations),
                allow_fallback,
            );

            match gpu_attempt {
                Ok((Some(dev), out)) => {
                    log_line(&format!("[compute] gpu_success device={dev}"));
                    Ok((selected, out))
                }
                Ok((None, out)) => {
                    log_line("[compute] cpu_fallback_taken");
                    Ok((ComputeBackendSelected::Cpu, out))
                }
                Err(err) if allow_fallback && is_gpu_not_implemented(&err) => {
                    log_line("[compute] gpu_not_implemented -> cpu");
                    Ok((ComputeBackendSelected::Cpu, cpu_prove_single(challenge, x, y_ref, discriminant_bits, num_iterations)?))
                }
                Err(err) => Err(err),
            }
        }
    }
}

pub fn prove_single_with_progress<F>(
    challenge: &[u8],
    x: &[u8],
    y_ref: &[u8],
    discriminant_bits: usize,
    num_iterations: u64,
    progress_interval: u64,
    progress: F,
) -> anyhow::Result<(ComputeBackendSelected, Vec<u8>)>
where
    F: FnMut(u64) + Send + 'static,
{
    let progress_shared: Arc<std::sync::Mutex<F>> = Arc::new(std::sync::Mutex::new(progress));

    let req = requested_backend_from_env();
    let allow_fallback = allow_cpu_fallback_from_env();

    let selected = pick_backend(req);
    log_line(&format!(
        "[compute] prove_single_with_progress req={req:?} selected={selected:?} fallback={allow_fallback}",
    ));

    match selected {
        ComputeBackendSelected::Cpu => {
            let ps = Arc::clone(&progress_shared);
            let cb = move |n: u64| {
                if let Ok(mut p) = ps.lock() {
                    (*p)(n);
                }
            };

            Ok((ComputeBackendSelected::Cpu, cpu_prove_single_with_progress(challenge, x, y_ref, discriminant_bits, num_iterations, progress_interval, cb)?))
        }
        ComputeBackendSelected::Cuda | ComputeBackendSelected::OpenCl => {
            let sched = scheduler_for_backend(selected)?
                .context("GPU backend selected but no scheduler could be created")?;

            let ps1 = Arc::clone(&progress_shared);
            let cpu_fallback_1 = move || {
                let ps_inner = Arc::clone(&ps1);
                let cb = move |n: u64| {
                    if let Ok(mut p) = ps_inner.lock() {
                        (*p)(n);
                    }
                };

                cpu_prove_single_with_progress(
                    challenge,
                    x,
                    y_ref,
                    discriminant_bits,
                    num_iterations,
                    progress_interval,
                    cb,
                )
            };

            let backend = selected_to_gpu_backend(selected);
            let ps_gpu = Arc::clone(&progress_shared);

            let gpu_attempt = sched.run_with_cpu_fallback(
                move |dev| {
                    let ps_inner = Arc::clone(&ps_gpu);
                    let cb = move |n: u64| {
                        if let Ok(mut p) = ps_inner.lock() {
                            (*p)(n);
                        }
                    };

                    execute::prove_single_with_progress(
                        backend,
                        dev,
                        challenge,
                        x,
                        y_ref,
                        discriminant_bits,
                        num_iterations,
                        progress_interval,
                        cb,
                    )
                },
                cpu_fallback_1,
                allow_fallback,
            );

            match gpu_attempt {
                Ok((Some(dev), out)) => {
                    log_line(&format!("[compute] gpu_success device={dev}"));
                    Ok((selected, out))
                }
                Ok((None, out)) => {
                    log_line("[compute] cpu_fallback_taken");
                    Ok((ComputeBackendSelected::Cpu, out))
                }
                Err(err) if allow_fallback && is_gpu_not_implemented(&err) => {
                    log_line("[compute] gpu_not_implemented -> cpu");
                    let ps2 = Arc::clone(&progress_shared);
                    let cb = move |n: u64| {
                        if let Ok(mut p) = ps2.lock() {
                            (*p)(n);
                        }
                    };

                    Ok((ComputeBackendSelected::Cpu, cpu_prove_single_with_progress(challenge, x, y_ref, discriminant_bits, num_iterations, progress_interval, cb)?))
                }
                Err(err) => Err(err),
            }
        }
    }
}

pub fn prove_batch(
    challenge: &[u8],
    x: &[u8],
    discriminant_bits: usize,
    jobs: &[ChiavdfBatchJob<'_>],
) -> anyhow::Result<(ComputeBackendSelected, Vec<Vec<u8>>)> {
    let req = requested_backend_from_env();
    let allow_fallback = allow_cpu_fallback_from_env();

    let selected = pick_backend(req);
    log_line(&format!(
        "[compute] prove_batch req={req:?} selected={selected:?} fallback={allow_fallback}",
    ));

    match selected {
        ComputeBackendSelected::Cpu => Ok((ComputeBackendSelected::Cpu, cpu_prove_batch(challenge, x, discriminant_bits, jobs)?)),
        ComputeBackendSelected::Cuda | ComputeBackendSelected::OpenCl => {
            let sched = scheduler_for_backend(selected)?
                .context("GPU backend selected but no scheduler could be created")?;

            let backend = selected_to_gpu_backend(selected);
            let gpu_attempt = sched.run_with_cpu_fallback(
                |dev| execute::prove_batch(backend, dev, challenge, x, discriminant_bits, jobs),
                || cpu_prove_batch(challenge, x, discriminant_bits, jobs),
                allow_fallback,
            );

            match gpu_attempt {
                Ok((Some(dev), out)) => {
                    log_line(&format!("[compute] gpu_success device={dev}"));
                    Ok((selected, out))
                }
                Ok((None, out)) => {
                    log_line("[compute] cpu_fallback_taken");
                    Ok((ComputeBackendSelected::Cpu, out))
                }
                Err(err) if allow_fallback && is_gpu_not_implemented(&err) => {
                    log_line("[compute] gpu_not_implemented -> cpu");
                    Ok((ComputeBackendSelected::Cpu, cpu_prove_batch(challenge, x, discriminant_bits, jobs)?))
                }
                Err(err) => Err(err),
            }
        }
    }
}

pub fn prove_batch_with_progress<F>(
    challenge: &[u8],
    x: &[u8],
    discriminant_bits: usize,
    jobs: &[ChiavdfBatchJob<'_>],
    progress_interval: u64,
    progress: F,
) -> anyhow::Result<(ComputeBackendSelected, Vec<Vec<u8>>)>
where
    F: FnMut(u64) + Send + 'static,
{
    let progress_shared: Arc<std::sync::Mutex<F>> = Arc::new(std::sync::Mutex::new(progress));

    let req = requested_backend_from_env();
    let allow_fallback = allow_cpu_fallback_from_env();

    let selected = pick_backend(req);
    log_line(&format!(
        "[compute] prove_batch_with_progress req={req:?} selected={selected:?} fallback={allow_fallback}",
    ));

    match selected {
        ComputeBackendSelected::Cpu => {
            let ps = Arc::clone(&progress_shared);
            let cb = move |n: u64| {
                if let Ok(mut p) = ps.lock() {
                    (*p)(n);
                }
            };

            Ok((ComputeBackendSelected::Cpu, cpu_prove_batch_with_progress(challenge, x, discriminant_bits, jobs, progress_interval, cb)?))
        }
        ComputeBackendSelected::Cuda | ComputeBackendSelected::OpenCl => {
            let sched = scheduler_for_backend(selected)?
                .context("GPU backend selected but no scheduler could be created")?;

            let ps1 = Arc::clone(&progress_shared);
            let cpu_fallback_1 = move || {
                let ps_inner = Arc::clone(&ps1);
                let cb = move |n: u64| {
                    if let Ok(mut p) = ps_inner.lock() {
                        (*p)(n);
                    }
                };

                cpu_prove_batch_with_progress(
                    challenge,
                    x,
                    discriminant_bits,
                    jobs,
                    progress_interval,
                    cb,
                )
            };

            let backend = selected_to_gpu_backend(selected);
            let ps_gpu = Arc::clone(&progress_shared);

            let gpu_attempt = sched.run_with_cpu_fallback(
                move |dev| {
                    let ps_inner = Arc::clone(&ps_gpu);
                    let cb = move |n: u64| {
                        if let Ok(mut p) = ps_inner.lock() {
                            (*p)(n);
                        }
                    };

                    execute::prove_batch_with_progress(
                        backend,
                        dev,
                        challenge,
                        x,
                        discriminant_bits,
                        jobs,
                        progress_interval,
                        cb,
                    )
                },
                cpu_fallback_1,
                allow_fallback,
            );

            match gpu_attempt {
                Ok((Some(dev), out)) => {
                    log_line(&format!("[compute] gpu_success device={dev}"));
                    Ok((selected, out))
                }
                Ok((None, out)) => {
                    log_line("[compute] cpu_fallback_taken");
                    Ok((ComputeBackendSelected::Cpu, out))
                }
                Err(err) if allow_fallback && is_gpu_not_implemented(&err) => {
                    log_line("[compute] gpu_not_implemented -> cpu");
                    let ps2 = Arc::clone(&progress_shared);
                    let cb = move |n: u64| {
                        if let Ok(mut p) = ps2.lock() {
                            (*p)(n);
                        }
                    };

                    Ok((ComputeBackendSelected::Cpu, cpu_prove_batch_with_progress(challenge, x, discriminant_bits, jobs, progress_interval, cb)?))
                }
                Err(err) => Err(err),
            }
        }
    }
}
