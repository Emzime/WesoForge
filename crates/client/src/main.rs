mod bench;
mod cli;
mod constants;
mod format;
mod shutdown;
mod terminal;
mod ui;

use clap::Parser;
use std::io::IsTerminal;
use std::time::Duration;

use bbr_client_chiavdf_fast::{set_bucket_memory_budget_bytes, set_enable_streaming_stats};
use bbr_client_core::submitter::{SubmitterConfig, ensure_submitter_config};
use bbr_client_core::logging::Logger;
use bbr_client_engine::{EngineConfig, EngineEvent, start_engine};
use bbr_client_gpu as client_gpu;

use crate::bench::run_benchmark;
use crate::cli::{Cli, ComputeBackendArg, WorkMode};
use crate::constants::PROGRESS_BAR_STEPS;
use crate::format::{format_job_done_line, humanize_submit_reason};
use crate::shutdown::{ShutdownController, ShutdownEvent, spawn_ctrl_c_handler};
use crate::terminal::TuiTerminal;
use crate::ui::Ui;

fn format_outcome_status(outcome: &bbr_client_engine::JobOutcome) -> String {
    if let Some(err) = &outcome.error {
        return err.clone();
    }

    let reason = outcome.submit_reason.as_deref().unwrap_or("unknown").trim();
    let mut status = humanize_submit_reason(reason);

    if outcome.output_mismatch {
        status.push_str(" (output mismatch)");
    }
    if let Some(detail) = outcome.submit_detail.as_deref() {
        if !detail.is_empty() && detail != reason {
            status.push_str(&format!(" ({detail})"));
        }
    }
    status
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn print_gpu_list() -> anyhow::Result<()> {
    // Read user selection from env (set by CLI parsing in `main` before calling this function).
    let cfg = client_gpu::GpuConfig::from_env();

    println!("GPU selection: {}", cfg.devices.to_env_string());

    // CUDA devices
    #[cfg(feature = "cuda")]
    {
        match client_gpu::cuda::enumerate_devices() {
            Ok(list) => {
                if list.is_empty() {
                    println!("CUDA: no devices detected");
                } else {
                    println!("CUDA devices:");
                    for d in list {
                        let enabled = cfg.devices.matches(d.index);
                        println!(
                            "  [{idx}] {name} | VRAM={mem} | SM={sm} | CC={cc} | enabled={enabled}",
                            idx = d.index,
                            name = d.name,
                            mem = d.total_mem_bytes
                                .map(human_bytes)
                                .unwrap_or_else(|| "unknown".to_string()),
                            sm = d.sm_count
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "unknown".to_string()),
                            cc = d.compute_capability
                                .map(|(ma, mi)| format!("{ma}.{mi}"))
                                .unwrap_or_else(|| "unknown".to_string()),
                            enabled = enabled
                        );
                    }
                }
            }
            Err(err) => println!("CUDA: unavailable ({err:#})"),
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        println!("CUDA: not compiled (feature 'cuda' disabled)");
    }

    // OpenCL devices
    #[cfg(feature = "opencl")]
    {
        match client_gpu::opencl::enumerate_devices() {
            Ok(list) => {
                if list.is_empty() {
                    println!("OpenCL: no devices detected");
                } else {
                    println!("OpenCL devices:");
                    for d in list {
                        let enabled = cfg.devices.matches(d.index);
                        println!(
                            "  [{idx}] {vendor} {name} | type={ty} | VRAM={mem} | CU={cu} | enabled={enabled}",
                            idx = d.index,
                            vendor = d.vendor.unwrap_or_else(|| "unknown".to_string()),
                            name = d.name,
                            ty = d.device_type.unwrap_or_else(|| "unknown".to_string()),
                            mem = d.global_mem_bytes
                                .map(human_bytes)
                                .unwrap_or_else(|| "unknown".to_string()),
                            cu = d.compute_units
                                .map(|v| v.to_string())
                                .unwrap_or_else(|| "unknown".to_string()),
                            enabled = enabled
                        );
                    }
                }
            }
            Err(err) => println!("OpenCL: unavailable ({err:#})"),
        }
    }
    #[cfg(not(feature = "opencl"))]
    {
        println!("OpenCL: not compiled (feature 'opencl' disabled)");
    }

    Ok(())
}

fn human_bytes(v: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = v as f64;
    let mut idx = 0usize;
    while value >= 1024.0 && idx + 1 < UNITS.len() {
        value /= 1024.0;
        idx += 1;
    }
    if idx == 0 {
        format!("{v} {}", UNITS[idx])
    } else {
        format!("{:.2} {}", value, UNITS[idx])
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Select compute backend (CPU/GPU) without altering the engine logic.
    // The engine calls into `bbr-client-compute`, which reads these env vars.
    let compute_backend_str = match cli.compute_backend {
        ComputeBackendArg::Cpu => "cpu",
        ComputeBackendArg::Gpu => "gpu",
        ComputeBackendArg::Auto => "auto",
    };
    std::env::set_var("BBR_COMPUTE_BACKEND", compute_backend_str);
    // Default: allow GPU->CPU fallback when GPU is requested or in auto mode.
    if cli.compute_backend != ComputeBackendArg::Cpu {
        std::env::set_var("BBR_COMPUTE_FALLBACK", "1");
    }

    // GPU-specific options (backend-controlled). Keep the engine unaware of these details.
    //
    // Device selection:
    // - `BBR_GPU_DEVICES` is the preferred multi-device selector (`all`, `none`, or `0,2,3`).
    // - `BBR_GPU_DEVICE` is a legacy single-device selector, used only if `BBR_GPU_DEVICES` is not set.
    if let Some(v) = cli.gpu_devices.as_deref() {
        std::env::set_var("BBR_GPU_DEVICES", v);
    } else if let Some(v) = cli.gpu_device {
        std::env::set_var("BBR_GPU_DEVICE", v.to_string());
    }

    if let Some(v) = cli.gpu_streams {
        std::env::set_var("BBR_GPU_STREAMS", v.to_string());
    }
    if let Some(v) = cli.gpu_batch_size {
        std::env::set_var("BBR_GPU_BATCH_SIZE", v.to_string());
    }
    if let Some(v) = cli.gpu_mem_budget_bytes {
        std::env::set_var("BBR_GPU_MEM_BUDGET", v.to_string());
    }

    if cli.list_gpus {
        print_gpu_list()?;
        return Ok(());
    }

    if cli.bench {
        set_bucket_memory_budget_bytes(cli.mem_budget_bytes);
        set_enable_streaming_stats(true);
        run_benchmark(cli.mode, cli.parallel as usize)?;
        return Ok(());
    }

    let interactive = std::io::stdin().is_terminal();
    let submitter = match ensure_submitter_config(interactive) {
        Ok(Some(cfg)) => cfg,
        Ok(None) => SubmitterConfig::default(),
        Err(err) => {
            eprintln!("warning: failed to read/write submitter config: {err:#}");
            SubmitterConfig::default()
        }
    };

    if cli.parallel == 0 {
        anyhow::bail!("--parallel must be >= 1");
    }
    let parallel = cli.parallel as usize;

    // Centralized logs in `<repo_root>/logs/` (see `bbr_client_core::logging`).
    Logger::init("cli");
    if let Some(l) = Logger::global() {
        l.line(&format!(
            "[cli] started | version={} | parallel={} | backend={} | compute_backend={}",
            env!("CARGO_PKG_VERSION"),
            parallel,
            cli.backend_url,
            compute_backend_str
        ));
    }

    let tui_enabled = !cli.no_tui && std::io::stdout().is_terminal();
    let warn_tui_too_many_workers = tui_enabled && parallel > 32;
    let progress_steps = if tui_enabled { PROGRESS_BAR_STEPS } else { 0 };

    let use_groups = cli.mode == WorkMode::Group;

    let engine = start_engine(EngineConfig {
        backend_url: cli.backend_url.clone(),
        parallel,
        use_groups,
        mem_budget_bytes: cli.mem_budget_bytes,
        submitter,
        idle_sleep: Duration::ZERO,
        progress_steps,
        progress_tick: Duration::ZERO,
        recent_jobs_max: 0,
        pin_mode: cli.pin.into(),
    });

    if let Some(l) = Logger::global() {
        l.engine_started(
            "cli",
            parallel,
            if use_groups { "group" } else { "proof" },
            cli.mem_budget_bytes,
        );
    }

    let mut events = engine.subscribe();

    let shutdown = std::sync::Arc::new(ShutdownController::new());
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::unbounded_channel::<ShutdownEvent>();
    let tui_terminal = if tui_enabled && std::io::stdin().is_terminal() {
        Some(TuiTerminal::enter(shutdown.clone(), shutdown_tx.clone())?)
    } else {
        None
    };
    if tui_terminal.is_none() {
        spawn_ctrl_c_handler(shutdown.clone(), shutdown_tx);
    }

    let startup = format!(
        "wesoforge {} parallel={}",
        env!("CARGO_PKG_VERSION"),
        parallel
    );

    let mut ui = if tui_enabled {
        Some(Ui::new(parallel))
    } else {
        None
    };
    if let Some(ui) = &ui {
        ui.println(&startup);
    } else {
        println!("{startup}");
    }
    if warn_tui_too_many_workers {
        let msg = format!(
            "warning: --parallel={} is high; TUI rendering is not optimized for this many progress bars. Consider running with --no-tui.",
            parallel
        );
        if let Some(ui) = &ui {
            ui.println(&msg);
        } else {
            eprintln!("{msg}");
        }
    }

    let mut worker_busy = vec![false; parallel];
    let mut worker_speed: Vec<u64> = vec![0; parallel];

    let logger = Logger::global();
    let mut last_progress_ms: Vec<u128> = vec![0; parallel];

    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut immediate_exit = false;

    loop {
        tokio::select! {
            ev_opt = shutdown_rx.recv() => {
                match ev_opt {
                    Some(ShutdownEvent::Graceful) => {
                        if let Some(l) = logger {
                            l.warn("stop requested - graceful");
                        }
                        if let Some(ui) = &mut ui {
                            ui.set_stop_message("Stop requested — finishing current work before exiting (press CTRL+C again to exit immediately).");
                        } else {
                            eprintln!("Stop requested — finishing current work before exiting (press CTRL+C again to exit immediately).");
                        }
                        engine.request_stop();
                    }
                    Some(ShutdownEvent::Immediate) => {
                        if let Some(l) = logger {
                            l.warn("stop requested - immediate");
                        }
                        if let Some(ui) = &mut ui {
                            ui.set_stop_message("Stop requested again — exiting immediately.");
                        } else {
                            eprintln!("Stop requested again — exiting immediately.");
                        }
                        immediate_exit = true;
                        break;
                    }
                    None => {}
                }
            }
            _ = ticker.tick(), if tui_enabled => {
                if let Some(ui) = &ui {
                    let busy = worker_busy.iter().filter(|v| **v).count();
                    let speed: u64 = worker_speed.iter().sum();
                    ui.tick_global(speed, busy, parallel);
                }
            }
            evt = events.recv() => {
                let evt = match evt {
                    Ok(v) => v,
                    Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                };

                match evt {
                    EngineEvent::Started | EngineEvent::StopRequested => {}
                    EngineEvent::WorkerJobStarted { worker_idx, job } => {
                        if let Some(slot) = worker_busy.get_mut(worker_idx) {
                            *slot = true;
                        }
                        if let Some(ui) = &mut ui {
                            ui.set_worker_job(worker_idx, &job);
                        }

                        if let Some(l) = logger {
                            l.job_started(
                                worker_idx,
                                job.job_id,
                                job.height,
                                job.field_vdf,
                                job.number_of_iterations,
                            );
                        }
                    }
                    EngineEvent::WorkerProgress { worker_idx, iters_done, iters_total, iters_per_sec } => {
                        if let Some(slot) = worker_speed.get_mut(worker_idx) {
                            *slot = iters_per_sec;
                        }
                        if let Some(ui) = &mut ui {
                            ui.set_worker_progress(worker_idx, iters_done);
                        }

                        if let Some(l) = logger {
                            if l.log_progress_enabled() {
                                if worker_idx >= last_progress_ms.len() {
                                    last_progress_ms.resize(worker_idx + 1, 0);
                                }
                                let now = now_ms();
                                if now.saturating_sub(last_progress_ms[worker_idx]) >= 1_000 {
                                    last_progress_ms[worker_idx] = now;
                                    l.line(&format!(
                                        "[worker] progress | worker={} | {}/{} | it_per_s={}",
                                        worker_idx, iters_done, iters_total, iters_per_sec
                                    ));
                                }
                            }
                        }
                    }
                    EngineEvent::WorkerStage { .. } => {}
                    EngineEvent::JobFinished { outcome } => {
                        let worker_idx = outcome.worker_idx;
                        if let Some(slot) = worker_busy.get_mut(worker_idx) {
                            *slot = false;
                        }
                        if let Some(slot) = worker_speed.get_mut(worker_idx) {
                            *slot = 0;
                        }
                        if let Some(ui) = &mut ui {
                            ui.set_worker_idle(worker_idx);
                        }

                        if let Some(l) = logger {
                            l.job_finished(
                                outcome.worker_idx,
                                outcome.job.job_id,
                                outcome.job.height,
                                outcome.job.field_vdf,
                                outcome.total_ms,
                                outcome.submit_reason.as_deref().unwrap_or(""),
                                outcome.error.is_some(),
                                outcome.output_mismatch,
                            );
                        }

                        let status = format_outcome_status(&outcome);
                        let duration = Duration::from_millis(outcome.total_ms);
                        let line = format_job_done_line(
                            outcome.job.height,
                            outcome.job.field_vdf,
                            &status,
                            outcome.job.number_of_iterations,
                            duration,
                        );

                        if let Some(ui) = &ui {
                            ui.println(&line);
                        } else {
                            println!("{line}");
                        }
                    }
                    EngineEvent::Warning { message } => {
                        if let Some(l) = logger {
                            l.warn(&message);
                        }
                        if let Some(ui) = &ui {
                            ui.println(&message);
                        } else {
                            eprintln!("{message}");
                        }
                    }
                    EngineEvent::Error { message } => {
                        if let Some(l) = logger {
                            l.error(&message);
                        }
                        if let Some(ui) = &ui {
                            ui.println(&message);
                        } else {
                            eprintln!("{message}");
                        }
                    }
                    EngineEvent::Stopped => break,
                }
            }
        }
    }

    if let Some(ui) = &ui {
        ui.freeze();
    }

    if immediate_exit {
        drop(tui_terminal);
        std::process::exit(130);
    }

    engine.wait().await?;
    Ok(())
}
