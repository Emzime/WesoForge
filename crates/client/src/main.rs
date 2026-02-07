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
use bbr_client_core::submitter::{ensure_submitter_config, SubmitterConfig};
use bbr_client_engine::{start_engine, EngineConfig, EngineEvent, GpuConfig, GpuMode};

use crate::bench::run_benchmark;
use crate::cli::{Cli, GpuModeCli};
use crate::constants::PROGRESS_BAR_STEPS;
use crate::format::{format_job_done_line, humanize_submit_reason};
use crate::shutdown::{spawn_ctrl_c_handler, ShutdownController, ShutdownEvent};
use crate::terminal::TuiTerminal;
use crate::ui::Ui;

fn format_outcome_status(outcome: &bbr_client_engine::JobOutcome) -> String {
    if let Some(err) = &outcome.error {
        return err.clone();
    }

    let reason = outcome
        .submit_reason
        .as_deref()
        .unwrap_or("unknown")
        .trim();
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

fn map_gpu_mode(cli_mode: GpuModeCli) -> GpuMode {
    match cli_mode {
        GpuModeCli::Off => GpuMode::Off,
        GpuModeCli::Auto => GpuMode::Auto,
        GpuModeCli::Cuda => GpuMode::Cuda,
        GpuModeCli::Opencl => GpuMode::OpenCl,
    }
}

fn apply_legacy_env_override(cfg: &mut GpuConfig) {
    // Legacy env override: WESOFORGE_GPU=cuda|opencl|auto
    // This is kept for backward compatibility with existing scripts.
    if let Ok(v) = std::env::var("WESOFORGE_GPU") {
        match v.trim().to_ascii_lowercase().as_str() {
            "cuda" => cfg.mode = GpuMode::Cuda,
            "opencl" => cfg.mode = GpuMode::OpenCl,
            "auto" | "1" | "true" | "yes" => cfg.mode = GpuMode::Auto,
            "off" | "0" | "false" | "no" => cfg.mode = GpuMode::Off,
            _ => {}
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(algo) = cli.bench {
        set_bucket_memory_budget_bytes(cli.mem_budget_bytes);
        set_enable_streaming_stats(true);

        // Note: `--bench` runs the chiavdf-fast CPU benchmark and does not start the engine.
        // GPU settings apply to the engine path, not this benchmark.
        if let Ok(v) = std::env::var("WESOFORGE_GPU") {
            let v = v.trim().to_string();
            if !v.is_empty() {
                println!(
                    "Note: WESOFORGE_GPU={} is set, but --bench does not use the engine/GPU path.",
                    v
                );
            }
        }

        run_benchmark(algo)?;
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

    let cpu_workers_u16 = cli.effective_cpu_workers();
    let cpu_workers = cpu_workers_u16 as usize;

    let tui_enabled = !cli.no_tui && std::io::stdout().is_terminal();
    let progress_steps = if tui_enabled { PROGRESS_BAR_STEPS } else { 0 };

    let engine = start_engine(EngineConfig {
        backend_url: cli.backend_url.clone(),
        parallel: cpu_workers,
        gpu: {
            let mut cfg = GpuConfig::default();

            // CLI-configured GPU mode and worker count.
            cfg.mode = map_gpu_mode(cli.gpu_mode);
            cfg.workers_per_device = cli.gpu_workers_per_device as usize;

            // Device filters and initial state.
            cfg.allow = cli.gpu_allow.clone().unwrap_or_default();
            cfg.deny = cli.gpu_deny.clone().unwrap_or_default();
            cfg.start_disabled = cli.gpu_start_disabled.clone().unwrap_or_default();

            // Legacy env override (kept for compatibility).
            apply_legacy_env_override(&mut cfg);

            cfg
        },
        mem_budget_bytes: cli.mem_budget_bytes,
        submitter,
        idle_sleep: Duration::ZERO,
        progress_steps,
        progress_tick: Duration::ZERO,
        recent_jobs_max: 0,
    });

    let mut events = engine.subscribe();

    // Important: size the TUI using the *real* worker count (CPU + GPU) from the engine snapshot.
    // This avoids TUI being limited to CPU workers only.
    let initial_snapshot = engine.latest_snapshot();
    let total_workers = initial_snapshot.workers.len();

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
        "wesoforge {} cpu_workers={} gpu_mode={:?} gpu_workers_per_device={}",
        env!("CARGO_PKG_VERSION"),
        cpu_workers,
        engine.latest_snapshot().workers.iter().filter(|w| matches!(w.worker_kind, bbr_client_engine::WorkerKind::Gpu { .. })).count().max(0), // placeholder-like info
        cli.gpu_workers_per_device
    );

    let mut ui = if tui_enabled { Some(Ui::new(total_workers)) } else { None };
    if let Some(ui) = &ui {
        ui.println(&startup);
    } else {
        println!("{startup}");
    }

    if cpu_workers == 0 && matches!(cli.gpu_mode, GpuModeCli::Off) {
        let msg = "warning: cpu-workers=0 and gpu-mode=off => no workers will run.";
        if let Some(ui) = &ui {
            ui.println(msg);
        } else {
            eprintln!("{msg}");
        }
    }

    let mut worker_busy = vec![false; total_workers];
    let mut worker_speed: Vec<u64> = vec![0; total_workers];

    let mut ticker = tokio::time::interval(Duration::from_secs(1));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    let mut immediate_exit = false;

    loop {
        tokio::select! {
            ev_opt = shutdown_rx.recv() => {
                match ev_opt {
                    Some(ShutdownEvent::Graceful) => {
                        if let Some(ui) = &mut ui {
                            ui.set_stop_message("Stop requested — finishing current work before exiting (press CTRL+C again to exit immediately).");
                        } else {
                            eprintln!("Stop requested — finishing current work before exiting (press CTRL+C again to exit immediately).");
                        }
                        engine.request_stop();
                    }
                    Some(ShutdownEvent::Immediate) => {
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
                    ui.tick_global(speed, busy, total_workers);
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
                    }

                    EngineEvent::WorkerProgress { worker_idx, iters_done, iters_per_sec, .. } => {
                        if let Some(slot) = worker_speed.get_mut(worker_idx) {
                            *slot = iters_per_sec;
                        }
                        if let Some(ui) = &mut ui {
                            ui.set_worker_progress(worker_idx, iters_done);
                        }
                    }

                    EngineEvent::WorkerStage { .. } => {}

                    EngineEvent::WorkerEnabled { worker_idx, enabled } => {
                        let msg = format!("Worker {} {}", worker_idx, if enabled { "enabled" } else { "disabled" });
                        if let Some(ui) = &ui {
                            ui.println(&msg);
                        } else {
                            println!("{msg}");
                        }
                    }

                    EngineEvent::GpuEnabled { device_key, enabled } => {
                        let msg = format!("GPU {} {}", device_key, if enabled { "enabled" } else { "disabled" });
                        if let Some(ui) = &ui {
                            ui.println(&msg);
                        } else {
                            println!("{msg}");
                        }
                    }

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
                        if let Some(ui) = &ui {
                            ui.println(&message);
                        } else {
                            eprintln!("{message}");
                        }
                    }

                    EngineEvent::Error { message } => {
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
