#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;
use std::time::Duration;

use reqwest::Url;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, State};
use tokio::sync::Mutex;

#[cfg(feature = "support-devtools")]
use tauri::Manager;

use bbr_client_core::submitter::{load_submitter_config, save_submitter_config, SubmitterConfig};
use bbr_client_engine::{start_engine, EngineConfig, EngineEvent, EngineHandle, GpuConfig, GpuMode, StatusSnapshot};

struct GuiState {
    engine: Mutex<Option<EngineHandle>>,
    progress: Mutex<Vec<WorkerProgressUpdate>>,
}

#[derive(Debug, Clone, Serialize)]
struct WorkerProgressUpdate {
    worker_idx: usize,
    iters_done: u64,
    iters_total: u64,
    iters_per_sec: u64,
}

impl Default for GuiState {
    fn default() -> Self {
        Self {
            engine: Mutex::new(None),
            progress: Mutex::new(Vec::new()),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
enum GpuModeOpt {
    Off,
    Auto,
    Cuda,
    Opencl,
}

impl From<GpuModeOpt> for GpuMode {
    fn from(value: GpuModeOpt) -> Self {
        match value {
            GpuModeOpt::Off => GpuMode::Off,
            GpuModeOpt::Auto => GpuMode::Auto,
            GpuModeOpt::Cuda => GpuMode::Cuda,
            GpuModeOpt::Opencl => GpuMode::OpenCl,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct StartOptions {
    /// Backward-compatible name already used by the UI.
    /// If provided, it sets cpu_workers.
    parallel: Option<u32>,

    /// Number of CPU workers. 0 disables CPU workers.
    cpu_workers: Option<u32>,

    /// GPU mode. If omitted, defaults to Off (unless WESOFORGE_GPU env overrides).
    gpu_mode: Option<GpuModeOpt>,

    /// GPU workers per detected device.
    gpu_workers_per_device: Option<u32>,

    /// Optional allow list of device keys (e.g. ["cuda:0","opencl:1"]).
    gpu_allow: Option<Vec<String>>,

    /// Optional deny list of device keys.
    gpu_deny: Option<Vec<String>>,

    /// Optional list of device keys that should start disabled.
    gpu_start_disabled: Option<Vec<String>>,
}

#[cfg(feature = "prod-backend")]
const DEFAULT_BACKEND_URL: &str = "https://weso.forgeros.fr/";

#[cfg(not(feature = "prod-backend"))]
const DEFAULT_BACKEND_URL: &str = "http://127.0.0.1:8080";

fn default_backend_url() -> Url {
    if let Ok(v) = std::env::var("BBR_BACKEND_URL") {
        if let Ok(url) = Url::parse(v.trim()) {
            return url;
        }
    }
    Url::parse(DEFAULT_BACKEND_URL).expect("DEFAULT_BACKEND_URL must be a valid URL")
}

const GUI_PROGRESS_STEPS: u64 = 200;
const GUI_PROGRESS_TICK: Duration = Duration::from_millis(100);

fn apply_legacy_env_override(cfg: &mut GpuConfig) {
    // Legacy env override: WESOFORGE_GPU=cuda|opencl|auto|off
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

#[tauri::command]
async fn get_submitter_config() -> Result<Option<SubmitterConfig>, String> {
    load_submitter_config().map_err(|e| format!("{e:#}"))
}

#[tauri::command]
async fn set_submitter_config(cfg: SubmitterConfig) -> Result<(), String> {
    save_submitter_config(&cfg).map_err(|e| format!("{e:#}"))
}

#[tauri::command]
async fn engine_progress(state: State<'_, Arc<GuiState>>) -> Result<Vec<WorkerProgressUpdate>, String> {
    let progress = state.progress.lock().await;
    Ok(progress.clone())
}

#[tauri::command]
async fn start_client(app: AppHandle, state: State<'_, Arc<GuiState>>, opts: StartOptions) -> Result<(), String> {
    let mut guard = state.engine.lock().await;
    if guard.is_some() {
        return Err("already running".to_string());
    }

    let state_for_task = state.inner().clone();
    let submitter = match load_submitter_config() {
        Ok(Some(cfg)) => cfg,
        Ok(None) => SubmitterConfig::default(),
        Err(err) => return Err(format!("{err:#}")),
    };

    // cpu_workers selection:
    // - cpu_workers has priority
    // - then legacy "parallel"
    // - then default 4
    let cpu_workers = opts
        .cpu_workers
        .or(opts.parallel)
        .unwrap_or(4);

    if cpu_workers > 512 {
        return Err("cpu_workers must be between 0 and 512.".to_string());
    }
    let cpu_workers = cpu_workers as usize;

    let mut gpu_cfg = GpuConfig::default();

    if let Some(mode) = opts.gpu_mode.clone() {
        gpu_cfg.mode = mode.into();
    }
    if let Some(n) = opts.gpu_workers_per_device {
        if !(1..=64).contains(&n) {
            return Err("gpu_workers_per_device must be between 1 and 64.".to_string());
        }
        gpu_cfg.workers_per_device = n as usize;
    }
    if let Some(v) = opts.gpu_allow.clone() {
        gpu_cfg.allow = v;
    }
    if let Some(v) = opts.gpu_deny.clone() {
        gpu_cfg.deny = v;
    }
    if let Some(v) = opts.gpu_start_disabled.clone() {
        gpu_cfg.start_disabled = v;
    }

    // Keep env override for compatibility with existing scripts.
    apply_legacy_env_override(&mut gpu_cfg);

    let engine = start_engine(EngineConfig {
        backend_url: default_backend_url(),
        parallel: cpu_workers,
        gpu: gpu_cfg,
        mem_budget_bytes: 128 * 1024 * 1024,
        submitter,
        idle_sleep: Duration::ZERO,
        progress_steps: GUI_PROGRESS_STEPS,
        progress_tick: GUI_PROGRESS_TICK,
        recent_jobs_max: EngineConfig::DEFAULT_RECENT_JOBS_MAX,
    });

    let mut events = engine.subscribe();
    let app = app.clone();

    {
        let mut progress = state.progress.lock().await;
        progress.clear();
        progress.reserve(cpu_workers);
        for worker_idx in 0..cpu_workers {
            progress.push(WorkerProgressUpdate {
                worker_idx,
                iters_done: 0,
                iters_total: 0,
                iters_per_sec: 0,
            });
        }
    }

    tokio::spawn(async move {
        loop {
            let ev = match events.recv().await {
                Ok(ev) => ev,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
            };

            match &ev {
                EngineEvent::WorkerProgress {
                    worker_idx,
                    iters_done,
                    iters_total,
                    iters_per_sec,
                } => {
                    let mut progress = state_for_task.progress.lock().await;
                    while progress.len() <= *worker_idx {
                        let idx = progress.len();
                        progress.push(WorkerProgressUpdate {
                            worker_idx: idx,
                            iters_done: 0,
                            iters_total: 0,
                            iters_per_sec: 0,
                        });
                    }
                    progress[*worker_idx] = WorkerProgressUpdate {
                        worker_idx: *worker_idx,
                        iters_done: *iters_done,
                        iters_total: *iters_total,
                        iters_per_sec: *iters_per_sec,
                    };
                }
                EngineEvent::WorkerJobStarted { worker_idx, job } => {
                    {
                        let mut progress = state_for_task.progress.lock().await;
                        while progress.len() <= *worker_idx {
                            let idx = progress.len();
                            progress.push(WorkerProgressUpdate {
                                worker_idx: idx,
                                iters_done: 0,
                                iters_total: 0,
                                iters_per_sec: 0,
                            });
                        }
                        progress[*worker_idx] = WorkerProgressUpdate {
                            worker_idx: *worker_idx,
                            iters_done: 0,
                            iters_total: job.number_of_iterations,
                            iters_per_sec: 0,
                        };
                    }
                    let _ = app.emit("engine-event", ev);
                }
                EngineEvent::JobFinished { outcome } => {
                    let worker_idx = outcome.worker_idx;
                    {
                        let mut progress = state_for_task.progress.lock().await;
                        while progress.len() <= worker_idx {
                            let idx = progress.len();
                            progress.push(WorkerProgressUpdate {
                                worker_idx: idx,
                                iters_done: 0,
                                iters_total: 0,
                                iters_per_sec: 0,
                            });
                        }
                        progress[worker_idx] = WorkerProgressUpdate {
                            worker_idx,
                            iters_done: 0,
                            iters_total: 0,
                            iters_per_sec: 0,
                        };
                    }
                    let _ = app.emit("engine-event", ev);
                }
                EngineEvent::Error { message } => {
                    eprintln!("{message}");
                    let _ = app.emit("engine-event", ev);
                }
                _ => {
                    let is_stopped = matches!(ev, EngineEvent::Stopped);
                    let _ = app.emit("engine-event", ev);
                    if is_stopped {
                        break;
                    }
                }
            }
        }

        let mut guard = state_for_task.engine.lock().await;
        *guard = None;

        let mut progress = state_for_task.progress.lock().await;
        progress.clear();
    });

    *guard = Some(engine);
    Ok(())
}

#[tauri::command]
async fn stop_client(state: State<'_, Arc<GuiState>>) -> Result<(), String> {
    let guard = state.engine.lock().await;
    let Some(engine) = guard.as_ref() else {
        return Ok(());
    };
    engine.request_stop();
    Ok(())
}

#[tauri::command]
async fn client_running(state: State<'_, Arc<GuiState>>) -> Result<bool, String> {
    let guard = state.engine.lock().await;
    Ok(guard.is_some())
}

#[tauri::command]
async fn engine_snapshot(state: State<'_, Arc<GuiState>>) -> Result<Option<StatusSnapshot>, String> {
    let guard = state.engine.lock().await;
    Ok(guard.as_ref().map(|engine| engine.snapshot()))
}

#[tauri::command]
async fn set_worker_enabled(
    state: State<'_, Arc<GuiState>>,
    worker_idx: usize,
    enabled: bool,
) -> Result<(), String> {
    let guard = state.engine.lock().await;
    let Some(engine) = guard.as_ref() else {
        return Err("not running".to_string());
    };
    engine.set_worker_enabled(worker_idx, enabled);
    Ok(())
}

#[tauri::command]
async fn set_gpu_enabled(
    state: State<'_, Arc<GuiState>>,
    device_key: String,
    enabled: bool,
) -> Result<(), String> {
    let guard = state.engine.lock().await;
    let Some(engine) = guard.as_ref() else {
        return Err("not running".to_string());
    };
    engine.set_gpu_enabled(&device_key, enabled);
    Ok(())
}

fn main() {
    #[cfg(target_os = "linux")]
    {
        // On some Linux/Wayland setups WebKitGTK's DMABUF renderer can fail with errors like:
        // "Failed to create GBM buffer ... Invalid argument" and render a blank window.
        // Default to disabling it unless the user explicitly opted in.
        if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
            // SAFETY: this is executed at process startup before spawning any threads.
            unsafe {
                std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
            }
        }
    }

    let state = Arc::new(GuiState::default());
    tauri::Builder::default()
        .manage(state)
        .setup(|app| {
            let _ = app;
            #[cfg(feature = "support-devtools")]
            {
                if let Some(win) = app.get_webview_window("main") {
                    win.open_devtools();
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_submitter_config,
            set_submitter_config,
            engine_progress,
            start_client,
            stop_client,
            client_running,
            engine_snapshot,
            set_worker_enabled,
            set_gpu_enabled
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
