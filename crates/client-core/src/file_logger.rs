use chrono::Local;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// Best-effort file logger for runtime diagnostics.
///
/// Default output directory:
/// - `./logs/` (relative to the current working directory)
///
/// Environment variables:
/// - `BBR_LOG_FILE`: if set, logs are appended to this file (overrides the default path).
/// - `BBR_LOG_DIR`: if set, log files are created inside this directory (ignored when `BBR_LOG_FILE` is set).
/// - `BBR_LOG_PROGRESS=1`: if set, per-worker progress lines are logged (rate-limited).
#[derive(Debug)]
pub struct FileLogger {
    file: Option<File>,
    log_progress: bool,
    last_progress_ms: Vec<u128>,
}

impl FileLogger {
    pub fn new(app_tag: &str, parallel: usize) -> Self {
        let log_progress = matches!(
            std::env::var("BBR_LOG_PROGRESS").ok().as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        );

        let file = Self::open_from_env_or_default(app_tag);

        Self {
            file,
            log_progress,
            last_progress_ms: vec![0; parallel],
        }
    }

    pub fn enabled(&self) -> bool {
        self.file.is_some()
    }

    pub fn log_line(&mut self, line: &str) {
        let Some(f) = self.file.as_mut() else {
            return;
        };

        // Best-effort logging: ignore errors so we don't affect runtime.
        let _ = writeln!(f, "{line}");
        let _ = f.flush();
    }

    pub fn log_engine_started(&mut self, parallel: usize, use_groups: bool, mem_budget_bytes: u64) {
        if !self.enabled() {
            return;
        }
        self.log_line(&format!(
            "[engine] started | parallel={parallel} | mode={} | mem_budget_bytes={mem_budget_bytes}",
            if use_groups { "group" } else { "proof" }
        ));
    }

    pub fn log_job_started(
        &mut self,
        worker_idx: usize,
        job_id: u64,
        height: impl std::fmt::Display,
        field_vdf: impl std::fmt::Display,
        iterations: impl std::fmt::Display,
    ) {
        if !self.enabled() {
            return;
        }
        self.log_line(&format!(
            "[job] started | worker={worker_idx} | job_id={job_id} | height={height} | field_vdf={field_vdf} | iterations={iterations}"
        ));
    }

    pub fn log_job_finished(
        &mut self,
        worker_idx: usize,
        job_id: u64,
        height: impl std::fmt::Display,
        field_vdf: impl std::fmt::Display,
        total_ms: u64,
        submit_reason: &str,
        error_present: bool,
        output_mismatch: bool,
    ) {
        if !self.enabled() {
            return;
        }

        self.log_line(&format!(
            "[job] finished | worker={worker_idx} | job_id={job_id} | height={height} | field_vdf={field_vdf} | total_ms={total_ms} | submit_reason={submit_reason} | error_present={error_present} | output_mismatch={output_mismatch}"
        ));
    }

    pub fn log_warning(&mut self, message: &str) {
        if !self.enabled() {
            return;
        }
        self.log_line(&format!("[warn] {message}"));
    }

    pub fn log_error(&mut self, message: &str) {
        if !self.enabled() {
            return;
        }
        self.log_line(&format!("[error] {message}"));
    }

    pub fn log_worker_progress_rate_limited(
        &mut self,
        worker_idx: usize,
        iters_done: u64,
        iters_total: u64,
        iters_per_sec: u64,
    ) {
        if !self.enabled() || !self.log_progress {
            return;
        }

        // Rate-limit to ~1 line/sec/worker.
        let now = Self::now_ms();
        if worker_idx >= self.last_progress_ms.len() {
            self.last_progress_ms.resize(worker_idx + 1, 0);
        }
        if now.saturating_sub(self.last_progress_ms[worker_idx]) < 1_000 {
            return;
        }
        self.last_progress_ms[worker_idx] = now;

        self.log_line(&format!(
            "[worker] progress | worker={worker_idx} | {iters_done}/{iters_total} | it_per_s={iters_per_sec}",
        ));
    }

    fn now_ms() -> u128 {
        // `std::time::SystemTime` is enough here; monotonicity isn't required for rate-limiting.
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    }

    fn open_from_env_or_default(app_tag: &str) -> Option<File> {
        // Explicit file path override.
        if let Ok(v) = std::env::var("BBR_LOG_FILE") {
            let p = v.trim();
            if p.is_empty() {
                return None;
            }
            return OpenOptions::new().create(true).append(true).open(p).ok();
        }

        let dir = std::env::var("BBR_LOG_DIR")
            .ok()
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("logs"));

        Self::open_default_in_dir(&dir, app_tag)
    }

    fn open_default_in_dir(dir: &Path, app_tag: &str) -> Option<File> {
        let _ = std::fs::create_dir_all(dir);

        let ts = Local::now().format("%Y%m%d-%H%M%S");
        let pid = std::process::id();
        let filename = format!("{app_tag}-{ts}-{pid}.log");
        let path = dir.join(filename);

        OpenOptions::new().create(true).append(true).open(path).ok()
    }
}
