use chrono::Local;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// Centralized runtime file logger.
///
/// Default directory: `<repo_root>/logs/` where `repo_root` is detected by searching for `Cargo.toml`.
///
/// Environment variables:
/// - `BBR_LOG_FILE`: if set, logs are appended to this file (overrides directory logic). Empty disables logging.
/// - `BBR_LOG_DIR`: if set, log files are created inside this directory (ignored when `BBR_LOG_FILE` is set).
/// - `BBR_LOG_PROGRESS=1`: enable periodic progress lines (call sites rate-limit).
#[derive(Debug)]
pub struct Logger {
    file: Mutex<std::fs::File>,
    log_progress: bool,
}

static LOGGER: OnceLock<Logger> = OnceLock::new();

impl Logger {
    /// Initialize the global logger once. Subsequent calls are no-ops.
    pub fn init(app_tag: &str) {
        let _ = Self::try_init(app_tag);
    }

    /// Try to initialize the global logger; returns `true` if logging is enabled.
    pub fn try_init(app_tag: &str) -> bool {
        if LOGGER.get().is_some() {
            return true;
        }

        let log_progress = matches!(
            std::env::var("BBR_LOG_PROGRESS").ok().as_deref(),
            Some("1") | Some("true") | Some("yes") | Some("on")
        );

        let Some(file) = open_log_file(app_tag) else {
            return false;
        };

        let _ = LOGGER.set(Logger {
            file: Mutex::new(file),
            log_progress,
        });

        LOGGER.get().is_some()
    }

    /// Returns the global logger if enabled.
    pub fn global() -> Option<&'static Logger> {
        LOGGER.get()
    }

    pub fn log_progress_enabled(&self) -> bool {
        self.log_progress
    }

    pub fn line(&self, line: &str) {
        if let Ok(mut f) = self.file.lock() {
            let _ = writeln!(f, "{line}");
            let _ = f.flush();
        }
    }

    pub fn warn(&self, msg: &str) {
        self.line(&format!("[warn] {msg}"));
    }

    pub fn error(&self, msg: &str) {
        self.line(&format!("[error] {msg}"));
    }

    pub fn engine_started(&self, app: &str, parallel: usize, mode: &str, mem_budget_bytes: u64) {
        self.line(&format!(
            "[{app}] engine_started | parallel={parallel} | mode={mode} | mem_budget_bytes={mem_budget_bytes}"
        ));
    }

    pub fn job_started(
        &self,
        worker_idx: usize,
        job_id: u64,
        height: impl std::fmt::Display,
        field_vdf: impl std::fmt::Display,
        iterations: impl std::fmt::Display,
    ) {
        self.line(&format!(
            "[job] started | worker={worker_idx} | job_id={job_id} | height={height} | field_vdf={field_vdf} | iterations={iterations}"
        ));
    }

    pub fn job_finished(
        &self,
        worker_idx: usize,
        job_id: u64,
        height: impl std::fmt::Display,
        field_vdf: impl std::fmt::Display,
        total_ms: u64,
        submit_reason: &str,
        error_present: bool,
        output_mismatch: bool,
    ) {
        self.line(&format!(
            "[job] finished | worker={worker_idx} | job_id={job_id} | height={height} | field_vdf={field_vdf} | total_ms={total_ms} | submit_reason={submit_reason} | error_present={error_present} | output_mismatch={output_mismatch}"
        ));
    }
}

fn open_log_file(app_tag: &str) -> Option<std::fs::File> {
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
        .unwrap_or_else(default_logs_dir);

    let _ = std::fs::create_dir_all(&dir);

    let ts = Local::now().format("%Y%m%d-%H%M%S");
    let pid = std::process::id();
    let filename = format!("{app_tag}-{ts}-{pid}.log");
    let path = dir.join(filename);

    OpenOptions::new().create(true).append(true).open(path).ok()
}

fn default_logs_dir() -> PathBuf {
    // Prefer finding a repo root (Cargo.toml) from current dir, otherwise from current exe.
    if let Ok(cwd) = std::env::current_dir() {
        if let Some(root) = find_repo_root(&cwd) {
            return root.join("logs");
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            if let Some(root) = find_repo_root(dir) {
                return root.join("logs");
            }
        }
    }

    PathBuf::from("logs")
}

fn find_repo_root(start: &Path) -> Option<PathBuf> {
    for p in start.ancestors() {
        let cargo = p.join("Cargo.toml");
        let crates = p.join("crates");
        if cargo.is_file() && crates.is_dir() {
            return Some(p.to_path_buf());
        }
    }
    None
}
