use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::backend::BackendJobDto;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct InflightJobEntry {
    pub(crate) lease_id: String,
    pub(crate) lease_expires_at: i64,
    pub(crate) job: BackendJobDto,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct InflightFile {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    jobs: Vec<InflightJobEntry>,
}

pub(crate) struct InflightStore {
    path: PathBuf,
    jobs_by_id: BTreeMap<u64, InflightJobEntry>,
}

impl InflightStore {
    pub(crate) fn load() -> anyhow::Result<Option<Self>> {
        let path = inflight_path()?;

        // If the file doesn't exist, we still return an empty store: resume is "enabled",
        // there is simply nothing to restore.
        if !path.exists() {
            return Ok(Some(Self {
                path,
                jobs_by_id: BTreeMap::new(),
            }));
        }

        let raw = std::fs::read_to_string(&path)?;
        let file: InflightFile = serde_json::from_str(&raw)?;

        let mut jobs_by_id = BTreeMap::new();
        for entry in file.jobs {
            jobs_by_id.insert(entry.job.job_id, entry);
        }

        Ok(Some(Self { path, jobs_by_id }))
    }

    pub(crate) fn entries(&self) -> impl Iterator<Item = &InflightJobEntry> {
        self.jobs_by_id.values()
    }

    pub(crate) async fn persist(&self) -> anyhow::Result<()> {
        let path = self.path.clone();
        let file = InflightFile {
            version: 1,
            jobs: self.jobs_by_id.values().cloned().collect(),
        };

        // Write on a blocking thread because it may touch the filesystem.
        tokio::task::spawn_blocking(move || persist_file(&path, &file))
            .await
            .map_err(|err| anyhow::anyhow!("persist inflight leases: {err:#}"))??;

        Ok(())
    }

    /// Return the number of inflight jobs currently tracked.
    pub(crate) fn len(&self) -> usize {
        self.jobs_by_id.len()
    }
}

fn persist_file(path: &Path, file: &InflightFile) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let dir = path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("invalid inflight path: no parent directory"))?;

    let tmp_path = dir.join(format!(
        ".{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("inflight-leases.json")
    ));

    let payload = serde_json::to_vec_pretty(file)?;

    // Best-effort atomic write: write to a temporary file in the same directory, then rename.
    std::fs::write(&tmp_path, payload)?;

    // On Windows, rename fails if the destination exists.
    #[cfg(windows)]
    {
        let _ = std::fs::remove_file(path);
    }

    std::fs::rename(&tmp_path, path)?;

    Ok(())
}

fn xdg_state_home() -> anyhow::Result<PathBuf> {
    if let Some(dir) = std::env::var_os("XDG_STATE_HOME") {
        let dir = PathBuf::from(dir);
        if dir.as_os_str().is_empty() {
            anyhow::bail!("XDG_STATE_HOME is set but empty");
        }
        return Ok(dir);
    }

    // On Windows we should not require HOME to be set: the conventional location
    // for per-user application state is %LOCALAPPDATA%.
    #[cfg(windows)]
    {
        if let Some(dir) = std::env::var_os("LOCALAPPDATA") {
            let dir = PathBuf::from(dir);
            if dir.as_os_str().is_empty() {
                anyhow::bail!("LOCALAPPDATA is set but empty");
            }
            return Ok(dir);
        }

        // Some restricted environments might not set LOCALAPPDATA; fall back to APPDATA.
        if let Some(dir) = std::env::var_os("APPDATA") {
            let dir = PathBuf::from(dir);
            if dir.as_os_str().is_empty() {
                anyhow::bail!("APPDATA is set but empty");
            }
            return Ok(dir);
        }

        // Fall back to USERPROFILE if available.
        if let Some(dir) = std::env::var_os("USERPROFILE") {
            let dir = PathBuf::from(dir);
            if dir.as_os_str().is_empty() {
                anyhow::bail!("USERPROFILE is set but empty");
            }
            return Ok(dir.join("AppData").join("Local"));
        }
    }

    let home = std::env::var_os("HOME").ok_or_else(|| anyhow::anyhow!("HOME is not set"))?;
    let home = PathBuf::from(home);
    if home.as_os_str().is_empty() {
        anyhow::bail!("HOME is set but empty");
    }
    Ok(home.join(".local").join("state"))
}

fn inflight_path() -> anyhow::Result<PathBuf> {
    Ok(xdg_state_home()?
        .join("bbr-client")
        .join("inflight-leases.json"))
}
