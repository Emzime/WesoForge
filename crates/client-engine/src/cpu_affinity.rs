// Comments in English as requested.

use cfg_if::cfg_if;

/// Parse a CPU core list specification.
///
/// Supported formats:
/// - "2,3,6" (comma-separated)
/// - "10-15" (inclusive ranges)
/// - "2,3,10-15" (mixed)
///
/// Returns a sorted, de-duplicated list of core IDs.
pub fn parse_core_list(spec: &str) -> Result<Vec<usize>, String> {
    let s = spec.trim();
    if s.is_empty() {
        return Err("core list must not be empty".to_string());
    }

    let mut out: Vec<usize> = Vec::new();
    for part in s.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let a = a.trim();
            let b = b.trim();
            if a.is_empty() || b.is_empty() {
                return Err(format!("invalid core range: {part:?}"));
            }
            let start: usize = a
                .parse()
                .map_err(|_| format!("invalid core id in range: {part:?}"))?;
            let end: usize = b
                .parse()
                .map_err(|_| format!("invalid core id in range: {part:?}"))?;
            let (lo, hi) = if start <= end { (start, end) } else { (end, start) };
            for id in lo..=hi {
                out.push(id);
            }
        } else {
            let id: usize = part
                .parse()
                .map_err(|_| format!("invalid core id: {part:?}"))?;
            out.push(id);
        }
    }

    if out.is_empty() {
        return Err("core list must not be empty".to_string());
    }

    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// Build the effective core list from system cores + policy + allow/block lists.
///
/// # Notes
/// - `allowlist` has priority: when present, only those cores are considered.
/// - `reserve_core0` is applied only when allowlist is NOT present.
///   (If the user explicitly includes 0 in allowlist, we respect it.)
/// - `blocklist` is always applied as a final filter.
pub fn effective_core_ids(
    policy: CpuPinPolicy,
    allowlist: Option<&[usize]>,
    blocklist: Option<&[usize]>,
) -> Vec<usize> {
    cfg_if! {
        if #[cfg(any(target_os = "windows", target_os = "linux"))] {
            let mut cores: Vec<usize> = core_affinity::get_core_ids()
                .unwrap_or_default()
                .into_iter()
                .map(|c| c.id)
                .collect();

            cores.sort_unstable();

            if let Some(allow) = allowlist {
                let allow_set: std::collections::BTreeSet<usize> = allow.iter().copied().collect();
                cores.retain(|id| allow_set.contains(id));
            } else if policy.reserve_core0 {
                cores.retain(|&id| id != 0);
            }

            if let Some(block) = blocklist {
                let block_set: std::collections::BTreeSet<usize> = block.iter().copied().collect();
                cores.retain(|id| !block_set.contains(id));
            }

            if policy.reversed {
                cores.reverse();
            }

            cores
        } else {
            let _ = policy;
            let _ = allowlist;
            let _ = blocklist;
            Vec::new()
        }
    }
}

/// CPU pinning policy:
/// - exclude logical core 0 (reserve for OS)
/// - assign workers in reverse core order (last -> ... -> 1)
#[derive(Debug, Clone, Copy)]
pub struct CpuPinPolicy {
    /// If true, core 0 is never used.
    pub reserve_core0: bool,
    /// If true, cores are assigned in reverse order (last -> ... -> 1).
    pub reversed: bool,
}

impl Default for CpuPinPolicy {
    fn default() -> Self {
        Self {
            reserve_core0: true,
            reversed: true,
        }
    }
}

/// Returns the list of usable core IDs according to the policy.
///
/// On unsupported platforms (e.g. macOS), returns an empty vector.
pub fn usable_core_ids(policy: CpuPinPolicy) -> Vec<usize> {
    effective_core_ids(policy, None, None)
}

/// Attempts to pin the current thread to the core assigned for `worker_idx`.
///
/// Returns:
/// - Ok(Some(core_id)) when pinning was applied,
/// - Ok(None) when pinning is not supported or no usable cores are available.
pub fn pin_current_thread(worker_idx: usize, policy: CpuPinPolicy) -> Result<Option<usize>, String> {
    pin_current_thread_with_lists(worker_idx, policy, None, None)
}

/// Attempts to pin the current thread to the core assigned for `worker_idx`,
/// using optional allowlist/blocklist.
pub fn pin_current_thread_with_lists(
    worker_idx: usize,
    policy: CpuPinPolicy,
    allowlist: Option<&[usize]>,
    blocklist: Option<&[usize]>,
) -> Result<Option<usize>, String> {
    cfg_if! {
        if #[cfg(any(target_os = "windows", target_os = "linux"))] {
            let cores = effective_core_ids(policy, allowlist, blocklist);
            if cores.is_empty() {
                return Ok(None);
            }

            let core_id = cores[worker_idx % cores.len()];

            // core_affinity requires a CoreId object; resolve it by id.
            let core_ids = core_affinity::get_core_ids().unwrap_or_default();
            let chosen = core_ids.into_iter().find(|c| c.id == core_id);

            match chosen {
                Some(cid) => {
                    core_affinity::set_for_current(cid);
                    Ok(Some(core_id))
                }
                None => Ok(None),
            }
        } else {
            let _ = worker_idx;
            let _ = policy;
            let _ = allowlist;
            let _ = blocklist;
            Ok(None)
        }
    }
}
