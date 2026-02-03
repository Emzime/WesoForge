// Comments in English as requested.

use cfg_if::cfg_if;

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
    cfg_if! {
        if #[cfg(any(target_os = "windows", target_os = "linux"))] {
            let mut cores: Vec<usize> = core_affinity::get_core_ids()
                .unwrap_or_default()
                .into_iter()
                .map(|c| c.id)
                .collect();

            // Deterministic ordering.
            cores.sort_unstable();

            if policy.reserve_core0 {
                cores.retain(|&id| id != 0);
            }

            if policy.reversed {
                cores.reverse();
            }

            cores
        } else {
            Vec::new()
        }
    }
}

/// Attempts to pin the current thread to the core assigned for `worker_idx`.
///
/// Returns:
/// - Ok(Some(core_id)) when pinning was applied,
/// - Ok(None) when pinning is not supported or no usable cores are available.
pub fn pin_current_thread(worker_idx: usize, policy: CpuPinPolicy) -> Result<Option<usize>, String> {
    cfg_if! {
        if #[cfg(any(target_os = "windows", target_os = "linux"))] {
            let cores = usable_core_ids(policy);
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
            Ok(None)
        }
    }
}
