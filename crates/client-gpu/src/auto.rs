//! Automatic GPU backend selection and fallback.

use bbr_client_core::logging::Logger;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Condvar, Mutex,
};

use crate::{GpuBackend, GpuConfig, GpuDeviceSelection, GpuProbe};

/// Preferred GPU order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPreference {
    /// Prefer CUDA when available, then OpenCL.
    CudaFirst,
    /// Prefer OpenCL when available, then CUDA.
    OpenClFirst,
}

impl Default for GpuPreference {
    fn default() -> Self {
        Self::CudaFirst
    }
}

/// Read GPU runtime configuration from environment variables.
pub fn config_from_env() -> GpuConfig {
    GpuConfig::from_env()
}

fn gpu_globally_disabled(cfg: &GpuConfig) -> bool {
    cfg.is_disabled()
}

fn env_flag(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

fn parse_u32_list_csv(s: &str) -> Vec<u32> {
    s.split(',')
        .map(|t| t.trim())
        .filter(|t| !t.is_empty())
        .filter_map(|t| t.parse::<u32>().ok())
        .collect()
}

/// Optional: allow simulating multiple GPUs on a machine with a single GPU.
///
/// If set, this value is used as the "available devices" list for selection.
/// Example:
/// - `BBR_GPU_DEVICES_VIRTUAL=0,1,2,3`
fn virtual_devices_from_env() -> Option<Vec<u32>> {
    let v = std::env::var("BBR_GPU_DEVICES_VIRTUAL").ok()?;
    let v = v.trim();
    if v.is_empty() {
        return None;
    }
    let list = parse_u32_list_csv(v);
    if list.is_empty() {
        None
    } else {
        Some(list)
    }
}

fn sched_log(line: &str) {
    Logger::try_init("gpu");
    if let Some(l) = Logger::global() {
        l.line(line);
    }
}

/// Probe a GPU backend without initializing long-lived resources.
///
/// Note: if the user disabled GPU usage via `BBR_GPU_DEVICES=none`, probing will return unavailable.
pub fn probe(backend: GpuBackend) -> GpuProbe {
    let cfg = config_from_env();
    if gpu_globally_disabled(&cfg) {
        return GpuProbe::unavailable("GPU disabled by user (BBR_GPU_DEVICES=none)");
    }

    match backend {
        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::probe_with_config(&cfg)
            }
            #[cfg(not(feature = "cuda"))]
            {
                GpuProbe::unavailable("cuda feature not enabled")
            }
        }
        GpuBackend::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                crate::opencl::probe_with_config(&cfg)
            }
            #[cfg(not(feature = "opencl"))]
            {
                GpuProbe::unavailable("opencl feature not enabled")
            }
        }
    }
}

/// Pick the best available GPU backend following a preference order.
///
/// Returns `None` if no GPU backend is available or if GPU usage is disabled.
pub fn pick_best(preference: GpuPreference) -> Option<GpuBackend> {
    let cfg = config_from_env();
    pick_best_with_config(preference, &cfg)
}

/// Same as `pick_best`, but uses an explicit configuration.
pub fn pick_best_with_config(preference: GpuPreference, cfg: &GpuConfig) -> Option<GpuBackend> {
    if gpu_globally_disabled(cfg) {
        return None;
    }

    let candidates: [GpuBackend; 2] = match preference {
        GpuPreference::CudaFirst => [GpuBackend::Cuda, GpuBackend::OpenCl],
        GpuPreference::OpenClFirst => [GpuBackend::OpenCl, GpuBackend::Cuda],
    };

    for b in candidates {
        let p = probe_with_config(b, cfg);
        if p.available {
            return Some(b);
        }
    }

    None
}

/// Probe a backend using an explicit configuration.
pub fn probe_with_config(backend: GpuBackend, cfg: &GpuConfig) -> GpuProbe {
    if gpu_globally_disabled(cfg) {
        return GpuProbe::unavailable("GPU disabled by user (BBR_GPU_DEVICES=none)");
    }

    match backend {
        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::probe_with_config(cfg)
            }
            #[cfg(not(feature = "cuda"))]
            {
                GpuProbe::unavailable("cuda feature not enabled")
            }
        }
        GpuBackend::OpenCl => {
            #[cfg(feature = "opencl")]
            {
                crate::opencl::probe_with_config(cfg)
            }
            #[cfg(not(feature = "opencl"))]
            {
                GpuProbe::unavailable("opencl feature not enabled")
            }
        }
    }
}

/// Compute the set of device ordinals to use for a backend based on the configuration.
///
/// If `BBR_GPU_DEVICES_VIRTUAL` is set, it is used as the "available devices" list when
/// `GpuDeviceSelection::All` is active. This enables multi-GPU scheduling tests on a
/// single-GPU machine.
pub fn selected_devices_for_backend(backend: GpuBackend, cfg: &GpuConfig) -> Vec<u32> {
    if let Some(virtual_devices) = virtual_devices_from_env() {
        return match &cfg.devices {
            GpuDeviceSelection::None => Vec::new(),
            GpuDeviceSelection::List(list) => list.clone(),
            GpuDeviceSelection::All => virtual_devices,
        };
    }

    match &cfg.devices {
        GpuDeviceSelection::None => Vec::new(),
        GpuDeviceSelection::List(list) => list.clone(),
        GpuDeviceSelection::All => match backend {
            GpuBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    return crate::cuda::list_device_ordinals().unwrap_or_default();
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Vec::new();
                }
            }
            GpuBackend::OpenCl => {
                #[cfg(feature = "opencl")]
                {
                    return crate::opencl::list_device_indices().unwrap_or_default();
                }
                #[cfg(not(feature = "opencl"))]
                {
                    return Vec::new();
                }
            }
        },
    }
}

/// Bucketing policy for job-level micro-batching.
#[derive(Debug, Clone)]
pub struct BucketPolicy {
    pub s_max: u64,
    pub m_max: u64,
    pub l_max: u64,
    pub micro_batch_s: usize,
    pub micro_batch_m: usize,
    pub micro_batch_l: usize,
    pub micro_batch_xl: usize,
}

impl Default for BucketPolicy {
    fn default() -> Self {
        Self {
            s_max: 220_000_000,
            m_max: 320_000_000,
            l_max: 450_000_000,
            micro_batch_s: 8,
            micro_batch_m: 4,
            micro_batch_l: 2,
            micro_batch_xl: 1,
        }
    }
}

impl BucketPolicy {
    pub fn bucket_for_cost(&self, cost: u64) -> usize {
        if cost < self.s_max {
            0
        } else if cost < self.m_max {
            1
        } else if cost < self.l_max {
            2
        } else {
            3
        }
    }
}

/// A micro-batch of jobs to be run on a single device.
#[derive(Debug, Clone)]
pub struct JobBatch {
    pub bucket: u8,
    pub job_indices: Vec<usize>,
    pub total_cost: u64,
}

/// A planned micro-batch assigned to a specific device ordinal.
#[derive(Debug, Clone)]
pub struct ScheduledJobBatch {
    pub device_ordinal: u32,
    pub batch: JobBatch,
}

/// A small counting semaphore implemented with `Mutex` + `Condvar`.
#[derive(Debug)]
struct PermitPool {
    state: Mutex<PermitState>,
    cv: Condvar,
}

#[derive(Debug)]
struct PermitState {
    permits: usize,
}

impl PermitPool {
    fn new(permits: usize) -> Self {
        Self {
            state: Mutex::new(PermitState { permits }),
            cv: Condvar::new(),
        }
    }

    fn try_acquire(&self) -> Option<PermitGuard<'_>> {
        let mut st = self.state.lock().expect("mutex poisoned");
        if st.permits == 0 {
            return None;
        }
        st.permits -= 1;
        Some(PermitGuard { pool: self })
    }

    fn acquire_blocking(&self) -> PermitGuard<'_> {
        let mut st = self.state.lock().expect("mutex poisoned");
        while st.permits == 0 {
            st = self.cv.wait(st).expect("condvar wait failed");
        }
        st.permits -= 1;
        PermitGuard { pool: self }
    }

    fn release(&self) {
        let mut st = self.state.lock().expect("mutex poisoned");
        st.permits += 1;
        self.cv.notify_one();
    }
}

struct PermitGuard<'a> {
    pool: &'a PermitPool,
}

impl Drop for PermitGuard<'_> {
    fn drop(&mut self) {
        self.pool.release();
    }
}

/// Multi-GPU scheduler for job-level micro-batching.
#[derive(Debug)]
pub struct MultiGpuScheduler {
    #[allow(dead_code)]
    backend: GpuBackend,
    devices: Vec<u32>,
    rr: AtomicUsize,
    failed: Mutex<Vec<bool>>,
    per_device_permits: Vec<Arc<PermitPool>>,
    debug: bool,
}

impl MultiGpuScheduler {
    pub fn new(backend: GpuBackend, cfg: &GpuConfig) -> anyhow::Result<Self> {
        if cfg.is_disabled() {
            anyhow::bail!("GPU disabled by user (BBR_GPU_DEVICES=none)");
        }

        let devices = selected_devices_for_backend(backend, cfg);
        if devices.is_empty() {
            anyhow::bail!("no GPU devices selected");
        }

        let device_count = devices.len();
        let streams = cfg.streams.unwrap_or(1).max(1) as usize;
        let per_device_permits = (0..device_count)
            .map(|_| Arc::new(PermitPool::new(streams)))
            .collect::<Vec<_>>();

        let debug = env_flag("BBR_GPU_SCHED_DEBUG");

        if debug {
            sched_log(&format!(
                "[sched] init backend={:?} devices={:?} streams_per_device={}",
                backend, devices, streams
            ));
            if let Some(v) = virtual_devices_from_env() {
                sched_log(&format!("[sched] virtual devices active: {:?}", v));
            }
        }

        Ok(Self {
            backend,
            devices,
            rr: AtomicUsize::new(0),
            failed: Mutex::new(vec![false; device_count]),
            per_device_permits,
            debug,
        })
    }

    pub fn device_ordinals(&self) -> &[u32] {
        &self.devices
    }

    pub fn usable_device_count(&self) -> usize {
        let failed = self.failed.lock().expect("mutex poisoned");
        failed.iter().filter(|&&f| !f).count()
    }

    pub fn mark_failed_ordinal(&self, ordinal: u32) {
        if let Some(idx) = self.devices.iter().position(|&d| d == ordinal) {
            let mut failed = self.failed.lock().expect("mutex poisoned");
            if let Some(v) = failed.get_mut(idx) {
                *v = true;
                if self.debug {
                    sched_log(&format!("[sched] mark_failed device={}", ordinal));
                }
            }
        }
    }

    fn try_acquire_any(&self) -> Option<(usize, PermitGuard<'_>)> {
        let n = self.devices.len();
        if n == 0 {
            return None;
        }

        let start = self.rr.fetch_add(1, Ordering::Relaxed) % n;
        let failed = self.failed.lock().expect("mutex poisoned");

        for offset in 0..n {
            let idx = (start + offset) % n;
            if failed[idx] {
                continue;
            }
            if let Some(g) = self.per_device_permits[idx].try_acquire() {
                return Some((idx, g));
            }
        }
        None
    }

    fn acquire_any_blocking(&self) -> anyhow::Result<(usize, PermitGuard<'_>)> {
        if let Some(v) = self.try_acquire_any() {
            return Ok(v);
        }

        let n = self.devices.len();
        if n == 0 {
            anyhow::bail!("no GPU devices configured");
        }

        let start = self.rr.fetch_add(1, Ordering::Relaxed) % n;
        let failed = self.failed.lock().expect("mutex poisoned");
        let mut picked: Option<usize> = None;
        for offset in 0..n {
            let idx = (start + offset) % n;
            if !failed[idx] {
                picked = Some(idx);
                break;
            }
        }
        drop(failed);

        let idx = picked.ok_or_else(|| anyhow::anyhow!("no usable GPU device"))?;
        let g = self.per_device_permits[idx].acquire_blocking();
        Ok((idx, g))
    }

    pub fn run_with_cpu_fallback<T, FGpu, FCpu>(
        &self,
        mut gpu: FGpu,
        cpu: FCpu,
        allow_fallback: bool,
    ) -> anyhow::Result<(Option<u32>, T)>
    where
        FGpu: FnMut(u32) -> anyhow::Result<T>,
        FCpu: FnOnce() -> anyhow::Result<T>,
    {
        fn is_not_implemented(err: &anyhow::Error) -> bool {
            err.chain()
                .any(|e| e.to_string().contains("BBR_GPU_NOT_IMPLEMENTED"))
        }

        let n = self.devices.len();
        if n > 0 {
            for _ in 0..n {
                let (idx, _permit) = match self.acquire_any_blocking() {
                    Ok(v) => v,
                    Err(err) => {
                        if self.debug {
                            sched_log(&format!("[sched] acquire device failed: {:#}", err));
                        }
                        if allow_fallback {
                            if self.debug {
                                sched_log("[sched] cpu_fallback (acquire failed)");
                            }
                            return cpu().map(|v| (None, v));
                        }
                        return Err(err);
                    }
                };

                let ordinal = self.devices[idx];

                if self.debug {
                    let usable = self.usable_device_count();
                    sched_log(&format!(
                        "[sched] try device={} (usable={}/{})",
                        ordinal,
                        usable,
                        self.devices.len()
                    ));
                }

                match gpu(ordinal) {
                    Ok(v) => {
                        if self.debug {
                            sched_log(&format!("[sched] success device={}", ordinal));
                        }
                        return Ok((Some(ordinal), v));
                    }
                    Err(err) => {
                        if is_not_implemented(&err) {
                            if self.debug {
                                sched_log(&format!(
                                    "[sched] gpu_not_implemented on device={} -> propagate",
                                    ordinal
                                ));
                            }
                            return Err(err);
                        }

                        if self.debug {
                            sched_log(&format!(
                                "[sched] device_failed {} err={:#}",
                                ordinal, err
                            ));
                        }
                        self.mark_failed_ordinal(ordinal);
                        continue;
                    }
                }
            }
        }

        if allow_fallback {
            if self.debug {
                sched_log("[sched] cpu_fallback (all devices failed/unusable)");
            }
            cpu().map(|v| (None, v))
        } else {
            Err(anyhow::anyhow!("all GPU devices failed"))
        }
    }
}
