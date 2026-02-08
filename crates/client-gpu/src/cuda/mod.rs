//! NVIDIA CUDA backend (driver API), dynamically loaded.
//!
//! This module provides:
//! - device enumeration for user selection (`--list-gpus`)
//! - lightweight availability probe
//!
//! The full GPU compute path will be implemented later.

pub mod compute;

use libloading::Library;

use crate::{GpuConfig, GpuDeviceSelection, GpuProbe};

/// CUDA device info (best-effort).
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub index: u32,
    pub name: String,
    pub total_mem_bytes: Option<u64>,
    pub sm_count: Option<i32>,
    pub compute_capability: Option<(i32, i32)>,
}

type CUdevice = i32;
type CUresult = i32;
const CUDA_SUCCESS: CUresult = 0;

type CuInitFn = unsafe extern "C" fn(flags: u32) -> CUresult;
type CuDeviceGetCountFn = unsafe extern "C" fn(count: *mut i32) -> CUresult;
type CuDeviceGetFn = unsafe extern "C" fn(device: *mut CUdevice, ordinal: i32) -> CUresult;
type CuDeviceGetNameFn = unsafe extern "C" fn(name: *mut i8, len: i32, dev: CUdevice) -> CUresult;
type CuDeviceTotalMemV2Fn = unsafe extern "C" fn(bytes: *mut usize, dev: CUdevice) -> CUresult;
type CuDeviceGetAttributeFn =
    unsafe extern "C" fn(pi: *mut i32, attrib: i32, dev: CUdevice) -> CUresult;

// CUDA device attributes (subset).
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: i32 = 16;
// These are present on modern drivers; keep best-effort if missing.
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: i32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: i32 = 76;

fn candidate_libraries() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        return &["nvcuda.dll"];
    }
    #[cfg(target_os = "linux")]
    {
        return &["libcuda.so.1", "libcuda.so"];
    }
    #[cfg(target_os = "macos")]
    {
        return &["libcuda.dylib"];
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        return &[];
    }
}

fn load_cuda_library() -> anyhow::Result<Library> {
    let mut last_err: Option<anyhow::Error> = None;
    for name in candidate_libraries() {
        match unsafe { Library::new(name) } {
            Ok(lib) => return Ok(lib),
            Err(err) => {
                last_err = Some(anyhow::anyhow!(err).context(format!("failed to load {name}")));
            }
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no CUDA library candidates for this OS")))
}

struct CudaSymbols<'a> {
    cu_init: libloading::Symbol<'a, CuInitFn>,
    cu_device_get_count: libloading::Symbol<'a, CuDeviceGetCountFn>,
    cu_device_get: libloading::Symbol<'a, CuDeviceGetFn>,
    cu_device_get_name: Option<libloading::Symbol<'a, CuDeviceGetNameFn>>,
    cu_device_total_mem_v2: Option<libloading::Symbol<'a, CuDeviceTotalMemV2Fn>>,
    cu_device_get_attribute: Option<libloading::Symbol<'a, CuDeviceGetAttributeFn>>,
}

unsafe fn resolve_symbols(lib: &Library) -> anyhow::Result<CudaSymbols<'_>> {
    let cu_init: libloading::Symbol<CuInitFn> = unsafe { lib.get(b"cuInit\0")}
        .map_err(|e| anyhow::anyhow!("missing symbol cuInit: {e}"))?;
    let cu_device_get_count: libloading::Symbol<CuDeviceGetCountFn> = unsafe { lib.get(b"cuDeviceGetCount\0")}
        .map_err(|e| anyhow::anyhow!("missing symbol cuDeviceGetCount: {e}"))?;
    let cu_device_get: libloading::Symbol<CuDeviceGetFn> = unsafe { lib.get(b"cuDeviceGet\0")}
        .map_err(|e| anyhow::anyhow!("missing symbol cuDeviceGet: {e}"))?;

    let cu_device_get_name = unsafe { lib.get(b"cuDeviceGetName\0") }.ok();
    let cu_device_total_mem_v2 = unsafe { lib.get(b"cuDeviceTotalMem_v2\0") }.ok();
    let cu_device_get_attribute = unsafe { lib.get(b"cuDeviceGetAttribute\0") }.ok();

    Ok(CudaSymbols {
        cu_init,
        cu_device_get_count,
        cu_device_get,
        cu_device_get_name,
        cu_device_total_mem_v2,
        cu_device_get_attribute,
    })
}

fn usable_count_from_selection(total: i32, sel: &GpuDeviceSelection) -> i32 {
    match sel {
        GpuDeviceSelection::All => total,
        GpuDeviceSelection::None => 0,
        GpuDeviceSelection::List(list) => list.iter().filter(|&&i| (i as i32) < total).count() as i32,
    }
}

/// Enumerate CUDA devices (best-effort).
pub fn enumerate_devices() -> anyhow::Result<Vec<CudaDeviceInfo>> {
    let lib = load_cuda_library()?;
    unsafe {
        let sym = resolve_symbols(&lib)?;
        let rc = (sym.cu_init)(0);
        if rc != CUDA_SUCCESS {
            anyhow::bail!("cuInit failed with code {rc}");
        }

        let mut count: i32 = 0;
        let rc = (sym.cu_device_get_count)(&mut count as *mut i32);
        if rc != CUDA_SUCCESS {
            anyhow::bail!("cuDeviceGetCount failed with code {rc}");
        }
        if count <= 0 {
            return Ok(Vec::new());
        }

        let mut out = Vec::with_capacity(count as usize);
        for ordinal in 0..count {
            let mut dev: CUdevice = 0;
            let rc = (sym.cu_device_get)(&mut dev as *mut CUdevice, ordinal);
            if rc != CUDA_SUCCESS {
                continue;
            }

            let name = if let Some(ref f) = sym.cu_device_get_name {
                let mut buf = vec![0i8; 256];
                let rc = f(buf.as_mut_ptr(), buf.len() as i32, dev);
                if rc == CUDA_SUCCESS {
                    let bytes: Vec<u8> = buf
                        .iter()
                        .take_while(|&&c| c != 0)
                        .map(|&c| c as u8)
                        .collect();
                    String::from_utf8_lossy(&bytes).to_string()
                } else {
                    format!("CUDA device {ordinal}")
                }
            } else {
                format!("CUDA device {ordinal}")
            };

            let total_mem_bytes = if let Some(ref f) = sym.cu_device_total_mem_v2 {
                let mut mem: usize = 0;
                let rc = f(&mut mem as *mut usize, dev);
                if rc == CUDA_SUCCESS {
                    Some(mem as u64)
                } else {
                    None
                }
            } else {
                None
            };

            let (sm_count, compute_capability) = if let Some(ref f) = sym.cu_device_get_attribute {
                let mut sm: i32 = 0;
                let rc_sm = f(&mut sm as *mut i32, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev);
                let sm_val = if rc_sm == CUDA_SUCCESS { Some(sm) } else { None };

                let mut major: i32 = 0;
                let mut minor: i32 = 0;
                let rc_ma = f(
                    &mut major as *mut i32,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    dev,
                );
                let rc_mi = f(
                    &mut minor as *mut i32,
                    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    dev,
                );
                let cc = if rc_ma == CUDA_SUCCESS && rc_mi == CUDA_SUCCESS {
                    Some((major, minor))
                } else {
                    None
                };

                (sm_val, cc)
            } else {
                (None, None)
            };

            out.push(CudaDeviceInfo {
                index: ordinal as u32,
                name,
                total_mem_bytes,
                sm_count,
                compute_capability,
            });
        }

        Ok(out)
    }
}

/// List CUDA device ordinals (best-effort).
pub fn list_device_ordinals() -> anyhow::Result<Vec<u32>> {
    Ok(enumerate_devices()?.into_iter().map(|d| d.index).collect())
}

/// Probe whether CUDA appears usable using environment-derived config.
pub fn probe() -> GpuProbe {
    probe_with_config(&GpuConfig::from_env())
}

/// Probe whether CUDA appears usable using an explicit config.
pub fn probe_with_config(cfg: &GpuConfig) -> GpuProbe {
    if cfg.is_disabled() {
        return GpuProbe::unavailable("GPU disabled by user (BBR_GPU_DEVICES=none)");
    }

    let list = match enumerate_devices() {
        Ok(v) => v,
        Err(err) => return GpuProbe::unavailable(err.to_string()),
    };

    let total = list.len() as i32;
    if total <= 0 {
        return GpuProbe::unavailable("no CUDA devices detected");
    }

    let usable = usable_count_from_selection(total, &cfg.devices);
    if usable <= 0 {
        return GpuProbe::unavailable("no CUDA devices match user selection");
    }

    GpuProbe::available(Some(format!("CUDA devices: {usable}/{total} usable")))
}
