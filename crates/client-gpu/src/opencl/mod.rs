//! OpenCL backend, dynamically loaded.
//!
//! This module provides:
//! - device enumeration for user selection (`--list-gpus`)
//! - lightweight availability probe
//!
//! The full GPU compute path will be implemented later.

use libloading::Library;

pub mod compute;

use crate::{GpuConfig, GpuDeviceSelection, GpuProbe};

type ClInt = i32;
type ClUInt = u32;
type ClULong = u64;
type ClPlatformId = *mut core::ffi::c_void;
type ClDeviceId = *mut core::ffi::c_void;
type ClDeviceType = u64;

const CL_SUCCESS: ClInt = 0;

// clGetPlatformInfo params
const CL_PLATFORM_NAME: ClUInt = 0x0902;
const CL_PLATFORM_VENDOR: ClUInt = 0x0903;

// clGetDeviceInfo params
const CL_DEVICE_NAME: ClUInt = 0x102B;
const CL_DEVICE_VENDOR: ClUInt = 0x102C;
const CL_DEVICE_GLOBAL_MEM_SIZE: ClUInt = 0x101F;
const CL_DEVICE_MAX_COMPUTE_UNITS: ClUInt = 0x1002;

const CL_DEVICE_TYPE_GPU: ClDeviceType = 1 << 2;

type ClGetPlatformIDsFn =
    unsafe extern "C" fn(num_entries: ClUInt, platforms: *mut ClPlatformId, num_platforms: *mut ClUInt) -> ClInt;

type ClGetPlatformInfoFn = unsafe extern "C" fn(
    platform: ClPlatformId,
    param_name: ClUInt,
    param_value_size: usize,
    param_value: *mut core::ffi::c_void,
    param_value_size_ret: *mut usize,
) -> ClInt;

type ClGetDeviceIDsFn = unsafe extern "C" fn(
    platform: ClPlatformId,
    device_type: ClDeviceType,
    num_entries: ClUInt,
    devices: *mut ClDeviceId,
    num_devices: *mut ClUInt,
) -> ClInt;

type ClGetDeviceInfoFn = unsafe extern "C" fn(
    device: ClDeviceId,
    param_name: ClUInt,
    param_value_size: usize,
    param_value: *mut core::ffi::c_void,
    param_value_size_ret: *mut usize,
) -> ClInt;

#[derive(Debug)]
struct OpenClSymbols<'a> {
    cl_get_platform_ids: libloading::Symbol<'a, ClGetPlatformIDsFn>,
    cl_get_platform_info: libloading::Symbol<'a, ClGetPlatformInfoFn>,
    cl_get_device_ids: libloading::Symbol<'a, ClGetDeviceIDsFn>,
    cl_get_device_info: libloading::Symbol<'a, ClGetDeviceInfoFn>,
}

unsafe fn resolve_symbols(lib: &Library) -> anyhow::Result<OpenClSymbols<'_>> {
    let cl_get_platform_ids: libloading::Symbol<ClGetPlatformIDsFn> = unsafe { lib.get(b"clGetPlatformIDs\0") }
        .map_err(|e| anyhow::anyhow!("missing symbol clGetPlatformIDs: {e}"))?;

    let cl_get_platform_info: libloading::Symbol<ClGetPlatformInfoFn> = unsafe { lib.get(b"clGetPlatformInfo\0") }
        .map_err(|e| anyhow::anyhow!("missing symbol clGetPlatformInfo: {e}"))?;

    let cl_get_device_ids: libloading::Symbol<ClGetDeviceIDsFn> = unsafe { lib.get(b"clGetDeviceIDs\0") }
        .map_err(|e| anyhow::anyhow!("missing symbol clGetDeviceIDs: {e}"))?;

    let cl_get_device_info: libloading::Symbol<ClGetDeviceInfoFn> = unsafe { lib.get(b"clGetDeviceInfo\0") }
        .map_err(|e| anyhow::anyhow!("missing symbol clGetDeviceInfo: {e}"))?;

    Ok(OpenClSymbols {
        cl_get_platform_ids,
        cl_get_platform_info,
        cl_get_device_ids,
        cl_get_device_info,
    })
}

unsafe fn read_platform_string(
    f: &libloading::Symbol<ClGetPlatformInfoFn>,
    platform: ClPlatformId,
    param: ClUInt,
) -> Option<String> {
    let mut size: usize = 0;

    let rc = unsafe { f(platform, param, 0, core::ptr::null_mut(), &mut size as *mut usize) };
    if rc != CL_SUCCESS || size == 0 {
        return None;
    }

    let mut buf = vec![0u8; size];
    let rc = unsafe {
        f(
            platform,
            param,
            buf.len(),
            buf.as_mut_ptr() as *mut core::ffi::c_void,
            core::ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS {
        return None;
    }

    assumed_c_string_to_utf8(&mut buf)
}

unsafe fn read_device_string(
    f: &libloading::Symbol<ClGetDeviceInfoFn>,
    device: ClDeviceId,
    param: ClUInt,
) -> Option<String> {
    let mut size: usize = 0;

    let rc = unsafe { f(device, param, 0, core::ptr::null_mut(), &mut size as *mut usize) };
    if rc != CL_SUCCESS || size == 0 {
        return None;
    }

    let mut buf = vec![0u8; size];
    let rc = unsafe {
        f(
            device,
            param,
            buf.len(),
            buf.as_mut_ptr() as *mut core::ffi::c_void,
            core::ptr::null_mut(),
        )
    };
    if rc != CL_SUCCESS {
        return None;
    }

    assumed_c_string_to_utf8(&mut buf)
}

fn assumed_c_string_to_utf8(buf: &mut Vec<u8>) -> Option<String> {
    if let Some(pos) = buf.iter().position(|&c| c == 0) {
        buf.truncate(pos);
    }
    String::from_utf8(buf.clone()).ok()
}

unsafe fn read_device_u64(
    f: &libloading::Symbol<ClGetDeviceInfoFn>,
    device: ClDeviceId,
    param: ClUInt,
) -> Option<u64> {
    let mut value: ClULong = 0;
    let mut size: usize = 0;

    let rc = unsafe {
        f(
            device,
            param,
            core::mem::size_of::<ClULong>(),
            &mut value as *mut ClULong as *mut core::ffi::c_void,
            &mut size as *mut usize,
        )
    };
    if rc != CL_SUCCESS || size != core::mem::size_of::<ClULong>() {
        return None;
    }

    Some(value as u64)
}

unsafe fn read_device_u32(
    f: &libloading::Symbol<ClGetDeviceInfoFn>,
    device: ClDeviceId,
    param: ClUInt,
) -> Option<u32> {
    let mut value: ClUInt = 0;
    let mut size: usize = 0;

    let rc = unsafe {
        f(
            device,
            param,
            core::mem::size_of::<ClUInt>(),
            &mut value as *mut ClUInt as *mut core::ffi::c_void,
            &mut size as *mut usize,
        )
    };
    if rc != CL_SUCCESS || size != core::mem::size_of::<ClUInt>() {
        return None;
    }

    Some(value as u32)
}

fn match_user_selection(selection: &GpuDeviceSelection, index: usize) -> bool {
    match selection {
        GpuDeviceSelection::All => true,
        GpuDeviceSelection::None => false,
        GpuDeviceSelection::Indices(list) => list.contains(&(index as u32)),
    }
}

pub fn probe(cfg: &GpuConfig) -> GpuProbe {
    let selection = cfg.selection.clone();

    let lib = match compute::try_load_opencl() {
        Ok(v) => v,
        Err(err) => return GpuProbe::unavailable(format!("OpenCL load failed: {err}")),
    };

    let symbols = unsafe {
        match resolve_symbols(&lib) {
            Ok(s) => s,
            Err(err) => return GpuProbe::unavailable(format!("OpenCL symbols missing: {err}")),
        }
    };

    // Enumerate platforms
    let mut num_platforms: ClUInt = 0;
    let rc = unsafe { (symbols.cl_get_platform_ids)(0, core::ptr::null_mut(), &mut num_platforms as *mut ClUInt) };
    if rc != CL_SUCCESS || num_platforms == 0 {
        return GpuProbe::unavailable("No OpenCL platform found".to_string());
    }

    let mut platforms = vec![core::ptr::null_mut(); num_platforms as usize];
    let rc = unsafe {
        (symbols.cl_get_platform_ids)(
            num_platforms,
            platforms.as_mut_ptr(),
            &mut num_platforms as *mut ClUInt,
        )
    };
    if rc != CL_SUCCESS {
        return GpuProbe::unavailable(format!("clGetPlatformIDs failed: rc={rc}"));
    }

    // Enumerate devices (GPU only) and apply user selection by enumeration index
    let mut any_selected = false;
    let mut idx: usize = 0;

    for platform in platforms {
        let _platform_name =
            unsafe { read_platform_string(&symbols.cl_get_platform_info, platform, CL_PLATFORM_NAME) }
                .unwrap_or_else(|| "<unknown>".to_string());
        let _platform_vendor =
            unsafe { read_platform_string(&symbols.cl_get_platform_info, platform, CL_PLATFORM_VENDOR) }
                .unwrap_or_else(|| "<unknown>".to_string());

        let mut num_devices: ClUInt = 0;
        let rc = unsafe {
            (symbols.cl_get_device_ids)(
                platform,
                CL_DEVICE_TYPE_GPU,
                0,
                core::ptr::null_mut(),
                &mut num_devices as *mut ClUInt,
            )
        };
        if rc != CL_SUCCESS || num_devices == 0 {
            continue;
        }

        let mut devices = vec![core::ptr::null_mut(); num_devices as usize];
        let rc = unsafe {
            (symbols.cl_get_device_ids)(
                platform,
                CL_DEVICE_TYPE_GPU,
                num_devices,
                devices.as_mut_ptr(),
                &mut num_devices as *mut ClUInt,
            )
        };
        if rc != CL_SUCCESS {
            continue;
        }

        for device in devices {
            // Keep reads to validate the device is queryable (also helps later for logs).
            let _name = unsafe { read_device_string(&symbols.cl_get_device_info, device, CL_DEVICE_NAME) }
                .unwrap_or_else(|| "<unknown>".to_string());
            let _vendor = unsafe { read_device_string(&symbols.cl_get_device_info, device, CL_DEVICE_VENDOR) }
                .unwrap_or_else(|| "<unknown>".to_string());
            let _mem =
                unsafe { read_device_u64(&symbols.cl_get_device_info, device, CL_DEVICE_GLOBAL_MEM_SIZE) }.unwrap_or(0);
            let _cu =
                unsafe { read_device_u32(&symbols.cl_get_device_info, device, CL_DEVICE_MAX_COMPUTE_UNITS) }.unwrap_or(0);

            if match_user_selection(&selection, idx) {
                any_selected = true;
            }
            idx += 1;
        }
    }

    if idx == 0 {
        return GpuProbe::unavailable("No OpenCL GPU device found".to_string());
    }

    if !any_selected {
        return GpuProbe::unavailable("No OpenCL GPU matches the user selection".to_string());
    }

    GpuProbe::available("OpenCL".to_string())
}
