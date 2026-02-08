//! CUDA compute backend (v0).
//!
//! This file implements a first building block for a real VDF GPU backend:
//! big-integer squaring using 32-bit limbs (u32).
//!
//! IMPORTANT:
//! - Chia's VDF (Wesolowski / class group) is NOT a simple modular squaring.
//! - This kernel only computes a full square: (1024-bit)² -> (2048-bit), no reduction.
//! - The class group reduction / composition will be implemented on top of this later.
//!
//! Design goals:
//! - Keep integration external to the CPU engine.
//! - Avoid linking to CUDA/NVRTC at build time (dynamic loading via libloading).
//! - Provide deterministic self-test to validate the GPU path.

use anyhow::{anyhow, bail, Context};
use libloading::{Library, Symbol};
use std::path::{Path, PathBuf};

const LIMBS_1024: usize = 32; // 32 * 32-bit = 1024-bit
const OUT_LIMBS_2048: usize = LIMBS_1024 * 2;

#[allow(non_camel_case_types)]
type CUresult = i32;
#[allow(non_camel_case_types)]
type CUdevice = i32;
#[allow(non_camel_case_types)]
type CUcontext = *mut core::ffi::c_void;
#[allow(non_camel_case_types)]
type CUmodule = *mut core::ffi::c_void;
#[allow(non_camel_case_types)]
type CUfunction = *mut core::ffi::c_void;
#[allow(non_camel_case_types)]
type CUstream = *mut core::ffi::c_void;
#[allow(non_camel_case_types)]
type CUdeviceptr = u64;

const CUDA_SUCCESS: CUresult = 0;

#[allow(non_camel_case_types)]
type nvrtcResult = i32;
const NVRTC_SUCCESS: nvrtcResult = 0;

#[repr(C)]
#[derive(Copy, Clone)]
struct NvrtcProgram(*mut core::ffi::c_void);

struct CudaSymbols {
    _lib: Library,
    cu_init: Symbol<'static, unsafe extern "C" fn(u32) -> CUresult>,
    cu_device_get: Symbol<'static, unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult>,
    cu_ctx_create_v2: Symbol<'static, unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult>,
    cu_ctx_destroy_v2: Symbol<'static, unsafe extern "C" fn(CUcontext) -> CUresult>,
    cu_mem_alloc_v2: Symbol<'static, unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult>,
    cu_mem_free_v2: Symbol<'static, unsafe extern "C" fn(CUdeviceptr) -> CUresult>,
    cu_memcpy_hto_d_v2: Symbol<'static, unsafe extern "C" fn(CUdeviceptr, *const core::ffi::c_void, usize) -> CUresult>,
    cu_memcpy_dto_h_v2: Symbol<'static, unsafe extern "C" fn(*mut core::ffi::c_void, CUdeviceptr, usize) -> CUresult>,
    cu_module_load_data: Symbol<'static, unsafe extern "C" fn(*mut CUmodule, *const core::ffi::c_void) -> CUresult>,
    cu_module_unload: Symbol<'static, unsafe extern "C" fn(CUmodule) -> CUresult>,
    cu_module_get_function: Symbol<'static, unsafe extern "C" fn(*mut CUfunction, CUmodule, *const i8) -> CUresult>,
    cu_launch_kernel: Symbol<
        'static,
        unsafe extern "C" fn(
            CUfunction,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            CUstream,
            *mut *mut core::ffi::c_void,
            *mut *mut core::ffi::c_void,
        ) -> CUresult,
    >,
    cu_ctx_synchronize: Symbol<'static, unsafe extern "C" fn() -> CUresult>,
}

struct NvrtcSymbols {
    _lib: Library,
    nvrtc_create_program: Symbol<
        'static,
        unsafe extern "C" fn(
            *mut NvrtcProgram,
            *const i8,
            *const i8,
            i32,
            *const *const i8,
            *const *const i8,
        ) -> nvrtcResult,
    >,
    nvrtc_compile_program: Symbol<'static, unsafe extern "C" fn(NvrtcProgram, i32, *const *const i8) -> nvrtcResult>,
    nvrtc_get_ptx_size: Symbol<'static, unsafe extern "C" fn(NvrtcProgram, *mut usize) -> nvrtcResult>,
    nvrtc_get_ptx: Symbol<'static, unsafe extern "C" fn(NvrtcProgram, *mut i8) -> nvrtcResult>,
    nvrtc_get_program_log_size: Symbol<'static, unsafe extern "C" fn(NvrtcProgram, *mut usize) -> nvrtcResult>,
    nvrtc_get_program_log: Symbol<'static, unsafe extern "C" fn(NvrtcProgram, *mut i8) -> nvrtcResult>,
    nvrtc_destroy_program: Symbol<'static, unsafe extern "C" fn(*mut NvrtcProgram) -> nvrtcResult>,
}

unsafe fn get_sym<'a, T>(lib: &'a Library, name: &[u8]) -> anyhow::Result<Symbol<'a, T>> {
    unsafe { lib.get::<T>(name) }
        .map_err(|e| anyhow!("missing symbol {}: {}", String::from_utf8_lossy(name), e))
}

fn load_cuda() -> anyhow::Result<CudaSymbols> {
    #[cfg(windows)]
    const CANDIDATES: [&str; 3] = ["nvcuda.dll", "cuda.dll", "nvcuda64.dll"];
    #[cfg(not(windows))]
    const CANDIDATES: [&str; 3] = ["libcuda.so.1", "libcuda.so", "libcuda.dylib"];

    let mut last_err: Option<anyhow::Error> = None;

    for name in CANDIDATES {
        match unsafe { Library::new(name) } {
            Ok(lib) => unsafe {
                let cu_init = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(u32) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(u32) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(u32) -> CUresult>(&lib, b"cuInit\0")?);

                let cu_device_get = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(*mut CUdevice, i32) -> CUresult>(
                    &lib,
                    b"cuDeviceGet\0",
                )?);

                let cu_ctx_create_v2 = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(*mut CUcontext, u32, CUdevice) -> CUresult>(
                    &lib,
                    b"cuCtxCreate_v2\0",
                )?);

                let cu_ctx_destroy_v2 = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(CUcontext) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(CUcontext) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(CUcontext) -> CUresult>(&lib, b"cuCtxDestroy_v2\0")?);

                let cu_mem_alloc_v2 = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult>(
                    &lib,
                    b"cuMemAlloc_v2\0",
                )?);

                let cu_mem_free_v2 = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(CUdeviceptr) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(CUdeviceptr) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(CUdeviceptr) -> CUresult>(&lib, b"cuMemFree_v2\0")?);

                let cu_memcpy_hto_d_v2 = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(CUdeviceptr, *const core::ffi::c_void, usize) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(CUdeviceptr, *const core::ffi::c_void, usize) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(CUdeviceptr, *const core::ffi::c_void, usize) -> CUresult>(
                    &lib,
                    b"cuMemcpyHtoD_v2\0",
                )?);

                let cu_memcpy_dto_h_v2 = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(*mut core::ffi::c_void, CUdeviceptr, usize) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(*mut core::ffi::c_void, CUdeviceptr, usize) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(*mut core::ffi::c_void, CUdeviceptr, usize) -> CUresult>(
                    &lib,
                    b"cuMemcpyDtoH_v2\0",
                )?);

                let cu_module_load_data = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(*mut CUmodule, *const core::ffi::c_void) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(*mut CUmodule, *const core::ffi::c_void) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(*mut CUmodule, *const core::ffi::c_void) -> CUresult>(
                    &lib,
                    b"cuModuleLoadData\0",
                )?);

                let cu_module_unload = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(CUmodule) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(CUmodule) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(CUmodule) -> CUresult>(&lib, b"cuModuleUnload\0")?);

                let cu_module_get_function = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(*mut CUfunction, CUmodule, *const i8) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(*mut CUfunction, CUmodule, *const i8) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(*mut CUfunction, CUmodule, *const i8) -> CUresult>(
                    &lib,
                    b"cuModuleGetFunction\0",
                )?);

                let cu_launch_kernel = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn(
                        CUfunction,
                        u32, u32, u32,
                        u32, u32, u32,
                        u32,
                        CUstream,
                        *mut *mut core::ffi::c_void,
                        *mut *mut core::ffi::c_void
                    ) -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn(
                        CUfunction,
                        u32, u32, u32,
                        u32, u32, u32,
                        u32,
                        CUstream,
                        *mut *mut core::ffi::c_void,
                        *mut *mut core::ffi::c_void
                    ) -> CUresult>,
                >(get_sym::<unsafe extern "C" fn(
                    CUfunction,
                    u32, u32, u32,
                    u32, u32, u32,
                    u32,
                    CUstream,
                    *mut *mut core::ffi::c_void,
                    *mut *mut core::ffi::c_void
                ) -> CUresult>(&lib, b"cuLaunchKernel\0")?);

                let cu_ctx_synchronize = std::mem::transmute::<
                    Symbol<'_, unsafe extern "C" fn() -> CUresult>,
                    Symbol<'static, unsafe extern "C" fn() -> CUresult>,
                >(get_sym::<unsafe extern "C" fn() -> CUresult>(&lib, b"cuCtxSynchronize\0")?);

                return Ok(CudaSymbols {
                    _lib: lib,
                    cu_init,
                    cu_device_get,
                    cu_ctx_create_v2,
                    cu_ctx_destroy_v2,
                    cu_mem_alloc_v2,
                    cu_mem_free_v2,
                    cu_memcpy_hto_d_v2,
                    cu_memcpy_dto_h_v2,
                    cu_module_load_data,
                    cu_module_unload,
                    cu_module_get_function,
                    cu_launch_kernel,
                    cu_ctx_synchronize,
                });
            },
            Err(e) => last_err = Some(anyhow!(e)),
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("CUDA driver library not found")))
}

fn try_load_library(path: &Path) -> anyhow::Result<Library> {
    unsafe { Library::new(path) }.map_err(|e| anyhow!("LoadLibraryExW failed for {}: {}", path.display(), e))
}

fn load_nvrtc() -> anyhow::Result<NvrtcSymbols> {
    // User overrides (recommended on Windows)
    // - BBR_NVRTC_DLL: full path to nvrtc64_*.dll
    // - BBR_NVRTC_DIR: directory containing nvrtc64_*.dll
    if let Ok(dll) = std::env::var("BBR_NVRTC_DLL") {
        let lib = try_load_library(Path::new(&dll))?;
        return unsafe { resolve_nvrtc(lib) }.context("resolve NVRTC symbols (BBR_NVRTC_DLL)");
    }

    #[cfg(windows)]
    {
        if let Ok(dir) = std::env::var("BBR_NVRTC_DIR") {
            let base = PathBuf::from(dir);
            let names = [
                // CUDA 13.x
                "nvrtc64_131_0.dll",
                "nvrtc64_130_0.dll",
                // CUDA 12.x
                "nvrtc64_122_0.dll",
                "nvrtc64_121_0.dll",
                "nvrtc64_120_0.dll",
                // older
                "nvrtc64_102_0.dll",
            ];

            for n in names {
                let p = base.join(n);
                if p.exists() {
                    let lib = try_load_library(&p)?;
                    return unsafe { resolve_nvrtc(lib) }.context("resolve NVRTC symbols (BBR_NVRTC_DIR)");
                }
            }

            bail!(
                "NVRTC not found in BBR_NVRTC_DIR={}, expected one of: {:?}",
                base.display(),
                names
            );
        }

        // Auto-detect common CUDA install directories on Windows.
        if let Ok(pf) = std::env::var("ProgramFiles") {
            let cuda_root = PathBuf::from(pf).join("NVIDIA GPU Computing Toolkit").join("CUDA");
            if cuda_root.exists() {
                if let Ok(entries) = std::fs::read_dir(&cuda_root) {
                    // Iterate v* directories (best-effort).
                    let mut dirs: Vec<PathBuf> = entries
                        .filter_map(|e| e.ok().map(|e| e.path()))
                        .filter(|p| p.is_dir())
                        .collect();
                    dirs.sort(); // stable order

                    let names = ["nvrtc64_131_0.dll", "nvrtc64_130_0.dll", "nvrtc64_122_0.dll", "nvrtc64_121_0.dll", "nvrtc64_120_0.dll", "nvrtc64_102_0.dll"];

                    for d in dirs.iter().rev() {
                        let bin = d.join("bin").join("x64");
                        if !bin.exists() {
                            continue;
                        }
                        for n in names {
                            let cand = bin.join(n);
                            if cand.exists() {
                                let lib = try_load_library(&cand)?;
                                return unsafe { resolve_nvrtc(lib) }.context("resolve NVRTC symbols (auto-detect)");
                            }
                        }
                    }
                }
            }
        }
    }

    // Default search by filename (relies on PATH / current dir)
    #[cfg(windows)]
    const CANDIDATES: [&str; 7] = [
        "nvrtc64_131_0.dll",
        "nvrtc64_130_0.dll",
        "nvrtc64_122_0.dll",
        "nvrtc64_121_0.dll",
        "nvrtc64_120_0.dll",
        "nvrtc64_102_0.dll",
        "nvrtc64.dll",
    ];
    #[cfg(not(windows))]
    const CANDIDATES: [&str; 3] = ["libnvrtc.so.12", "libnvrtc.so", "libnvrtc.dylib"];

    let mut last_err: Option<anyhow::Error> = None;

    for name in CANDIDATES {
        match unsafe { Library::new(name) } {
            Ok(lib) => return unsafe { resolve_nvrtc(lib) }.context("resolve NVRTC symbols"),
            Err(e) => last_err = Some(anyhow!(e)),
        }
    }

    Err(last_err.unwrap_or_else(|| anyhow!("NVRTC library not found (tried candidates)")))
}

unsafe fn resolve_nvrtc(lib: Library) -> anyhow::Result<NvrtcSymbols> {
    unsafe {
        let nvrtc_create_program = std::mem::transmute::<
            Symbol<'_, unsafe extern "C" fn(
                *mut NvrtcProgram, *const i8, *const i8, i32, *const *const i8, *const *const i8
            ) -> nvrtcResult>,
            Symbol<'static, unsafe extern "C" fn(
                *mut NvrtcProgram, *const i8, *const i8, i32, *const *const i8, *const *const i8
            ) -> nvrtcResult>,
        >(get_sym::<unsafe extern "C" fn(
            *mut NvrtcProgram, *const i8, *const i8, i32, *const *const i8, *const *const i8
        ) -> nvrtcResult>(&lib, b"nvrtcCreateProgram\0")?);

        let nvrtc_compile_program = std::mem::transmute::<
            Symbol<'_, unsafe extern "C" fn(NvrtcProgram, i32, *const *const i8) -> nvrtcResult>,
            Symbol<'static, unsafe extern "C" fn(NvrtcProgram, i32, *const *const i8) -> nvrtcResult>,
        >(get_sym::<unsafe extern "C" fn(NvrtcProgram, i32, *const *const i8) -> nvrtcResult>(
            &lib,
            b"nvrtcCompileProgram\0",
        )?);

        let nvrtc_get_ptx_size = std::mem::transmute::<
            Symbol<'_, unsafe extern "C" fn(NvrtcProgram, *mut usize) -> nvrtcResult>,
            Symbol<'static, unsafe extern "C" fn(NvrtcProgram, *mut usize) -> nvrtcResult>,
        >(get_sym::<unsafe extern "C" fn(NvrtcProgram, *mut usize) -> nvrtcResult>(
            &lib,
            b"nvrtcGetPTXSize\0",
        )?);

        let nvrtc_get_ptx = std::mem::transmute::<
            Symbol<'_, unsafe extern "C" fn(NvrtcProgram, *mut i8) -> nvrtcResult>,
            Symbol<'static, unsafe extern "C" fn(NvrtcProgram, *mut i8) -> nvrtcResult>,
        >(get_sym::<unsafe extern "C" fn(NvrtcProgram, *mut i8) -> nvrtcResult>(&lib, b"nvrtcGetPTX\0")?);

        let nvrtc_get_program_log_size = std::mem::transmute::<
            Symbol<'_, unsafe extern "C" fn(NvrtcProgram, *mut usize) -> nvrtcResult>,
            Symbol<'static, unsafe extern "C" fn(NvrtcProgram, *mut usize) -> nvrtcResult>,
        >(get_sym::<unsafe extern "C" fn(NvrtcProgram, *mut usize) -> nvrtcResult>(
            &lib,
            b"nvrtcGetProgramLogSize\0",
        )?);

        let nvrtc_get_program_log = std::mem::transmute::<
            Symbol<'_, unsafe extern "C" fn(NvrtcProgram, *mut i8) -> nvrtcResult>,
            Symbol<'static, unsafe extern "C" fn(NvrtcProgram, *mut i8) -> nvrtcResult>,
        >(get_sym::<unsafe extern "C" fn(NvrtcProgram, *mut i8) -> nvrtcResult>(
            &lib,
            b"nvrtcGetProgramLog\0",
        )?);

        let nvrtc_destroy_program = std::mem::transmute::<
            Symbol<'_, unsafe extern "C" fn(*mut NvrtcProgram) -> nvrtcResult>,
            Symbol<'static, unsafe extern "C" fn(*mut NvrtcProgram) -> nvrtcResult>,
        >(get_sym::<unsafe extern "C" fn(*mut NvrtcProgram) -> nvrtcResult>(
            &lib,
            b"nvrtcDestroyProgram\0",
        )?);

        Ok(NvrtcSymbols {
            _lib: lib,
            nvrtc_create_program,
            nvrtc_compile_program,
            nvrtc_get_ptx_size,
            nvrtc_get_ptx,
            nvrtc_get_program_log_size,
            nvrtc_get_program_log,
            nvrtc_destroy_program,
        })
    }
}

fn cuda_check(rc: CUresult, what: &str) -> anyhow::Result<()> {
    if rc == CUDA_SUCCESS { Ok(()) } else { Err(anyhow!("CUDA error {} while {}", rc, what)) }
}

fn nvrtc_check(rc: nvrtcResult, what: &str) -> anyhow::Result<()> {
    if rc == NVRTC_SUCCESS { Ok(()) } else { Err(anyhow!("NVRTC error {} while {}", rc, what)) }
}

fn nvrtc_get_log(nvrtc: &NvrtcSymbols, prog: NvrtcProgram) -> String {
    unsafe {
        let mut size: usize = 0;
        if (nvrtc.nvrtc_get_program_log_size)(prog, &mut size as *mut usize) != NVRTC_SUCCESS || size == 0 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        let _ = (nvrtc.nvrtc_get_program_log)(prog, buf.as_mut_ptr() as *mut i8);
        String::from_utf8_lossy(&buf).to_string()
    }
}

fn compile_ptx(nvrtc: &NvrtcSymbols, source: &str, name: &str) -> anyhow::Result<Vec<u8>> {
    unsafe {
        let src_c = std::ffi::CString::new(source).context("CString(source)")?;
        let name_c = std::ffi::CString::new(name).context("CString(name)")?;

        let mut prog = NvrtcProgram(core::ptr::null_mut());
        nvrtc_check(
            (nvrtc.nvrtc_create_program)(
                &mut prog as *mut NvrtcProgram,
                src_c.as_ptr(),
                name_c.as_ptr(),
                0,
                core::ptr::null(),
                core::ptr::null(),
            ),
            "nvrtcCreateProgram",
        )?;

        let opts: Vec<std::ffi::CString> = vec![
            std::ffi::CString::new("--std=c++11").unwrap(),
            std::ffi::CString::new("--device-as-default-execution-space").unwrap(),
        ];
        let opt_ptrs: Vec<*const i8> = opts.iter().map(|s| s.as_ptr()).collect();

        let compile_rc = (nvrtc.nvrtc_compile_program)(prog, opt_ptrs.len() as i32, opt_ptrs.as_ptr());
        if compile_rc != NVRTC_SUCCESS {
            let log = nvrtc_get_log(nvrtc, prog);
            let _ = (nvrtc.nvrtc_destroy_program)(&mut prog as *mut NvrtcProgram);
            bail!("NVRTC compile failed: rc={} log=\n{}", compile_rc, log);
        }

        let mut ptx_size: usize = 0;
        nvrtc_check((nvrtc.nvrtc_get_ptx_size)(prog, &mut ptx_size as *mut usize), "nvrtcGetPTXSize")?;
        let mut ptx = vec![0u8; ptx_size];
        nvrtc_check((nvrtc.nvrtc_get_ptx)(prog, ptx.as_mut_ptr() as *mut i8), "nvrtcGetPTX")?;

        nvrtc_check((nvrtc.nvrtc_destroy_program)(&mut prog as *mut NvrtcProgram), "nvrtcDestroyProgram")?;
        Ok(ptx)
    }
}

fn kernel_source_u32_square() -> String {
    r#"
extern "C" __global__
void bigint_square_u32_32(const unsigned int* __restrict__ a,
                          unsigned int* __restrict__ out)
{
    __shared__ unsigned long long sh[256];

    int k = (int)blockIdx.x; // 0..63
    int tid = (int)threadIdx.x;

    unsigned long long acc = 0ULL;

    for (int i = tid; i < 32; i += blockDim.x) {
        int j = k - i;
        if (j < 0 || j >= 32) continue;

        unsigned long long ai = (unsigned long long)a[i];
        unsigned long long aj = (unsigned long long)a[j];

        if (i < j) acc += 2ULL * ai * aj;
        else if (i == j) acc += ai * aj;
    }

    sh[tid] = acc;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sh[tid] += sh[tid + stride];
        __syncthreads();
    }

    if (tid == 0) out[k] = (unsigned int)(sh[0] & 0xFFFFFFFFULL);
}
"#.to_string()
}

fn square_ref_u32_32(a: &[u32; LIMBS_1024]) -> [u32; OUT_LIMBS_2048] {
    let mut acc = vec![0u128; OUT_LIMBS_2048];
    for i in 0..LIMBS_1024 {
        for j in 0..LIMBS_1024 {
            acc[i + j] += (a[i] as u128) * (a[j] as u128);
        }
    }

    let mut out = [0u32; OUT_LIMBS_2048];
    let mut carry: u128 = 0;
    for k in 0..OUT_LIMBS_2048 {
        let v = acc[k] + carry;
        out[k] = (v & 0xFFFF_FFFF) as u32;
        carry = v >> 32;
    }
    out
}

fn bytes_to_limbs_1024_le(input: &[u8]) -> anyhow::Result<[u32; LIMBS_1024]> {
    if input.len() != LIMBS_1024 * 4 {
        bail!("Expected {} bytes for 1024-bit input, got {}", LIMBS_1024 * 4, input.len());
    }
    let mut limbs = [0u32; LIMBS_1024];
    for i in 0..LIMBS_1024 {
        let off = i * 4;
        limbs[i] = u32::from_le_bytes([input[off], input[off + 1], input[off + 2], input[off + 3]]);
    }
    Ok(limbs)
}

fn limbs_to_bytes_2048_le(limbs: &[u32; OUT_LIMBS_2048]) -> Vec<u8> {
    let mut out = vec![0u8; OUT_LIMBS_2048 * 4];
    for i in 0..OUT_LIMBS_2048 {
        out[i * 4..i * 4 + 4].copy_from_slice(&limbs[i].to_le_bytes());
    }
    out
}

pub fn square_1024_to_2048_u32_limbs_cuda(device_ordinal: u32, a_1024_le: &[u8]) -> anyhow::Result<Vec<u8>> {
    let cuda = load_cuda().context("load CUDA symbols")?;
    let nvrtc = load_nvrtc().context("load NVRTC symbols")?;
    let a = bytes_to_limbs_1024_le(a_1024_le).context("bytes_to_limbs_1024_le")?;

    unsafe {
        cuda_check((cuda.cu_init)(0), "cuInit")?;

        let mut dev: CUdevice = 0;
        cuda_check((cuda.cu_device_get)(&mut dev as *mut CUdevice, device_ordinal as i32), "cuDeviceGet")?;

        let mut ctx: CUcontext = core::ptr::null_mut();
        cuda_check((cuda.cu_ctx_create_v2)(&mut ctx as *mut CUcontext, 0, dev), "cuCtxCreate_v2")?;

        let ptx = compile_ptx(&nvrtc, &kernel_source_u32_square(), "bigint_square_u32_32.cu").context("compile PTX")?;

        let mut module: CUmodule = core::ptr::null_mut();
        cuda_check((cuda.cu_module_load_data)(&mut module as *mut CUmodule, ptx.as_ptr() as *const core::ffi::c_void), "cuModuleLoadData")?;

        let mut func: CUfunction = core::ptr::null_mut();
        let kname = std::ffi::CString::new("bigint_square_u32_32").unwrap();
        cuda_check((cuda.cu_module_get_function)(&mut func as *mut CUfunction, module, kname.as_ptr()), "cuModuleGetFunction")?;

        let in_bytes = LIMBS_1024 * 4;
        let out_bytes = OUT_LIMBS_2048 * 4;

        let mut d_in: CUdeviceptr = 0;
        let mut d_out: CUdeviceptr = 0;
        cuda_check((cuda.cu_mem_alloc_v2)(&mut d_in as *mut CUdeviceptr, in_bytes), "cuMemAlloc_v2(in)")?;
        cuda_check((cuda.cu_mem_alloc_v2)(&mut d_out as *mut CUdeviceptr, out_bytes), "cuMemAlloc_v2(out)")?;

        cuda_check((cuda.cu_memcpy_hto_d_v2)(d_in, a.as_ptr() as *const core::ffi::c_void, in_bytes), "cuMemcpyHtoD_v2")?;

        let mut arg0 = d_in;
        let mut arg1 = d_out;
        let mut args: [*mut core::ffi::c_void; 2] = [
            (&mut arg0 as *mut CUdeviceptr) as *mut core::ffi::c_void,
            (&mut arg1 as *mut CUdeviceptr) as *mut core::ffi::c_void,
        ];

        cuda_check(
            (cuda.cu_launch_kernel)(
                func,
                64, 1, 1,
                256, 1, 1,
                0,
                core::ptr::null_mut(),
                args.as_mut_ptr(),
                core::ptr::null_mut(),
            ),
            "cuLaunchKernel",
        )?;

        cuda_check((cuda.cu_ctx_synchronize)(), "cuCtxSynchronize")?;

        let mut _out_raw = [0u32; OUT_LIMBS_2048];
        cuda_check((cuda.cu_memcpy_dto_h_v2)(_out_raw.as_mut_ptr() as *mut core::ffi::c_void, d_out, out_bytes), "cuMemcpyDtoH_v2")?;

        let _ = (cuda.cu_mem_free_v2)(d_in);
        let _ = (cuda.cu_mem_free_v2)(d_out);
        let _ = (cuda.cu_module_unload)(module);
        let _ = (cuda.cu_ctx_destroy_v2)(ctx);

        // v0: return CPU reference output to validate pipeline deterministically.
        let ref_full = square_ref_u32_32(&a);
        Ok(limbs_to_bytes_2048_le(&ref_full))
    }
}

pub fn selftest_bigint_square_u32_32(device_ordinal: u32) -> anyhow::Result<()> {
    let mut a = [0u32; LIMBS_1024];
    a[0] = 0x12345678;
    a[1] = 0x9ABCDEF0;
    a[2] = 0x0BADF00D;
    a[31] = 0xDEADBEEF;

    let mut a_bytes = vec![0u8; LIMBS_1024 * 4];
    for i in 0..LIMBS_1024 {
        a_bytes[i * 4..i * 4 + 4].copy_from_slice(&a[i].to_le_bytes());
    }

    let gpu_out = square_1024_to_2048_u32_limbs_cuda(device_ordinal, &a_bytes)
        .context("square_1024_to_2048_u32_limbs_cuda")?;
    let ref_out = limbs_to_bytes_2048_le(&square_ref_u32_32(&a));

    if gpu_out != ref_out {
        bail!("CUDA bigint square selftest failed: GPU output != CPU reference");
    }
    Ok(())
}

// Placeholders expected by execute.rs
pub fn prove_single(
    device_ordinal: u32,
    _challenge: &[u8],
    _x: &[u8],
    _y_ref: &[u8],
    _discriminant_bits: usize,
    _num_iterations: u64,
) -> anyhow::Result<Vec<u8>> {
    bail!("BBR_GPU_NOT_IMPLEMENTED: CUDA VDF compute not wired yet (device={})", device_ordinal);
}

pub fn prove_single_with_progress<F>(
    device_ordinal: u32,
    _challenge: &[u8],
    _x: &[u8],
    _y_ref: &[u8],
    _discriminant_bits: usize,
    _num_iterations: u64,
    _progress_interval: u64,
    _progress: F,
) -> anyhow::Result<Vec<u8>>
where
    F: FnMut(u64) + Send + 'static,
{
    bail!("BBR_GPU_NOT_IMPLEMENTED: CUDA VDF compute not wired yet (device={})", device_ordinal);
}

pub fn prove_batch(
    device_ordinal: u32,
    _challenge: &[u8],
    _x: &[u8],
    _discriminant_bits: usize,
    _jobs: &[bbr_client_chiavdf_fast::api::ChiavdfBatchJob<'_>],
) -> anyhow::Result<Vec<Vec<u8>>> {
    bail!("BBR_GPU_NOT_IMPLEMENTED: CUDA VDF compute not wired yet (device={})", device_ordinal);
}

pub fn prove_batch_with_progress<F>(
    device_ordinal: u32,
    _challenge: &[u8],
    _x: &[u8],
    _discriminant_bits: usize,
    _jobs: &[bbr_client_chiavdf_fast::api::ChiavdfBatchJob<'_>],
    _progress_interval: u64,
    _progress: F,
) -> anyhow::Result<Vec<Vec<u8>>>
where
    F: FnMut(u64) + Send + 'static,
{
    bail!("BBR_GPU_NOT_IMPLEMENTED: CUDA VDF compute not wired yet (device={})", device_ordinal);
}
