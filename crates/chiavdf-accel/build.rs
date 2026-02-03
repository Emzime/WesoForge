// Comments in English as requested.

use std::env;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();

    // Always rerun if kernels/native code change
    println!("cargo:rerun-if-changed=kernels/cuda/vdf_batch.cu");

    // CUDA feature gate
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

    if cuda_enabled {
        // Heuristic: NVCC in PATH or CUDA_PATH set.
        let cuda_path = env::var("CUDA_PATH").ok();
        let nvcc = which::which("nvcc").ok();

        if nvcc.is_none() && cuda_path.is_none() {
            // Build succeeds but backend will be NotAvailable at runtime if used.
            println!("cargo:warning=CUDA feature enabled but nvcc/CUDA_PATH not found; building without CUDA objects");
            return;
        }

        // Compile CUDA kernel into an object and archive it.
        // On Windows, nvcc produces .obj; on Linux .o.
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let kernel_src = PathBuf::from("kernels/cuda/vdf_batch.cu");

        // Use `cc` crate for C++ compilation; for CUDA we call nvcc directly.
        // Keeping this simple; in production you may want robust flags per target/arch.
        let obj_path = out_dir.join(if target.contains("windows") { "vdf_batch.obj" } else { "vdf_batch.o" });

        let mut cmd = std::process::Command::new(nvcc.map(|p| p.to_string_lossy().to_string()).unwrap_or_else(|| "nvcc".to_string()));
        cmd.arg("-c")
            .arg(kernel_src)
            .arg("-O3")
            .arg("--use_fast_math")
            .arg("-Xcompiler")
            .arg(if target.contains("windows") { "/MD" } else { "-fPIC" })
            .arg("-o")
            .arg(&obj_path);

        let status = cmd.status().expect("failed to run nvcc");
        if !status.success() {
            panic!("nvcc failed compiling vdf_batch.cu");
        }

        // Archive into a static lib via `ar` (Linux) or `lib.exe` (Windows).
        if target.contains("windows") {
            let lib_exe = env::var("LIB_EXE").unwrap_or_else(|_| "lib.exe".to_string());
            let lib_path = out_dir.join("chiavdf_accel_cuda.lib");

            let status = std::process::Command::new(lib_exe)
                .arg(format!("/OUT:{}", lib_path.display()))
                .arg(obj_path)
                .status()
                .expect("failed to run lib.exe");
            if !status.success() {
                panic!("lib.exe failed creating chiavdf_accel_cuda.lib");
            }

            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=chiavdf_accel_cuda");
        } else {
            let lib_path = out_dir.join("libchiavdf_accel_cuda.a");
            let status = std::process::Command::new("ar")
                .arg("crus")
                .arg(&lib_path)
                .arg(&obj_path)
                .status()
                .expect("failed to run ar");
            if !status.success() {
                panic!("ar failed creating libchiavdf_accel_cuda.a");
            }

            println!("cargo:rustc-link-search=native={}", out_dir.display());
            println!("cargo:rustc-link-lib=static=chiavdf_accel_cuda");
        }
    }
}
