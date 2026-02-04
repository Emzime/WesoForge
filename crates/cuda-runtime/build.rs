// Comments in English as requested.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Rebuild if the kernel source changes.
    println!("cargo:rerun-if-changed=kernels/vdf.cu");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let out_ptx = out_dir.join("vdf.ptx");

    // Try to compile the CUDA C kernel to PTX using nvcc.
    // If nvcc is missing, fall back to the committed PTX in src/vdf_fallback.ptx.
    let nvcc_ok = try_compile_with_nvcc(&out_ptx).unwrap_or(false);
    if nvcc_ok {
        return;
    }

    // Fallback: copy the embedded PTX into OUT_DIR.
    let fallback = Path::new("src/vdf_fallback.ptx");
    let bytes = fs::read(fallback).expect("failed to read src/vdf_fallback.ptx");
    fs::write(&out_ptx, bytes).expect("failed to write fallback PTX to OUT_DIR");
}

fn try_compile_with_nvcc(out_ptx: &Path) -> anyhow::Result<bool> {
    let status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("kernels/vdf.cu")
        .arg("-o")
        .arg(out_ptx)
        .status();

    match status {
        Ok(s) if s.success() => Ok(true),
        Ok(_) => Ok(false),
        Err(_) => Ok(false),
    }
}
