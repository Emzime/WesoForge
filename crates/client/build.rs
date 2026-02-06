use std::path::{Path, PathBuf};

fn main() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if target_os == "linux" {
        // The chiavdf fast wrapper bundles prebuilt assembly objects that are not PIE/PIC-safe.
        // Rust defaults to PIE on many Linux distros, so we disable PIE for this binary.
        println!("cargo:rustc-link-arg-bin=wesoforge=-no-pie");
        return;
    }

    if target_os == "windows" {
        // Copy MPIR runtime DLLs next to the produced binary so wesoforge.exe can start.
        // This prevents STATUS_DLL_NOT_FOUND at runtime.
        if let Err(err) = copy_mpir_runtime_dlls() {
            println!("cargo:warning=MPIR DLL copy failed: {err}");
        }
    }
}

fn copy_mpir_runtime_dlls() -> Result<(), String> {
    println!("cargo:rerun-if-env-changed=MPIR_BIN");
    println!("cargo:rerun-if-env-changed=MPIR_DIR");
    println!("cargo:rerun-if-env-changed=VCPKG_ROOT");
    println!("cargo:rerun-if-env-changed=CARGO_TARGET_DIR");

    let out_dir = PathBuf::from(env_var("OUT_DIR")?);
    let target_bin_dir = infer_target_bin_dir(&out_dir)?;

    // Workspace root is the parent of crates/client
    let manifest_dir = PathBuf::from(env_var("CARGO_MANIFEST_DIR")?);
    let workspace_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest_dir.clone());

    let mut candidates: Vec<PathBuf> = Vec::new();

    if let Ok(p) = std::env::var("MPIR_BIN") {
        candidates.push(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("MPIR_DIR") {
        let base = PathBuf::from(p);
        candidates.push(base.join("bin"));
        candidates.push(base);
    }
    if let Ok(p) = std::env::var("VCPKG_ROOT") {
        let base = PathBuf::from(p);
        // Most common vcpkg triplets on Windows
        candidates.push(base.join("installed").join("x64-windows").join("bin"));
        candidates.push(base.join("installed").join("x64-windows-static").join("bin"));
    }

    // Common repo-local locations (WesoForge layout)
    candidates.push(workspace_root.join("chiavdf").join("mpir_gc_x64"));
    candidates.push(workspace_root.join("chiavdf").join("mpir_gc_x64").join("bin"));
    candidates.push(workspace_root.join("chiavdf").join("mpir_gc_x64").join("lib"));
    candidates.push(workspace_root.join("crates").join("chiavdf-fast").join("native"));

    // Also consider CARGO_TARGET_DIR if set (binary dir is inside it)
    if let Ok(p) = std::env::var("CARGO_TARGET_DIR") {
        candidates.push(PathBuf::from(p));
    }

    // Find first directory that contains mpir*.dll
    let (src_dir, dlls) = find_mpir_dlls(&candidates)
        .ok_or_else(|| "could not locate mpir*.dll in any known location".to_string())?;

    // Ensure destination exists and copy
    std::fs::create_dir_all(&target_bin_dir)
        .map_err(|e| format!("create_dir_all({}): {e}", target_bin_dir.display()))?;

    for dll in dlls {
        let src = src_dir.join(&dll);
        let dst = target_bin_dir.join(&dll);

        std::fs::copy(&src, &dst)
            .map_err(|e| format!("copy {} -> {}: {e}", src.display(), dst.display()))?;

        println!("cargo:warning=Copied {} -> {}", src.display(), dst.display());
    }

    Ok(())
}

fn find_mpir_dlls(candidates: &[PathBuf]) -> Option<(PathBuf, Vec<String>)> {
    for dir in candidates {
        if !dir.is_dir() {
            continue;
        }

        // Best-effort: rerun if these directories change
        println!("cargo:rerun-if-changed={}", dir.display());

        let mut dlls: Vec<String> = Vec::new();

        let rd = std::fs::read_dir(dir).ok()?;
        for entry in rd.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };

            let lower = name.to_ascii_lowercase();
            if lower.starts_with("mpir") && lower.ends_with(".dll") {
                dlls.push(name);
            }
        }

        if !dlls.is_empty() {
            dlls.sort();
            dlls.dedup();
            return Some((dir.clone(), dlls));
        }
    }
    None
}

fn infer_target_bin_dir(out_dir: &Path) -> Result<PathBuf, String> {
    // OUT_DIR example:
    // <workspace>\target\release\build\<crate-hash>\out
    // We want: <workspace>\target\release
    let mut cur = out_dir;

    while let Some(parent) = cur.parent() {
        if parent.file_name().and_then(|n| n.to_str()) == Some("build") {
            // cur is ".../target/<profile>/build/<...>"
            if let Some(profile_dir) = parent.parent() {
                return Ok(profile_dir.to_path_buf());
            }
        }
        cur = parent;
    }

    Err(format!(
        "failed to infer target bin dir from OUT_DIR={}",
        out_dir.display()
    ))
}

fn env_var(key: &str) -> Result<String, String> {
    std::env::var(key).map_err(|e| format!("missing env {key}: {e}"))
}
