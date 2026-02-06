use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    // Re-run if MPIR DLLs change.
    println!("cargo:rerun-if-changed=../../chiavdf/mpir_gc_x64");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .map(Path::to_path_buf)
        .expect("workspace root");

    let mpir_dir = workspace_root.join("chiavdf").join("mpir_gc_x64");
    if !mpir_dir.exists() {
        println!(
            "cargo:warning=MPIR directory not found: {}",
            mpir_dir.display()
        );
        return;
    }

    // OUT_DIR example:
    // <workspace>/target/release/build/<crate>/out
    // ancestors():
    // 0 out
    // 1 <crate>
    // 2 build
    // 3 release   <-- wanted
    // 4 target
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let target_profile_dir = out_dir
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .expect("target/<profile> dir");

    let dlls = [
        "mpir.dll",
        "mpir_broadwell.dll",
        "mpir_broadwell_avx.dll",
        "mpir_bulldozer.dll",
        "mpir_gc.dll",
        "mpir_haswell.dll",
        "mpir_piledriver.dll",
        "mpir_sandybridge.dll",
        "mpir_skylake_avx.dll",
    ];

    for dll in dlls {
        let src = mpir_dir.join(dll);
        if !src.exists() {
            println!("cargo:warning=MPIR DLL missing: {}", src.display());
            continue;
        }

        let dst = target_profile_dir.join(dll);
        if let Err(e) = fs::copy(&src, &dst) {
            println!(
                "cargo:warning=Failed to copy {} -> {}: {e}",
                src.display(),
                dst.display()
            );
            continue;
        }

        println!("cargo:warning=Copied {} -> {}", src.display(), dst.display());
    }
}
