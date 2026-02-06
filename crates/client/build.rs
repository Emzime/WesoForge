use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=../../chiavdf/mpir_gc_x64");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .map(Path::to_path_buf)
        .expect("workspace root");

    let mpir_dir = workspace_root.join("chiavdf").join("mpir_gc_x64");
    if !mpir_dir.exists() {
        // Don't spam warnings; runtime will fail anyway if MPIR is missing.
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
            continue;
        }

        let dst = target_profile_dir.join(dll);

        // Copy only if missing or different size to reduce I/O
        let do_copy = match (fs::metadata(&src), fs::metadata(&dst)) {
            (Ok(ms), Ok(md)) => ms.len() != md.len(),
            (Ok(_), Err(_)) => true,
            _ => false,
        };

        if do_copy {
            let _ = fs::copy(&src, &dst);
        }
    }
}
