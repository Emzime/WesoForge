fn main() {
    // `chiavdf`'s generated assembly isn't PIE-safe, and `bbr-client-gui` links
    // it via the engine. Disable PIE on Linux so the final link succeeds.
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "linux" {
        println!("cargo:rustc-link-arg=-no-pie");
    }
    tauri_build::build()
}
