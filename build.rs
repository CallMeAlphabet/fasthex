use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo::rustc-check-cfg=cfg(simd_portable)");

    if portable_simd_supported() {
        println!("cargo:rustc-cfg=simd_portable");
    }
}

fn portable_simd_supported() -> bool {
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());

    let Ok(out) = Command::new(rustc).arg("-vV").output() else {
        return false;
    };
    if !out.status.success() {
        return false;
    }

    String::from_utf8_lossy(&out.stdout)
        .lines()
        .find_map(|line| line.strip_prefix("release: "))
        .is_some_and(|release| release.contains("nightly") || release.contains("-dev"))
}
