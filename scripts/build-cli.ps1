#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

# Déterminer le répertoire racine correctement
$ScriptPath = $MyInvocation.MyCommand.Path
$ScriptDir = Split-Path -Parent $ScriptPath
$Root = Split-Path -Parent $ScriptDir  # Remonter d'un niveau pour sortir de "builder"
Set-Location $Root


function Test-CommandExists {
    param([Parameter(Mandatory = $true)][string]$Command)
    return $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

function Get-PackageManager {
    if (Test-CommandExists "winget") { return "winget" }
    if (Test-CommandExists "choco") { return "choco" }
    if (Test-CommandExists "scoop") { return "scoop" }
    return $null
}

function Test-IsElevated {
    try {
        $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    catch { return $false }
}

function Test-MsvcBuildToolsPresent {
    $candidates = @(
        "$env:ProgramFiles\Microsoft Visual Studio\2022\BuildTools",
        "$env:ProgramFiles(x86)\Microsoft Visual Studio\2022\BuildTools",
        "$env:ProgramFiles\Microsoft Visual Studio\2022\Community",
        "$env:ProgramFiles(x86)\Microsoft Visual Studio\2022\Community"
    )
    foreach ($c in $candidates) {
        if ($c -and (Test-Path -LiteralPath $c)) {
            $vcTools = Join-Path $c "VC\Tools\MSVC"
            if (Test-Path -LiteralPath $vcTools) { return $true }
        }
    }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        try {
            $out = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
            if ($out -and (Test-Path -LiteralPath $out.Trim())) { return $true }
        }
        catch { }
    }
    return $false
}

function Get-InstallConsent {
    param([Parameter(Mandatory = $true)][string[]]$Missing, [string]$PackageManager)

    $auto = $env:BBR_PREREQ_AUTO
    if ($auto) {
        switch ($auto.ToLowerInvariant()) {
            "1" { return $true }
            "true" { return $true }
            "yes" { return $true }
            "y" { return $true }
            "0" { return $false }
            "false" { return $false }
            "no" { return $false }
            "n" { return $false }
        }
    }

    Write-Host ""
    Write-Host "=== Prérequis: WesoForge CLI ===" -ForegroundColor Yellow
    Write-Host "Manquants :" -ForegroundColor Yellow
    foreach ($m in $Missing) { Write-Host "  - $m" -ForegroundColor Yellow }
    Write-Host ""
    Write-Host "Variables: BBR_PREREQ_AUTO=1 (installer sans prompt) / BBR_PREREQ_AUTO=0 (ne jamais installer)." -ForegroundColor DarkGray

    if (-not $PackageManager) {
        Write-Host "Aucun gestionnaire de paquets détecté (winget/choco/scoop)." -ForegroundColor Red
        return $false
    }

    if (($PackageManager -in @("winget", "choco")) -and (-not (Test-IsElevated))) {
        Write-Host "Note: l'installation via $PackageManager peut nécessiter une PowerShell en administrateur." -ForegroundColor Yellow
    }

    $ans = Read-Host "Autoriser l'installation automatique de TOUS les composants manquants via '$PackageManager' ? (y/N)"
    return ($ans -match '^(y|yes)$')
}

function Install-WithWinget { param([string]$Id) winget install --id $Id -e --source winget --accept-package-agreements --accept-source-agreements }
function Install-WithChoco { param([string]$Name) choco install $Name -y }
function Install-WithScoop { param([string]$Name) scoop install $Name }

function Ensure-Prerequisites-Cli {
    $pm = Get-PackageManager
    $missing = New-Object System.Collections.Generic.List[string]

    if (-not (Test-CommandExists "cargo") -or -not (Test-CommandExists "rustc")) { $missing.Add("Rust toolchain (rustup + cargo)") }
    if (-not (Test-MsvcBuildToolsPresent)) { $missing.Add("MSVC Build Tools (Windows)") }

    if ($missing.Count -eq 0) { return }

    if (-not (Get-InstallConsent -Missing $missing.ToArray() -PackageManager $pm)) {
        throw "Missing prerequisites."
    }

    switch ($pm) {
        "winget" {
            if (-not (Test-CommandExists "rustup")) { Install-WithWinget "Rustlang.Rustup" }
            if (-not (Test-MsvcBuildToolsPresent)) {
                winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --accept-package-agreements --accept-source-agreements --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
            }
        }
        "choco" {
            if (-not (Test-CommandExists "rustup")) { Install-WithChoco "rustup.install" }
            if (-not (Test-MsvcBuildToolsPresent)) {
                choco install visualstudio2022buildtools -y
                choco install visualstudio2022-workload-vctools -y
            }
        }
        "scoop" {
            if (-not (Test-CommandExists "rustup")) { Install-WithScoop "rustup" }
        }
        default { throw "No package manager available." }
    }
}

Ensure-Prerequisites-Cli

function Get-WorkspaceVersion {
    $inPkg = $false
    foreach ($line in Get-Content -LiteralPath (Join-Path $Root "Cargo.toml")) {
        if ($line -match '^\[workspace\.package\]') {
            $inPkg = $true
            continue
        }
        if ($inPkg -and $line -match '^\[') {
            $inPkg = $false
        }
        if ($inPkg -and $line -match '^version\s*=\s*"([^"]+)"') {
            return $Matches[1]
        }
    }
    throw "Failed to determine workspace version from Cargo.toml"
}

function Get-PlatformArch {
    $arch = $null
    try {
        $osArch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
        if ($null -ne $osArch) {
            $arch = $osArch.ToString()
        }
    }
    catch {
        # Fall back to env vars below.
    }

    if ([string]::IsNullOrWhiteSpace($arch)) {
        # Some Windows environments (notably 32-bit PowerShell or older runtimes) can fail to
        # provide RuntimeInformation.OSArchitecture. Fall back to environment variables.
        $arch = $env:PROCESSOR_ARCHITECTURE
        if ($arch -eq "x86" -and $env:PROCESSOR_ARCHITEW6432) {
            $arch = $env:PROCESSOR_ARCHITEW6432
        }
    }

    if ([string]::IsNullOrWhiteSpace($arch)) {
        throw "Failed to determine platform architecture (RuntimeInformation + PROCESSOR_ARCHITECTURE are unavailable)."
    }
    switch ($arch) {
        "X64" { return "amd64" }
        "AMD64" { return "amd64" }
        "Arm64" { return "arm64" }
        "ARM64" { return "arm64" }
        "x86" { return "x86" }
        default { return $arch.ToLowerInvariant() }
    }
}

$DistDir = $env:DIST_DIR
if ([string]::IsNullOrWhiteSpace($DistDir)) {
    $DistDir = Join-Path $Root "dist"
}
New-Item -ItemType Directory -Force -Path $DistDir | Out-Null

if ($env:BBR_SKIP_CARGO_BUILD -ne "1") {
    Write-Host "Building WesoForge CLI (wesoforge)..." -ForegroundColor Cyan
    cargo build -p bbr-client --release --features prod-backend
}

$Version = Get-WorkspaceVersion
$Arch = Get-PlatformArch

$TargetDir = $env:CARGO_TARGET_DIR
if ([string]::IsNullOrWhiteSpace($TargetDir)) {
    $TargetDir = Join-Path $Root "target"
}

$BinSrc = Join-Path $TargetDir "release\\wesoforge.exe"
if (!(Test-Path -LiteralPath $BinSrc)) {
    throw "Expected binary not found at: $BinSrc"
}

$BinDst = Join-Path $DistDir ("WesoForge-cli_Windows_{0}_{1}.exe" -f $Version, $Arch)
Copy-Item -Force -LiteralPath $BinSrc -Destination $BinDst

# MPIR runtime DLLs (required by chiavdf on Windows).
$MpirDir = Join-Path $Root "chiavdf\\mpir_gc_x64"
if (!(Test-Path -LiteralPath $MpirDir)) {
    throw "MPIR directory not found at: $MpirDir"
}
Copy-Item -Force -Path (Join-Path $MpirDir "mpir*.dll") -Destination $DistDir

Write-Host "Wrote: $BinDst" -ForegroundColor Green
