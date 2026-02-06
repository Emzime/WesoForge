# Cargo_test.ps1
# Interactive helper to build/test with optional features.
# Logs are written to ./logs/

$ErrorActionPreference = "Stop"

function Ensure-LogsDir {
    if (-not (Test-Path "logs")) {
        New-Item -ItemType Directory -Path "logs" | Out-Null
    }
}

function Read-Choice {
    param(
        [string]$Prompt,
        [hashtable]$Options,
        [string]$DefaultKey = ""
    )

    Write-Host ""
    Write-Host $Prompt -ForegroundColor Yellow
    foreach ($k in ($Options.Keys | Sort-Object)) {
        Write-Host "  [$k] $($Options[$k])"
    }

    while ($true) {
        $suffix = ""
        if ($DefaultKey -ne "") { $suffix = " (default: $DefaultKey)" }
        $input = Read-Host "Select$suffix"
        if ([string]::IsNullOrWhiteSpace($input)) {
            if ($DefaultKey -ne "") { return $DefaultKey }
            continue
        }
        if ($Options.ContainsKey($input)) { return $input }
        Write-Host "Invalid choice. Try again." -ForegroundColor Red
    }
}

function Read-YesNo {
    param(
        [string]$Prompt,
        [bool]$DefaultYes = $false
    )

    $def = if ($DefaultYes) { "Y" } else { "N" }
    while ($true) {
        $input = Read-Host "$Prompt (Y/N) (default: $def)"
        if ([string]::IsNullOrWhiteSpace($input)) { return $DefaultYes }
        switch ($input.Trim().ToUpperInvariant()) {
            "Y" { return $true }
            "N" { return $false }
            default { Write-Host "Please answer Y or N." -ForegroundColor Red }
        }
    }
}

function Read-NonEmpty {
    param(
        [string]$Prompt,
        [string]$DefaultValue = ""
    )

    while ($true) {
        $suffix = ""
        if ($DefaultValue -ne "") { $suffix = " (default: $DefaultValue)" }
        $input = Read-Host "$Prompt$suffix"
        if ([string]::IsNullOrWhiteSpace($input)) {
            if ($DefaultValue -ne "") { return $DefaultValue }
            continue
        }
        return $input.Trim()
    }
}

Ensure-LogsDir

$package = Read-NonEmpty -Prompt "Package (-p)" -DefaultValue "bbr-client-engine"

$mode = Read-Choice -Prompt "Mode" -Options @{
    "1" = "cargo build"
    "2" = "cargo test"
} -DefaultKey "1"

$profile = Read-Choice -Prompt "Profile" -Options @{
    "1" = "Debug"
    "2" = "Release (--release)"
} -DefaultKey "1"

$doClean = Read-YesNo -Prompt "Run cargo clean first?" -DefaultYes $false
$noDefaultFeatures = Read-YesNo -Prompt "Use --no-default-features?" -DefaultYes $false

$featureChoice = Read-Choice -Prompt "Features selection" -Options @{
    "0" = "No features"
    "1" = "cuda"
    "2" = "opencl"
    "3" = "cuda,opencl"
    "4" = "Custom (comma-separated)"
} -DefaultKey "0"

$features = ""
switch ($featureChoice) {
    "0" { $features = "" }
    "1" { $features = "cuda" }
    "2" { $features = "opencl" }
    "3" { $features = "cuda,opencl" }
    "4" { $features = Read-NonEmpty -Prompt "Enter features (comma-separated, e.g. cuda,opencl)" -DefaultValue "" }
}

$setEnvGpu = Read-Choice -Prompt "Set WESOFORGE_GPU env for this run?" -Options @{
    "0" = "Do not set"
    "1" = "auto"
    "2" = "cuda"
    "3" = "opencl"
    "4" = "off"
} -DefaultKey "0"

switch ($setEnvGpu) {
    "0" { }
    "1" { $env:WESOFORGE_GPU = "auto" }
    "2" { $env:WESOFORGE_GPU = "cuda" }
    "3" { $env:WESOFORGE_GPU = "opencl" }
    "4" { $env:WESOFORGE_GPU = "off" }
}

$env:WESOFORGE_TEST_ITERS = "tiny"

$logFile = "logs/cargo_test_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Host ""
Write-Host "Log: $logFile" -ForegroundColor Yellow

if ($doClean) {
    Write-Host "Running: cargo clean" -ForegroundColor DarkYellow
    cargo clean 2>&1 | Tee-Object -FilePath $logFile
}

$cargoArgs = @()

if ($mode -eq "2") {
    $cargoArgs += @("test", "-p", $package)
} else {
    $cargoArgs += @("build", "-p", $package)
}

if ($profile -eq "2") {
    $cargoArgs += "--release"
}

if ($noDefaultFeatures) {
    $cargoArgs += "--no-default-features"
}

$trimmed = $features.Trim()
if ($trimmed.Length -gt 0) {
    $featureList = $trimmed.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
    if ($featureList.Count -gt 0) {
        $cargoArgs += @("--features", ($featureList -join ","))
    }
}

if ($mode -eq "2") {
    $cargoArgs += @("--", "--nocapture")
}

Write-Host ""
Write-Host ("Command: cargo " + ($cargoArgs -join " ")) -ForegroundColor Cyan

cargo @cargoArgs 2>&1 | Tee-Object -FilePath $logFile

Write-Host ""
Write-Host "Done. Log saved to: $logFile" -ForegroundColor Green
