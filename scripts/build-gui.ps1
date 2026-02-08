#!/usr/bin/env pwsh
$ErrorActionPreference = "Stop"

# Déterminer le répertoire racine correctement
$ScriptPath = $MyInvocation.MyCommand.Path
$ScriptDir = Split-Path -Parent $ScriptPath
$Root = Split-Path -Parent $ScriptDir  # Remonter d'un niveau pour sortir de "builder"
Set-Location $Root

#region Build logging (WesoForge\logs\build-gui_*.log)

$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir ("build-gui_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))

function Write-Log {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Message,

        [ValidateSet("ERROR", "WARN", "INFO", "SUCCESS", "DEBUG")]
        [string]$Level = "INFO",

        [switch]$NoConsole
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"

    # Write to log file
    $logEntry | Out-File -FilePath $LogFile -Append -Encoding UTF8

    # Write to console unless suppressed
    if (-not $NoConsole) {
        switch ($Level) {
            "ERROR" { Write-Host $Message -ForegroundColor Red }
            "WARN" { Write-Host $Message -ForegroundColor Yellow }
            "INFO" { Write-Host $Message -ForegroundColor Cyan }
            "SUCCESS" { Write-Host $Message -ForegroundColor Green }
            "DEBUG" { Write-Host $Message -ForegroundColor Gray }
            default { Write-Host $Message }
        }
    }
}

function Log-Error {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [Exception]$Exception,
        [System.Management.Automation.ErrorRecord]$ErrorRecord
    )

    Write-Log -Message $Message -Level "ERROR"
    if ($Exception) {
        Write-Log -Message ("Exception: {0}" -f $Exception.Message) -Level "ERROR"
        if ($Exception.StackTrace) {
            Write-Log -Message ("StackTrace: {0}" -f $Exception.StackTrace) -Level "DEBUG" -NoConsole
        }
    }

    # Capture the full error record for the log file (often contains useful context).
    if ($ErrorRecord) {
        $details = $ErrorRecord | Out-String
        if (-not [string]::IsNullOrWhiteSpace($details)) {
            foreach ($l in ($details -split "`r?`n")) {
                if (-not [string]::IsNullOrWhiteSpace($l)) {
                    Write-Log -Message ("PSERROR: {0}" -f $l.TrimEnd()) -Level "DEBUG" -NoConsole
                }
            }
        }
    }
}

function Initialize-Logging {
    if (-not (Test-Path -LiteralPath $LogDir)) {
        New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    }

    # Create an empty file (overwrite if exists)
    "" | Out-File -FilePath $LogFile -Encoding UTF8

    Write-Host ("Log file: {0}" -f $LogFile) -ForegroundColor Gray
    Write-Log -Message "=== WesoForge GUI Build Started ===" -Level "INFO"
    Write-Log -Message ("Root: {0}" -f $Root) -Level "DEBUG" -NoConsole
    Write-Log -Message ("PowerShell: {0}" -f $PSVersionTable.PSVersion) -Level "DEBUG" -NoConsole
    Write-Log -Message ("Computer: {0}" -f $env:COMPUTERNAME) -Level "DEBUG" -NoConsole
    Write-Log -Message ("Username: {0}" -f $env:USERNAME) -Level "DEBUG" -NoConsole
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$CommandLabel,
        [Parameter(Mandatory = $true)][ScriptBlock]$Command
    )

    Write-Log -Message ("Running: {0}" -f $CommandLabel) -Level "INFO"

    & $Command 2>&1 | ForEach-Object {
        $line = $_.ToString()

        # Always print raw output lines (including empty ones) to console,
        # but never pass empty/whitespace-only lines into Write-Log.
        Write-Host $line

        if (-not [string]::IsNullOrWhiteSpace($line)) {
            Write-Log -Message $line -Level "DEBUG" -NoConsole
        }
    }

    if ($LASTEXITCODE -ne 0) {
        throw ("Command failed ({0}) with exit code {1}" -f $CommandLabel, $LASTEXITCODE)
    }
}

Initialize-Logging

#endregion


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
    Ensure-Prerequisites-Gui
        $currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($currentIdentity)
        return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    }
    catch {
        return $false
    }
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
            if (Test-Path -LiteralPath $vcTools) {
                return $true
            }
        }
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        try {
            $out = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
            if ($out -and (Test-Path -LiteralPath $out.Trim())) { return $true }
        }
        catch {
            # Ignore
        }
    }

    return $false
}

function Show-PrereqBanner {
    param(
        [Parameter(Mandatory = $true)][string]$ScriptName,
        [Parameter(Mandatory = $true)][string[]]$Required
    )

    Write-Host ""
    Write-Host "=== Prérequis: $ScriptName ===" -ForegroundColor Yellow
    Write-Host "Ce script nécessite les composants suivants :" -ForegroundColor Yellow
    foreach ($r in $Required) {
        Write-Host "  - $r" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "Le script peut installer automatiquement les composants manquants (avec votre autorisation)." -ForegroundColor Yellow
    Write-Host "Variables: BBR_PREREQ_AUTO=1 (installer sans prompt) / BBR_PREREQ_AUTO=0 (ne jamais installer)." -ForegroundColor DarkGray
    Write-Host ""
}

function Get-InstallConsent {
    param(
        [Parameter(Mandatory = $true)][string[]]$Missing,
        [Parameter(Mandatory = $true)][string]$PackageManager
    )

    if ($Missing.Count -eq 0) { return $false }

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

    Write-Host "Composants manquants :" -ForegroundColor Yellow
    foreach ($m in $Missing) {
        Write-Host "  - $m" -ForegroundColor Yellow
    }
    Write-Host ""

    if (-not $PackageManager) {
        Write-Host "Aucun gestionnaire de paquets détecté (winget/choco/scoop)." -ForegroundColor Red
        Write-Host "Installez manuellement les prérequis, ou installez winget (App Installer) / Chocolatey / Scoop." -ForegroundColor Red
        return $false
    }

    if (($PackageManager -in @("winget", "choco")) -and (-not (Test-IsElevated))) {
        Write-Host "Note: l'installation via $PackageManager peut nécessiter une PowerShell en administrateur." -ForegroundColor Yellow
    }

    $ans = Read-Host "Autoriser l'installation automatique de TOUS les composants manquants via '$PackageManager' ? (y/N)"
    return ($ans -match '^(y|yes)$')
}

function Install-WithWinget {
    param([Parameter(Mandatory = $true)][string]$Id)
    Invoke-LoggedCommand -CommandLabel "winget install $Id" -Command {
        winget install --id $Id -e --source winget --accept-package-agreements --accept-source-agreements
    }
}

function Install-WithChoco {
    param([Parameter(Mandatory = $true)][string]$Name)
    Invoke-LoggedCommand -CommandLabel "choco install $Name" -Command {
        choco install $Name -y
    }
}

function Install-WithScoop {
    param([Parameter(Mandatory = $true)][string]$Name)
    Invoke-LoggedCommand -CommandLabel "scoop install $Name" -Command {
        scoop install $Name
    }
}

function Ensure-Rust {
    param([Parameter(Mandatory = $true)][string]$Pm)

    if (Test-CommandExists "cargo" -and Test-CommandExists "rustc") { return }

    switch ($Pm) {
        "winget" { Install-WithWinget -Id "Rustlang.Rustup" }
        "choco" { Install-WithChoco -Name "rustup.install" }
        "scoop" { Install-WithScoop -Name "rustup" }
        default { throw "No package manager available for Rust toolchain." }
    }

    # Try to refresh PATH for current session.
    if (-not (Test-CommandExists "cargo")) {
        $cargoHome = $env:CARGO_HOME
        if ([string]::IsNullOrWhiteSpace($cargoHome)) { $cargoHome = Join-Path $HOME ".cargo" }
        $cargoBin = Join-Path $cargoHome "bin"
        if (Test-Path -LiteralPath $cargoBin) { $env:PATH = "$cargoBin;$env:PATH" }
    }
}

function Ensure-Git {
    param([Parameter(Mandatory = $true)][string]$Pm)

    if (Test-CommandExists "git") { return }

    switch ($Pm) {
        "winget" { Install-WithWinget -Id "Git.Git" }
        "choco" { Install-WithChoco -Name "git" }
        "scoop" { Install-WithScoop -Name "git" }
        default { throw "No package manager available for Git." }
    }
}

function Ensure-Node {
    param([Parameter(Mandatory = $true)][string]$Pm)

    if (Test-CommandExists "node") { return }

    switch ($Pm) {
        "winget" { Install-WithWinget -Id "OpenJS.NodeJS.LTS" }
        "choco" { Install-WithChoco -Name "nodejs-lts" }
        "scoop" { Install-WithScoop -Name "nodejs-lts" }
        default { throw "No package manager available for Node.js." }
    }
}

function Ensure-Pnpm {
    if (Test-CommandExists "pnpm") { return }

    if (Test-CommandExists "corepack") {
        Invoke-LoggedCommand -CommandLabel "corepack enable" -Command { corepack enable }
        Invoke-LoggedCommand -CommandLabel "corepack prepare pnpm@latest --activate" -Command { corepack prepare pnpm@latest --activate }
        return
    }

    if (Test-CommandExists "npm") {
        Invoke-LoggedCommand -CommandLabel "npm i -g pnpm" -Command { npm i -g pnpm }
        return
    }

    throw "pnpm missing and neither corepack nor npm are available. Install Node.js first."
}

function Ensure-TauriCli {
    try {
        $null = cargo tauri --version 2>$null
        return
    }
    catch {
        # continue
    }

    Invoke-LoggedCommand -CommandLabel "cargo install tauri-cli" -Command {
        cargo install tauri-cli --locked
    }
}

function Ensure-MsvcBuildTools {
    param([Parameter(Mandatory = $true)][string]$Pm)

    if (Test-MsvcBuildToolsPresent) { return }

    Write-Log -Message "MSVC Build Tools not detected. Attempting install..." -Level "WARN"

    switch ($Pm) {
        "winget" {
            Invoke-LoggedCommand -CommandLabel "winget install Microsoft.VisualStudio.2022.BuildTools" -Command {
                winget install --id Microsoft.VisualStudio.2022.BuildTools -e --source winget --accept-package-agreements --accept-source-agreements --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
            }
        }
        "choco" {
            Invoke-LoggedCommand -CommandLabel "choco install visualstudio2022buildtools" -Command { choco install visualstudio2022buildtools -y }
            Invoke-LoggedCommand -CommandLabel "choco install visualstudio2022-workload-vctools" -Command { choco install visualstudio2022-workload-vctools -y }
        }
        default {
            Write-Log -Message "Cannot auto-install MSVC Build Tools with package manager '$Pm'." -Level "WARN"
        }
    }
}

function Ensure-Prerequisites-Gui {
    $required = @(
        "Git",
        "Rust toolchain (rustup + cargo)",
        "Node.js",
        "pnpm",
        "Tauri CLI (cargo tauri)",
        "MSVC Build Tools (Windows)"
    )

    Show-PrereqBanner -ScriptName "WesoForge GUI build" -Required $required

    $pm = Get-PackageManager

    $missing = New-Object System.Collections.Generic.List[string]
    if (-not (Test-CommandExists "git")) { $missing.Add("Git") }
    if (-not (Test-CommandExists "cargo") -or -not (Test-CommandExists "rustc")) { $missing.Add("Rust toolchain") }
    if (-not (Test-CommandExists "node")) { $missing.Add("Node.js") }
    if (-not (Test-CommandExists "pnpm") -and -not (Test-CommandExists "corepack") -and -not (Test-CommandExists "npm")) { $missing.Add("pnpm (needs Node)") }
    try { $null = cargo tauri --version 2>$null } catch { $missing.Add("Tauri CLI") }
    if (-not (Test-MsvcBuildToolsPresent)) { $missing.Add("MSVC Build Tools") }

    if ($missing.Count -eq 0) {
        Write-Log -Message "All prerequisites already present." -Level "SUCCESS"
        return
    }

    $consent = Get-InstallConsent -Missing $missing.ToArray() -PackageManager $pm
    if (-not $consent) {
        Write-Log -Message "Missing prerequisites and installation not authorized. Aborting before compilation." -Level "ERROR"
        throw "Missing prerequisites."
    }

    if (-not $pm) {
        throw "Missing prerequisites and no package manager available."
    }

    Write-Log -Message "Installing missing prerequisites using '$pm' (authorized)..." -Level "INFO"
    Ensure-Git -Pm $pm
    Ensure-Rust -Pm $pm
    Ensure-Node -Pm $pm
    Ensure-Pnpm
    Ensure-MsvcBuildTools -Pm $pm
    Ensure-TauriCli

    Write-Log -Message "Prerequisite install phase completed." -Level "SUCCESS"
}



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

function Get-WebView2ArchFolder([string] $arch) {
    switch ($arch) {
        "amd64" { return "x64" }
        "arm64" { return "arm64" }
        "x86" { return "x86" }
        default { throw "Unsupported architecture for WebView2Loader.dll lookup: $arch" }
    }
}

function Ensure-IconIco {
    $png = Join-Path $Root "crates/client-gui/icons/icon.png"
    $ico = Join-Path $Root "crates/client-gui/icons/icon.ico"

    if (Test-Path -LiteralPath $ico) {
        return
    }
    if (!(Test-Path -LiteralPath $png)) {
        throw "Missing icon source PNG at: $png"
    }

    Write-Host "Generating missing icon.ico from icon.png..." -ForegroundColor Cyan

    Add-Type -AssemblyName System.Drawing

    $img = [System.Drawing.Image]::FromFile($png)
    try {
        $size = 256
        $bmp = New-Object System.Drawing.Bitmap $size, $size
        try {
            $g = [System.Drawing.Graphics]::FromImage($bmp)
            try {
                $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
                $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
                $g.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
                $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
                $g.DrawImage($img, 0, 0, $size, $size)
            }
            finally {
                $g.Dispose()
            }

            $hicon = $bmp.GetHicon()
            try {
                $icon = [System.Drawing.Icon]::FromHandle($hicon)
                try {
                    $fs = New-Object System.IO.FileStream($ico, [System.IO.FileMode]::Create)
                    try {
                        $icon.Save($fs)
                    }
                    finally {
                        $fs.Dispose()
                    }
                }
                finally {
                    $icon.Dispose()
                }
            }
            finally {
                Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class Win32 {
  [DllImport("user32.dll", CharSet = CharSet.Auto)]
  public static extern bool DestroyIcon(IntPtr handle);
}
"@
                [Win32]::DestroyIcon($hicon) | Out-Null
            }
        }
        finally {
            $bmp.Dispose()
        }
    }
    finally {
        $img.Dispose()
    }
}

function Ensure-UiDependencies {
    $UiDir = Join-Path $Root "ui"
    if (!(Test-Path -LiteralPath (Join-Path $UiDir "package.json"))) {
        throw "GUI ui/package.json not found at: $UiDir"
    }

    if ($env:BBR_SKIP_PNPM_INSTALL -eq "1") {
        Write-Log -Message "BBR_SKIP_PNPM_INSTALL=1; skipping pnpm install." -Level "WARN"
        return
    }

    if (!(Get-Command pnpm -ErrorAction SilentlyContinue)) {
        throw "pnpm not found (needed to build the GUI frontend)."
    }

    $NodeModules = Join-Path $UiDir "node_modules"
    if (!(Test-Path -LiteralPath $NodeModules)) {
        Write-Host "Installing GUI frontend dependencies (ui/node_modules missing)..." -ForegroundColor Cyan
        Write-Log -Message "Installing UI dependencies (node_modules missing)..." -Level "INFO"
        & pnpm -C $UiDir install
        if ($LASTEXITCODE -ne 0) {
            throw "pnpm install failed (exit code: $LASTEXITCODE)"
        }
    } else {
        Write-Log -Message "ui/node_modules present; skipping pnpm install." -Level "DEBUG" -NoConsole
    }
}

try {
    $DistDir = $env:DIST_DIR
    if ([string]::IsNullOrWhiteSpace($DistDir)) {
        $DistDir = Join-Path $Root "dist"
    }
    New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
    Write-Log -Message ("Dist directory: {0}" -f $DistDir) -Level "INFO"

    $StageRoot = Join-Path $DistDir "_staging_gui"
    if (Test-Path -LiteralPath $StageRoot) {
        Write-Log -Message ("Removing previous staging dir: {0}" -f $StageRoot) -Level "INFO"
        Remove-Item -Recurse -Force -LiteralPath $StageRoot
    }

    $AppDirName = "WesoForge"
    $AppDir = Join-Path $StageRoot $AppDirName
    New-Item -ItemType Directory -Force -Path $AppDir | Out-Null
    Write-Log -Message ("Staging app dir: {0}" -f $AppDir) -Level "INFO"

    Ensure-IconIco
    Ensure-UiDependencies

    if ($env:BBR_SKIP_CARGO_BUILD -ne "1") {
        Write-Host "Building WesoForge GUI (Tauri, no bundle)..." -ForegroundColor Cyan
        Push-Location (Join-Path $Root "crates/client-gui")
        try {
            Invoke-LoggedCommand -CommandLabel "cargo tauri build --no-bundle --features prod-backend" -Command {
                cargo tauri build --no-bundle --features prod-backend
            }
        }
        finally {
            Pop-Location
        }
    } else {
        Write-Log -Message "BBR_SKIP_CARGO_BUILD=1; skipping cargo build." -Level "WARN"
    }

    $Version = Get-WorkspaceVersion
    $Arch = Get-PlatformArch
    $WebViewArch = Get-WebView2ArchFolder $Arch

    $TargetDir = $env:CARGO_TARGET_DIR
    if ([string]::IsNullOrWhiteSpace($TargetDir)) {
        $TargetDir = Join-Path $Root "target"
    }

    $ExeSrc = Join-Path $TargetDir "release\\bbr-client-gui.exe"
    if (!(Test-Path -LiteralPath $ExeSrc)) {
        throw "Expected GUI binary not found at: $ExeSrc"
    }

    $ExeDst = Join-Path $AppDir "WesoForge.exe"
    Copy-Item -Force -LiteralPath $ExeSrc -Destination $ExeDst
    Write-Log -Message ("Copied GUI exe: {0} -> {1}" -f $ExeSrc, $ExeDst) -Level "INFO"

    # MPIR runtime DLLs (required by chiavdf on Windows).
    $MpirDir = Join-Path $Root "chiavdf\\mpir_gc_x64"
    if (!(Test-Path -LiteralPath $MpirDir)) {
        throw "MPIR directory not found at: $MpirDir"
    }
    Copy-Item -Force -Path (Join-Path $MpirDir "mpir*.dll") -Destination $AppDir

    # WebView2 loader DLL (required by wry/webview2 on Windows when dynamically linked).
    $WebView2BuildRoot = Join-Path $TargetDir "release\\build"
    $WebView2Loader = Get-ChildItem -LiteralPath $WebView2BuildRoot -Recurse -Filter "WebView2Loader.dll" |
        Where-Object { $_.FullName -match "\\out\\$WebViewArch\\" } |
        Select-Object -First 1
    if ($null -eq $WebView2Loader) {
        throw "WebView2Loader.dll not found under $WebView2BuildRoot (arch folder: $WebViewArch)"
    }
    Copy-Item -Force -LiteralPath $WebView2Loader.FullName -Destination $AppDir

    $ZipPath = Join-Path $DistDir ("WesoForge-gui_Windows_{0}_{1}.zip" -f $Version, $Arch)
    if (Test-Path -LiteralPath $ZipPath) {
        Remove-Item -Force -LiteralPath $ZipPath
    }
    Compress-Archive -LiteralPath $AppDir -DestinationPath $ZipPath -Force

    Write-Log -Message ("Wrote zip: {0}" -f $ZipPath) -Level "SUCCESS"
    Write-Log -Message "=== WesoForge GUI Build Completed Successfully ===" -Level "SUCCESS"

    exit 0
}
catch {
    $err = $_
    Log-Error -Message "Build failed." -Exception $err.Exception -ErrorRecord $err
    Write-Log -Message "=== WesoForge GUI Build Failed ===" -Level "ERROR"
    exit 1
}
