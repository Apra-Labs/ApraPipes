# ApraPipes Samples Build Script
# This script builds samples as a STANDALONE project that links against the already-built aprapipes library
#
# Prerequisites:
#   1. Main library must be built first (run build_windows_cuda_vs19.ps1 from root)
#   2. vcpkg must be set up (happens automatically when building main library)

param(
    [string]$BuildType = "RelWithDebInfo",
    [switch]$Clean = $false
)

$ErrorActionPreference = "Stop"
$samplesDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $samplesDir
$buildDir = Join-Path $samplesDir "_build"
$vcpkgToolchain = Join-Path $rootDir "vcpkg\scripts\buildsystems\vcpkg.cmake"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  ApraPipes Samples - Standalone Build  " -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# ============================================================================
# Step 1: Verify Prerequisites
# ============================================================================

Write-Host "[1/5] Verifying prerequisites..." -ForegroundColor Yellow

# Check if main library exists (aprapipesd.lib for Debug, aprapipes.lib for others)
$libName = if ($BuildType -eq "Debug") { "aprapipesd.lib" } else { "aprapipes.lib" }
$libPath = Join-Path $rootDir "_build\$BuildType\$libName"

if (-not (Test-Path $libPath)) {
    Write-Host ""
    Write-Host "ERROR: Main aprapipes library not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected location: $libPath" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please build the main library first:" -ForegroundColor Yellow
    Write-Host "  cd $rootDir" -ForegroundColor White
    Write-Host "  .\build_windows_cuda_vs19.ps1" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host "   Found aprapipes library: $libPath" -ForegroundColor Green

# Check if vcpkg toolchain exists
if (-not (Test-Path $vcpkgToolchain)) {
    Write-Host ""
    Write-Host "ERROR: vcpkg toolchain not found!" -ForegroundColor Red
    Write-Host "Please run the main build script first to set up vcpkg" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "   Found vcpkg toolchain: $vcpkgToolchain" -ForegroundColor Green

# Check if hello_pipeline source exists
$helloSource = Join-Path $samplesDir "basic\hello_pipeline\main.cpp"
if (-not (Test-Path $helloSource)) {
    Write-Host ""
    Write-Host "ERROR: hello_pipeline source not found!" -ForegroundColor Red
    Write-Host "Expected: $helloSource" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

Write-Host "   Found hello_pipeline source" -ForegroundColor Green

# ============================================================================
# Step 2: Clean (if requested)
# ============================================================================

Write-Host "[2/5] Build directory setup..." -ForegroundColor Yellow

if ($Clean -and (Test-Path $buildDir)) {
    Write-Host "   Cleaning existing build directory..." -ForegroundColor Gray
    Remove-Item -Recurse -Force $buildDir -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# Create build directory
if (-not (Test-Path $buildDir)) {
    Write-Host "   Creating build directory..." -ForegroundColor Gray
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Write-Host "   Build directory: $buildDir" -ForegroundColor Green

# ============================================================================
# Step 3: Configure CMake
# ============================================================================

Write-Host "[3/5] Configuring CMake..." -ForegroundColor Yellow

Push-Location $buildDir
try {
    Write-Host "   Running CMake configuration..." -ForegroundColor Gray
    Write-Host "   This finds the aprapipes library and sets up the build..." -ForegroundColor Gray
    Write-Host ""

    # Point vcpkg to the main build's installed packages
    $vcpkgInstalledDir = Join-Path $rootDir "_build\vcpkg_installed\x64-windows"

    $cmakeArgs = @(
        "-G", "Visual Studio 16 2019",
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DCMAKE_TOOLCHAIN_FILE=$vcpkgToolchain",
        "-DVCPKG_INSTALLED_DIR=$vcpkgInstalledDir",
        ".."
    )

    & cmake @cmakeArgs

    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "   CMake configuration completed successfully" -ForegroundColor Green

} catch {
    Write-Host ""
    Write-Host "CMake configuration failed: $_" -ForegroundColor Red
    Pop-Location
    exit 1
}

# ============================================================================
# Step 4: Build Samples
# ============================================================================

Write-Host "[4/5] Building samples..." -ForegroundColor Yellow

try {
    Write-Host "   Compiling with configuration: $BuildType" -ForegroundColor Gray
    Write-Host ""

    & cmake --build . --config $BuildType

    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "   Build completed successfully" -ForegroundColor Green

} catch {
    Write-Host ""
    Write-Host "Build failed: $_" -ForegroundColor Red
    Pop-Location
    exit 1
} finally {
    Pop-Location
}

# ============================================================================
# Step 5: Copy Runtime DLLs
# ============================================================================

Write-Host "[5/6] Copying runtime DLLs..." -ForegroundColor Yellow

$mainBuildBin = Join-Path $rootDir "_build\$BuildType"
$sampleBin = Join-Path $buildDir $BuildType

# Copy Boost DLLs (required at runtime)
$boostDlls = Get-ChildItem -Path $mainBuildBin -Filter "boost_*.dll"
foreach ($dll in $boostDlls) {
    Copy-Item -Path $dll.FullName -Destination $sampleBin -Force
}

Write-Host "   Copied $($boostDlls.Count) Boost DLLs" -ForegroundColor Green

# ============================================================================
# Step 6: Verify Output
# ============================================================================

Write-Host "[6/6] Verifying output..." -ForegroundColor Yellow

$exePath = Join-Path $buildDir "$BuildType\hello_pipeline.exe"

if (-not (Test-Path $exePath)) {
    Write-Host ""
    Write-Host "ERROR: Sample executable not found at expected location!" -ForegroundColor Red
    Write-Host "Expected: $exePath" -ForegroundColor Yellow
    Write-Host ""
    exit 1
}

$exeSize = (Get-Item $exePath).Length
$exeSizeMB = [math]::Round($exeSize / 1MB, 2)
Write-Host "   Found hello_pipeline.exe ($exeSizeMB MB)" -ForegroundColor Green

# ============================================================================
# Summary
# ============================================================================

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "  Samples Build Completed Successfully  " -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Sample executable:" -ForegroundColor Cyan
Write-Host "  $exePath" -ForegroundColor White
Write-Host ""
Write-Host "To run the sample:" -ForegroundColor Cyan
Write-Host "  cd $buildDir\$BuildType" -ForegroundColor White
Write-Host "  .\hello_pipeline.exe" -ForegroundColor White
Write-Host ""
Write-Host "Or directly:" -ForegroundColor Cyan
Write-Host "  $exePath" -ForegroundColor White
Write-Host ""
