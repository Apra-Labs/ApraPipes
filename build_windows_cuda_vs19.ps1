# ApraPipes Component-Based Build Script for Windows with CUDA and Visual Studio 2019
# Updated: 2025-10-09 - Phase 5.5 Complete
# Requirements: Visual Studio 2019, CUDA 11.8 (or compatible), vcpkg

param(
    [switch]$Clean = $false,
    [switch]$SkipTests = $false,
    [string]$BuildType = "RelWithDebInfo",
    [string]$Preset = "",
    [string]$Components = "",
    [switch]$Help = $false
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Help function
function Show-Help {
    Write-Host ""
    Write-Host "=== ApraPipes Component-Based Build Script for VS 2019 ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "USAGE:" -ForegroundColor Yellow
    Write-Host "  .\build_windows_cuda_vs19.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "OPTIONS:" -ForegroundColor Yellow
    Write-Host "  -Help                Display this help message"
    Write-Host "  -Clean               Remove existing build directories before building"
    Write-Host "  -SkipTests           Skip running test executable verification"
    Write-Host "  -BuildType <type>    Build configuration (default: RelWithDebInfo)"
    Write-Host "                       Options: Debug, Release, RelWithDebInfo, MinSizeRel"
    Write-Host "  -Preset <name>       Use predefined component preset"
    Write-Host "  -Components <list>   Semicolon-separated component list"
    Write-Host ""
    Write-Host "PRESETS:" -ForegroundColor Yellow
    Write-Host "  minimal              CORE only (~10-15 min build time)"
    Write-Host "                       - Pipeline infrastructure, basic I/O"
    Write-Host ""
    Write-Host "  video                CORE + VIDEO + IMAGE_PROCESSING (~25-30 min)"
    Write-Host "                       - Mp4, H264, RTSP streaming"
    Write-Host "                       - OpenCV CPU image processing"
    Write-Host ""
    Write-Host "  cuda                 CORE + VIDEO + IMAGE_PROCESSING + CUDA_COMPONENT (~15-20 min)"
    Write-Host "                       - GPU-accelerated processing (NVJPEG, NPP, NVCODEC)"
    Write-Host "                       - Fastest build when vcpkg cache exists"
    Write-Host ""
    Write-Host "  full                 ALL components (~60-90 min)"
    Write-Host "                       - Complete build (backward compatible)"
    Write-Host "                       - Includes AUDIO, QR, WEBCAM, FACE_DETECTION, etc."
    Write-Host ""
    Write-Host "AVAILABLE COMPONENTS:" -ForegroundColor Yellow
    Write-Host "  CORE                 Pipeline infrastructure (always required)"
    Write-Host "  VIDEO                Mp4, H264, RTSP codecs and streaming"
    Write-Host "  IMAGE_PROCESSING     OpenCV CPU-based image processing"
    Write-Host "  CUDA_COMPONENT       GPU acceleration (NPP, NVJPEG, NVCODEC)"
    Write-Host "  WEBCAM               Webcam capture via OpenCV"
    Write-Host "  QR                   QR code reading"
    Write-Host "  AUDIO                Audio capture and Whisper transcription"
    Write-Host "  FACE_DETECTION       Face detection and landmarks"
    Write-Host "  THUMBNAIL            Thumbnail generation"
    Write-Host "  IMAGE_VIEWER         Image viewing GUI"
    Write-Host ""
    Write-Host "EXAMPLES:" -ForegroundColor Yellow
    Write-Host "  # Minimal build (fastest)"
    Write-Host "  .\build_windows_cuda_vs19.ps1 -Preset minimal"
    Write-Host ""
    Write-Host "  # Video processing (most common)"
    Write-Host "  .\build_windows_cuda_vs19.ps1 -Preset video"
    Write-Host ""
    Write-Host "  # GPU-accelerated build"
    Write-Host "  .\build_windows_cuda_vs19.ps1 -Preset cuda"
    Write-Host ""
    Write-Host "  # Full build (all components)"
    Write-Host "  .\build_windows_cuda_vs19.ps1 -Preset full"
    Write-Host ""
    Write-Host "  # Clean build with minimal preset"
    Write-Host "  .\build_windows_cuda_vs19.ps1 -Clean -Preset minimal"
    Write-Host ""
    Write-Host "  # Custom component selection"
    Write-Host "  .\build_windows_cuda_vs19.ps1 -Components 'CORE;VIDEO;WEBCAM'"
    Write-Host ""
    Write-Host "  # Debug build with video preset"
    Write-Host "  .\build_windows_cuda_vs19.ps1 -Preset video -BuildType Debug"
    Write-Host ""
    Write-Host "BUILD TIME ESTIMATES (tested on Phase 5.5):" -ForegroundColor Yellow
    Write-Host "  Minimal:  10-15 minutes"
    Write-Host "  Video:    25-30 minutes"
    Write-Host "  CUDA:     15-20 minutes (with cache) or 60+ min (first time)"
    Write-Host "  Full:     60-90 minutes"
    Write-Host ""
    Write-Host "For more information, see COMPONENTS_GUIDE.md" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}

# Show help if requested
if ($Help) {
    Show-Help
}

# Parse preset into components
function Get-ComponentsFromPreset {
    param([string]$PresetName)

    switch ($PresetName.ToLower()) {
        "minimal" {
            return "CORE"
        }
        "video" {
            return "CORE;VIDEO;IMAGE_PROCESSING"
        }
        "cuda" {
            return "CORE;VIDEO;IMAGE_PROCESSING;CUDA_COMPONENT"
        }
        "full" {
            return "ALL"
        }
        "" {
            # Default to ALL for backward compatibility
            return "ALL"
        }
        default {
            Write-Host "ERROR: Unknown preset '$PresetName'" -ForegroundColor Red
            Write-Host "Available presets: minimal, video, cuda, full" -ForegroundColor Yellow
            Write-Host "Use -Help for more information" -ForegroundColor Yellow
            exit 1
        }
    }
}

# Determine component list
$componentList = ""
if ($Preset -ne "") {
    $componentList = Get-ComponentsFromPreset -PresetName $Preset
    Write-Host "Using preset: $Preset" -ForegroundColor Cyan
} elseif ($Components -ne "") {
    $componentList = $Components
    Write-Host "Using custom components: $Components" -ForegroundColor Cyan
} else {
    # Default to ALL
    $componentList = "ALL"
    Write-Host "No preset or components specified - building ALL components (default)" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "=== ApraPipes Build Script for Windows + CUDA + VS 2019 ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "BUILD CONFIGURATION:" -ForegroundColor Yellow
Write-Host "  Components: $componentList"
Write-Host "  Build Type: $BuildType"
Write-Host "  Clean Build: $Clean"
Write-Host ""

# Function to check if a command exists
function Test-Command {
    param($Command)
    $null -ne (Get-Command $Command -ErrorAction SilentlyContinue)
}

# Step 1: Verify Visual Studio 2019 installation
Write-Host "[1/10] Verifying Visual Studio 2019 installation..." -ForegroundColor Yellow
$vs2019Paths = @(
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional",
    "C:\Program Files (x86)\Microsoft Visual Studio\2019\Enterprise"
)

$vs2019Found = $false
$vs2019Edition = ""
foreach ($path in $vs2019Paths) {
    if (Test-Path $path) {
        $vs2019Found = $true
        $vs2019Edition = Split-Path $path -Leaf
        Write-Host "   Found: Visual Studio 2019 $vs2019Edition" -ForegroundColor Green
        break
    }
}

if (-not $vs2019Found) {
    Write-Host "   ERROR: Visual Studio 2019 not found!" -ForegroundColor Red
    Write-Host "   Please install Visual Studio 2019 Community, Professional, or Enterprise" -ForegroundColor Red
    Write-Host "   CUDA 11.8 is compatible with VS 2019 but NOT with VS 2022 v17.4+" -ForegroundColor Yellow
    exit 1
}

# Step 2: Verify CUDA installation
Write-Host "[2/10] Verifying CUDA installation..." -ForegroundColor Yellow
$cudaBasePath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

if (-not (Test-Path $cudaBasePath)) {
    Write-Host "   ERROR: CUDA toolkit not found at $cudaBasePath" -ForegroundColor Red
    Write-Host "   Required for CUDA-enabled components" -ForegroundColor Red
    if ($componentList -match "CUDA") {
        Write-Host "   Cannot build CUDA components without CUDA toolkit" -ForegroundColor Red
        exit 1
    }
}

$cudaVersions = Get-ChildItem $cudaBasePath -Directory | Select-Object -ExpandProperty Name
Write-Host "   Found CUDA versions: $($cudaVersions -join ', ')" -ForegroundColor Green

# Check for CUDA 11.8 specifically (recommended for VS 2019)
if ($cudaVersions -contains "v11.8") {
    Write-Host "   Using CUDA 11.8 (recommended for VS 2019)" -ForegroundColor Green
    $env:CUDA_PATH = "$cudaBasePath\v11.8"
} else {
    Write-Host "   WARNING: CUDA 11.8 not found. Using available version may cause compatibility issues." -ForegroundColor Yellow
}

# Step 3: Verify CMake installation
Write-Host "[3/10] Verifying CMake installation..." -ForegroundColor Yellow
if (-not (Test-Command "cmake")) {
    Write-Host "   WARNING: CMake not found in PATH" -ForegroundColor Yellow
    Write-Host "   CMake will be downloaded by vcpkg during the build process" -ForegroundColor Yellow
} else {
    $cmakeVersion = (cmake --version | Select-Object -First 1) -replace 'cmake version ', ''
    Write-Host "   Found CMake: $cmakeVersion" -ForegroundColor Green
}

# Step 4: Clean existing build directories (if requested)
if ($Clean) {
    Write-Host "[4/10] Cleaning existing build directories..." -ForegroundColor Yellow

    $dirsToClean = @("_build", "_debugbuild")
    foreach ($dir in $dirsToClean) {
        $fullPath = Join-Path $scriptDir $dir
        if (Test-Path $fullPath) {
            Write-Host "   Removing $dir..." -ForegroundColor Gray
            try {
                Remove-Item -Recurse -Force $fullPath -ErrorAction SilentlyContinue
                # Wait a bit for file locks to release
                Start-Sleep -Seconds 2
                Write-Host "   Removed $dir" -ForegroundColor Green
            } catch {
                Write-Host "   Warning: Some files in $dir could not be removed (possibly locked)" -ForegroundColor Yellow
            }
        }
    }
} else {
    Write-Host "[4/10] Skipping clean (use -Clean to remove existing builds)..." -ForegroundColor Yellow
}

# Step 5: Modify CMakeLists.txt to use VS 2019 toolset (v142)
Write-Host "[5/10] Configuring CMakeLists.txt for VS 2019..." -ForegroundColor Yellow
$cmakeListsPath = Join-Path $scriptDir "base\CMakeLists.txt"

if (Test-Path $cmakeListsPath) {
    $content = Get-Content $cmakeListsPath -Raw

    # Replace v143 (VS 2022) with v142 (VS 2019)
    if ($content -match 'set\(VCPKG_PLATFORM_TOOLSET "v143"') {
        $content = $content -replace 'set\(VCPKG_PLATFORM_TOOLSET "v143" CACHE STRING "v143" FORCE\)', 'set(VCPKG_PLATFORM_TOOLSET "v142" CACHE STRING "v142" FORCE)'
        Set-Content -Path $cmakeListsPath -Value $content -NoNewline
        Write-Host "   Updated VCPKG_PLATFORM_TOOLSET to v142" -ForegroundColor Green
    } elseif ($content -match 'set\(VCPKG_PLATFORM_TOOLSET "v142"') {
        Write-Host "   Already configured for v142 (VS 2019)" -ForegroundColor Green
    } else {
        Write-Host "   Warning: Could not find VCPKG_PLATFORM_TOOLSET setting" -ForegroundColor Yellow
    }
} else {
    Write-Host "   ERROR: CMakeLists.txt not found at $cmakeListsPath" -ForegroundColor Red
    exit 1
}

# Step 6: Bootstrap vcpkg
Write-Host "[6/10] Bootstrapping vcpkg..." -ForegroundColor Yellow
$vcpkgDir = Join-Path $scriptDir "vcpkg"
$vcpkgExe = Join-Path $vcpkgDir "vcpkg.exe"
$bootstrapScript = Join-Path $vcpkgDir "bootstrap-vcpkg.bat"

if (-not (Test-Path $vcpkgDir)) {
    Write-Host "   ERROR: vcpkg directory not found at $vcpkgDir" -ForegroundColor Red
    Write-Host "   Ensure you have cloned the repository with submodules:" -ForegroundColor Yellow
    Write-Host "   git clone --recursive https://github.com/Apra-Labs/ApraPipes.git" -ForegroundColor Yellow
    exit 1
}

Push-Location $vcpkgDir
try {
    if (Test-Path $bootstrapScript) {
        Write-Host "   Running bootstrap-vcpkg.bat..." -ForegroundColor Gray
        & cmd /c $bootstrapScript 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0 -and -not (Test-Path $vcpkgExe)) {
            throw "vcpkg bootstrap failed with exit code $LASTEXITCODE"
        }
    } else {
        Write-Host "   ERROR: bootstrap-vcpkg.bat not found" -ForegroundColor Red
        exit 1
    }
} finally {
    Pop-Location
}

if (-not (Test-Path $vcpkgExe)) {
    Write-Host "   ERROR: vcpkg.exe was not created after bootstrap" -ForegroundColor Red
    exit 1
}

Write-Host "   vcpkg bootstrapped successfully" -ForegroundColor Green

# Step 7: Integrate vcpkg with Visual Studio
Write-Host "[7/10] Integrating vcpkg with Visual Studio..." -ForegroundColor Yellow
Push-Location $vcpkgDir
try {
    & .\vcpkg.exe integrate install 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "   Warning: vcpkg integrate returned exit code $LASTEXITCODE" -ForegroundColor Yellow
    } else {
        Write-Host "   vcpkg integration completed" -ForegroundColor Green
    }
} finally {
    Pop-Location
}

# Step 8: Configure CMake with Visual Studio 2019
Write-Host "[8/10] Configuring CMake with Visual Studio 2019..." -ForegroundColor Yellow
$buildDir = Join-Path $scriptDir "_build"
$baseDir = Join-Path $scriptDir "base"
$toolchainFile = Join-Path $vcpkgDir "scripts\buildsystems\vcpkg.cmake"

# Create build directory
if (-not (Test-Path $buildDir)) {
    New-Item -ItemType Directory -Path $buildDir | Out-Null
}

Push-Location $buildDir
try {
    Write-Host "   Running CMake configuration..." -ForegroundColor Gray

    # Estimate dependencies based on components
    $estimatedPackages = 42  # Base
    if ($componentList -match "VIDEO") { $estimatedPackages = 48 }
    if ($componentList -match "CUDA") { $estimatedPackages = 117 }
    if ($componentList -match "ALL") { $estimatedPackages = 120 }

    Write-Host "   This may take a while as vcpkg installs ~$estimatedPackages dependencies..." -ForegroundColor Gray

    $cmakeArgs = @(
        "-G", "Visual Studio 16 2019",
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DENABLE_CUDA=ON",
        "-DENABLE_WINDOWS=ON",
        "-DENABLE_LINUX=OFF",
        "-DENABLE_COMPONENTS=$componentList",
        "-DCMAKE_TOOLCHAIN_FILE=$toolchainFile",
        $baseDir
    )

    & cmake @cmakeArgs

    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed with exit code $LASTEXITCODE"
    }

    Write-Host "   CMake configuration completed successfully" -ForegroundColor Green

    # Show component configuration result
    $buildLog = Get-Content "CMakeCache.txt" | Select-String "APRAPIPES_ENABLE_"
    if ($buildLog) {
        Write-Host "   Components enabled:" -ForegroundColor Gray
        $buildLog | ForEach-Object {
            if ($_ -match "APRAPIPES_ENABLE_(\w+):BOOL=ON") {
                Write-Host "     - $($matches[1])" -ForegroundColor Green
            }
        }
    }
} finally {
    Pop-Location
}

# Step 9: Build the project
Write-Host "[9/10] Building the project..." -ForegroundColor Yellow
Push-Location $buildDir
try {
    Write-Host "   Building with configuration: $BuildType" -ForegroundColor Gray

    # Estimate build time
    $estimatedTime = "5-10 minutes"
    if ($componentList -eq "CORE") { $estimatedTime = "10-15 minutes" }
    if ($componentList -match "VIDEO") { $estimatedTime = "25-30 minutes" }
    if ($componentList -match "CUDA") { $estimatedTime = "15-20 minutes (or 60+ min first time)" }
    if ($componentList -eq "ALL") { $estimatedTime = "60-90 minutes" }

    Write-Host "   Estimated build time: $estimatedTime" -ForegroundColor Gray

    & cmake --build . --config $BuildType

    if ($LASTEXITCODE -ne 0) {
        throw "Build failed with exit code $LASTEXITCODE"
    }

    Write-Host "   Build completed successfully" -ForegroundColor Green
} finally {
    Pop-Location
}

# Step 10: Verify the executable
Write-Host "[10/10] Verifying aprapipesut.exe..." -ForegroundColor Yellow
$exePath = Join-Path $buildDir "$BuildType\aprapipesut.exe"

if (-not (Test-Path $exePath)) {
    Write-Host "   ERROR: aprapipesut.exe not found at $exePath" -ForegroundColor Red
    exit 1
}

$exeSize = (Get-Item $exePath).Length
$exeSizeMB = [math]::Round($exeSize / 1MB, 2)
Write-Host "   Found aprapipesut.exe ($exeSizeMB MB)" -ForegroundColor Green

# Check library too
$libPath = Join-Path $buildDir "$BuildType\aprapipes.lib"
if (Test-Path $libPath) {
    $libSize = (Get-Item $libPath).Length
    $libSizeMB = [math]::Round($libSize / 1MB, 2)
    Write-Host "   Found aprapipes.lib ($libSizeMB MB)" -ForegroundColor Green
}

# Test the executable
if (-not $SkipTests) {
    Write-Host "   Testing executable..." -ForegroundColor Gray
    Push-Location (Split-Path $exePath)
    try {
        $helpOutput = & .\aprapipesut.exe --help 2>&1 | Select-Object -First 5
        if ($LASTEXITCODE -eq 0 -or $helpOutput -match "Boost.Test") {
            Write-Host "   Executable runs successfully!" -ForegroundColor Green

            # List test suites
            Write-Host "   Listing test suites..." -ForegroundColor Gray
            $testList = & .\aprapipesut.exe --list_content 2>&1
            $testCount = ($testList | Measure-Object -Line).Lines
            $suiteSample = ($testList | Select-Object -First 3) -join ', '
            Write-Host "   Found $testCount test entries" -ForegroundColor Green
            Write-Host "   Sample: $suiteSample..." -ForegroundColor Gray

            # Show component-specific tests if CUDA enabled
            if ($componentList -match "CUDA") {
                $cudaTests = $testList | Select-String "nppi|nvjpeg|nvcodec|cuda" | Select-Object -First 3
                if ($cudaTests) {
                    Write-Host "   CUDA tests available: $($cudaTests -join ', ')..." -ForegroundColor Green
                }
            }
        } else {
            Write-Host "   Warning: Executable may have issues" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "   Warning: Could not test executable: $_" -ForegroundColor Yellow
    } finally {
        Pop-Location
    }
} else {
    Write-Host "   Skipping executable tests (use without -SkipTests to run)" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "=== BUILD COMPLETED SUCCESSFULLY ===" -ForegroundColor Green
Write-Host ""
Write-Host "Build Configuration:" -ForegroundColor Cyan
Write-Host "  - Components: $componentList" -ForegroundColor White
Write-Host "  - Visual Studio: 2019 $vs2019Edition" -ForegroundColor White
Write-Host "  - CUDA: $(if ($env:CUDA_PATH) { Split-Path $env:CUDA_PATH -Leaf } else { 'Auto-detected' })" -ForegroundColor White
Write-Host "  - Build Type: $BuildType" -ForegroundColor White
Write-Host "  - Platform Toolset: v142" -ForegroundColor White
Write-Host ""
Write-Host "Output Files:" -ForegroundColor Cyan
Write-Host "  - Executable: $exePath" -ForegroundColor White
Write-Host "  - Library: $(Join-Path $buildDir "$BuildType\aprapipes.lib")" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  - Run all tests: cd _build\$BuildType && .\aprapipesut.exe -p" -ForegroundColor White
Write-Host "  - Run specific test: .\aprapipesut.exe --run_test=<test_name>" -ForegroundColor White
Write-Host "  - List all tests: .\aprapipesut.exe --list_content" -ForegroundColor White
Write-Host ""
Write-Host "Component Information:" -ForegroundColor Cyan
Write-Host "  - See COMPONENTS_GUIDE.md for detailed component documentation" -ForegroundColor White
Write-Host "  - See TESTING_PHASE5.5_REPORT.md for test results and validation" -ForegroundColor White
Write-Host ""
Write-Host "Build different configurations:" -ForegroundColor Cyan
Write-Host "  - Minimal build:  .\build_windows_cuda_vs19.ps1 -Preset minimal" -ForegroundColor White
Write-Host "  - Video build:    .\build_windows_cuda_vs19.ps1 -Preset video" -ForegroundColor White
Write-Host "  - CUDA build:     .\build_windows_cuda_vs19.ps1 -Preset cuda" -ForegroundColor White
Write-Host "  - Full build:     .\build_windows_cuda_vs19.ps1 -Preset full" -ForegroundColor White
Write-Host ""
