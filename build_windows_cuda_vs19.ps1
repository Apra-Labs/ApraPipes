# ApraPipes Build Script for Windows with CUDA and Visual Studio 2019
# Generated from successful build on 2025-10-06
# Requirements: Visual Studio 2019, CUDA 11.8 (or compatible), Git Bash or WSL

param(
    [switch]$Clean = $false,
    [switch]$SkipTests = $false,
    [string]$BuildType = "RelWithDebInfo"
)

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

Write-Host "=== ApraPipes Build Script for Windows + CUDA + VS 2019 ===" -ForegroundColor Cyan
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
    exit 1
}

# Step 2: Verify CUDA installation
Write-Host "[2/10] Verifying CUDA installation..." -ForegroundColor Yellow
$cudaBasePath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"

if (-not (Test-Path $cudaBasePath)) {
    Write-Host "   ERROR: CUDA toolkit not found at $cudaBasePath" -ForegroundColor Red
    exit 1
}

$cudaVersions = Get-ChildItem $cudaBasePath -Directory | Select-Object -ExpandProperty Name
Write-Host "   Found CUDA versions: $($cudaVersions -join ', ')" -ForegroundColor Green

# Check for CUDA 11.8 specifically (recommended)
if ($cudaVersions -contains "v11.8") {
    Write-Host "   Using CUDA 11.8 (recommended)" -ForegroundColor Green
    $env:CUDA_PATH = "$cudaBasePath\v11.8"
} else {
    Write-Host "   WARNING: CUDA 11.8 not found. Using available version may cause compatibility issues." -ForegroundColor Yellow
}

# Step 3: Verify CMake installation
Write-Host "[3/10] Verifying CMake installation..." -ForegroundColor Yellow
if (-not (Test-Command "cmake")) {
    Write-Host "   ERROR: CMake not found in PATH" -ForegroundColor Red
    Write-Host "   CMake will be downloaded by vcpkg during the build process" -ForegroundColor Yellow
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
            } catch {
                Write-Host "   Warning: Some files in $dir could not be removed (possibly locked by Visual Studio)" -ForegroundColor Yellow
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
    exit 1
}

Push-Location $vcpkgDir
try {
    if (Test-Path $bootstrapScript) {
        Write-Host "   Running bootstrap-vcpkg.bat..." -ForegroundColor Gray
        & cmd /c $bootstrapScript
        if ($LASTEXITCODE -ne 0) {
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
    & .\vcpkg.exe integrate install
    if ($LASTEXITCODE -ne 0) {
        throw "vcpkg integrate failed with exit code $LASTEXITCODE"
    }
    Write-Host "   vcpkg integration completed" -ForegroundColor Green
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
    Write-Host "   This may take a while as vcpkg installs ~140 dependencies..." -ForegroundColor Gray

    $cmakeArgs = @(
        "-G", "Visual Studio 16 2019",
        "-A", "x64",
        "-DCMAKE_BUILD_TYPE=$BuildType",
        "-DENABLE_CUDA=ON",
        "-DENABLE_WINDOWS=ON",
        "-DENABLE_LINUX=OFF",
        "-DCMAKE_TOOLCHAIN_FILE=$toolchainFile",
        $baseDir
    )

    & cmake @cmakeArgs

    if ($LASTEXITCODE -ne 0) {
        throw "CMake configuration failed with exit code $LASTEXITCODE"
    }

    Write-Host "   CMake configuration completed successfully" -ForegroundColor Green
} finally {
    Pop-Location
}

# Step 9: Build the project
Write-Host "[9/10] Building the project..." -ForegroundColor Yellow
Push-Location $buildDir
try {
    Write-Host "   Building with configuration: $BuildType" -ForegroundColor Gray
    Write-Host "   This may take several minutes..." -ForegroundColor Gray

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
            $testList = & .\aprapipesut.exe --list_content 2>&1 | Select-Object -First 10
            Write-Host "   Sample test suites: $($testList[0..2] -join ', ')..." -ForegroundColor Green
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
Write-Host "  - Visual Studio: 2019 $vs2019Edition" -ForegroundColor White
Write-Host "  - CUDA: $(if ($env:CUDA_PATH) { Split-Path $env:CUDA_PATH -Leaf } else { 'Auto-detected' })" -ForegroundColor White
Write-Host "  - Build Type: $BuildType" -ForegroundColor White
Write-Host "  - Platform Toolset: v142" -ForegroundColor White
Write-Host ""
Write-Host "Output Files:" -ForegroundColor Cyan
Write-Host "  - Executable: $exePath" -ForegroundColor White
Write-Host "  - Libraries: $(Join-Path $buildDir "$BuildType\*.lib")" -ForegroundColor White
Write-Host ""
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "  - Run tests: cd _build\$BuildType && .\aprapipesut.exe" -ForegroundColor White
Write-Host "  - Run specific test: .\aprapipesut.exe --run_test=<test_name>" -ForegroundColor White
Write-Host "  - List all tests: .\aprapipesut.exe --list_content" -ForegroundColor White
Write-Host ""
