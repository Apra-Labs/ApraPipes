<#
.SYNOPSIS
    Package ApraPipes SDK artifact for distribution.

.DESCRIPTION
    Creates a self-contained SDK directory with all binaries, libraries, headers,
    examples, and sample data needed to use ApraPipes.

    This script is designed to run in GitHub Actions CI but can also be run locally
    for testing. It handles all platforms: Windows, Linux x64, macOS, and ARM64/Jetson.

    SDK Structure:
        aprapipes-sdk-{platform}/
        ├── bin/           # Executables and shared libraries
        │   ├── aprapipes_cli(.exe)
        │   ├── aprapipesut(.exe)
        │   ├── aprapipes.node
        │   └── *.dll / *.so / *.dylib
        ├── lib/           # Static libraries
        │   └── *.lib / *.a
        ├── include/       # Header files
        ├── examples/
        │   ├── basic/     # JSON pipeline examples
        │   ├── cuda/      # CUDA examples (if applicable)
        │   ├── jetson/    # Jetson examples (ARM64 only)
        │   └── node/      # Node.js examples
        ├── data/          # Sample input files
        ├── README.md      # SDK documentation
        └── VERSION        # Version info

.PARAMETER SdkDir
    Output directory where SDK will be created. Will be created if it doesn't exist.

.PARAMETER BuildDir
    Path to the CMake build directory containing compiled binaries.
    - Linux: typically "build/"
    - Windows: typically "build/Release/"

.PARAMETER SourceDir
    Path to the source repository root (contains base/, examples/, data/, docs/).

.PARAMETER Platform
    Target platform: "windows", "linux", "macos", or "arm64".

.PARAMETER Cuda
    Whether this is a CUDA-enabled build. If true, includes CUDA examples.
    CUDA runtime DLLs are NOT included (they are delay-loaded).

.PARAMETER Jetson
    Include Jetson-specific examples (ARM64 only). Set to "ON" to include.

.PARAMETER VcpkgBinDir
    Optional path to vcpkg bin directory for Windows runtime DLLs.
    Required for Windows builds to include OpenCV, FFmpeg, etc.

.PARAMETER DebugOutput
    Write detailed debug information to sdk_debug.txt in SourceDir.

.EXAMPLE
    # Windows CI usage
    .\package-sdk.ps1 -SdkDir "D:\sdk" -BuildDir "D:\build\Release" `
        -SourceDir "D:\aprapipes" -Platform windows -Cuda ON `
        -VcpkgBinDir "D:\build\vcpkg_installed\x64-windows-cuda\bin"

.EXAMPLE
    # Linux x64 CI usage
    .\package-sdk.ps1 -SdkDir "/home/runner/sdk" -BuildDir "/home/runner/build" `
        -SourceDir "/home/runner/aprapipes" -Platform linux

.EXAMPLE
    # macOS CI usage
    .\package-sdk.ps1 -SdkDir "/Users/runner/sdk" -BuildDir "/Users/runner/build" `
        -SourceDir "/Users/runner/aprapipes" -Platform macos

.EXAMPLE
    # ARM64/Jetson CI usage
    .\package-sdk.ps1 -SdkDir "/data/sdk" -BuildDir "/data/build" `
        -SourceDir "/data/aprapipes" -Platform arm64 -Cuda ON -Jetson ON

.EXAMPLE
    # Local testing on Windows
    .\package-sdk.ps1 -SdkDir "C:\temp\sdk" -BuildDir "C:\ak\aprapipes\build\Release" `
        -SourceDir "C:\ak\aprapipes" -Platform windows -DebugOutput

.NOTES
    Known Issues / Design Decisions:

    1. CUDA DLLs Exclusion: CUDA runtime DLLs (cudart*, cublas*, npp*, nvjpeg*)
       are NOT included in the SDK. The CLI uses /DELAYLOAD so it can start
       without these DLLs. CUDA features work when DLLs are available at runtime.

    2. Debug DLLs Exclusion: Windows debug DLLs (*d.dll) are excluded to reduce
       SDK size. Only release builds are packaged.

    3. vcpkg DLLs: On Windows, vcpkg-installed libraries (OpenCV, FFmpeg, Boost)
       must be copied to SDK/bin for the CLI to work. The VcpkgBinDir parameter
       is required for Windows builds.

    4. VERSION file: Generated from `git describe --tags --always`. Falls back
       to "0.0.0-g<short-hash>" if no tags exist.

    Exit Codes:
        0 - Success
        1 - Invalid parameters or missing required directories
        2 - Build directory doesn't exist or is empty
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$SdkDir,

    [Parameter(Mandatory=$true)]
    [string]$BuildDir,

    [Parameter(Mandatory=$true)]
    [string]$SourceDir,

    [Parameter(Mandatory=$true)]
    [ValidateSet("windows", "linux", "macos", "arm64")]
    [string]$Platform,

    [Parameter(Mandatory=$false)]
    [string]$Cuda = "OFF",

    [Parameter(Mandatory=$false)]
    [string]$Jetson = "OFF",

    [Parameter(Mandatory=$false)]
    [string]$VcpkgBinDir = "",

    [Parameter(Mandatory=$false)]
    [switch]$DebugOutput
)

$ErrorActionPreference = "Stop"

# =============================================================================
# Validation
# =============================================================================

Write-Host "=== ApraPipes SDK Packaging ===" -ForegroundColor Cyan
Write-Host "Platform:   $Platform"
Write-Host "CUDA:       $Cuda"
Write-Host "Jetson:     $Jetson"
Write-Host "SDK Dir:    $SdkDir"
Write-Host "Build Dir:  $BuildDir"
Write-Host "Source Dir: $SourceDir"

if (-not (Test-Path $SourceDir)) {
    Write-Error "Source directory not found: $SourceDir"
    exit 1
}

if (-not (Test-Path $BuildDir)) {
    Write-Error "Build directory not found: $BuildDir"
    exit 2
}

# Derived paths
$includeDir = Join-Path $SourceDir "base/include"
$examplesDir = Join-Path $SourceDir "examples"
$dataDir = Join-Path $SourceDir "data"
$docsDir = Join-Path $SourceDir "docs"

# =============================================================================
# Create SDK Directory Structure
# =============================================================================

Write-Host ""
Write-Host "=== Creating SDK Structure ===" -ForegroundColor Cyan

$directories = @(
    "$SdkDir/bin",
    "$SdkDir/lib",
    "$SdkDir/include",
    "$SdkDir/examples/basic",
    "$SdkDir/examples/node",
    "$SdkDir/data"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Path $dir -Force | Out-Null
    Write-Host "  Created: $dir"
}

# =============================================================================
# Generate VERSION File
# =============================================================================

Write-Host ""
Write-Host "=== Generating VERSION ===" -ForegroundColor Cyan

Push-Location $SourceDir
try {
    $version = git describe --tags --always 2>$null
    if (-not $version) {
        $shortHash = git rev-parse --short HEAD 2>$null
        $version = "0.0.0-g$shortHash"
    }
} finally {
    Pop-Location
}

Set-Content -Path "$SdkDir/VERSION" -Value $version -NoNewline
Write-Host "  Version: $version"

# =============================================================================
# Copy Binaries (Platform-Specific)
# =============================================================================

Write-Host ""
Write-Host "=== Copying Binaries ===" -ForegroundColor Cyan

if ($Platform -in @("linux", "macos", "arm64")) {
    # Unix-like: executables and shared libraries in build/
    $binaries = @(
        @{ Source = "$BuildDir/aprapipes_cli"; Dest = "$SdkDir/bin/" },
        @{ Source = "$BuildDir/aprapipesut"; Dest = "$SdkDir/bin/" },
        @{ Source = "$BuildDir/aprapipes.node"; Dest = "$SdkDir/bin/" }
    )

    foreach ($item in $binaries) {
        if (Test-Path $item.Source) {
            Copy-Item $item.Source $item.Dest -Force
            Write-Host "  Copied: $(Split-Path -Leaf $item.Source)"
        }
    }

    # Copy shared libraries (platform-specific extension)
    if ($Platform -eq "macos") {
        # macOS uses .dylib
        $dylibFiles = Get-ChildItem "$BuildDir/*.dylib" -ErrorAction SilentlyContinue
        foreach ($dylib in $dylibFiles) {
            Copy-Item $dylib.FullName "$SdkDir/bin/" -Force
        }
        Write-Host "  Copied: $($dylibFiles.Count) shared libraries (.dylib)"
    } else {
        # Linux/ARM64 uses .so
        $soFiles = Get-ChildItem "$BuildDir/*.so*" -ErrorAction SilentlyContinue
        foreach ($so in $soFiles) {
            Copy-Item $so.FullName "$SdkDir/bin/" -Force
        }
        Write-Host "  Copied: $($soFiles.Count) shared libraries (.so)"
    }

    # Copy static libraries
    $aFiles = Get-ChildItem "$BuildDir/*.a" -ErrorAction SilentlyContinue
    foreach ($a in $aFiles) {
        Copy-Item $a.FullName "$SdkDir/lib/" -Force
    }
    Write-Host "  Copied: $($aFiles.Count) static libraries (.a)"

} else {
    # Windows: executables in build/Release/

    # Debug info about build directory contents
    Write-Host "  Build directory: $BuildDir"
    if (Test-Path $BuildDir) {
        $exeCount = (Get-ChildItem "$BuildDir/*.exe" -ErrorAction SilentlyContinue).Count
        $dllCount = (Get-ChildItem "$BuildDir/*.dll" -ErrorAction SilentlyContinue).Count
        Write-Host "  Found: $exeCount EXE files, $dllCount DLL files"
    } else {
        Write-Host "  WARNING: Build directory does not exist!"
    }

    # Copy executables
    $exeFiles = Get-ChildItem "$BuildDir/*.exe" -ErrorAction SilentlyContinue
    foreach ($exe in $exeFiles) {
        Copy-Item $exe.FullName "$SdkDir/bin/" -Force
        Write-Host "  Copied: $($exe.Name)"
    }

    # Copy Node.js addon
    if (Test-Path "$BuildDir/aprapipes.node") {
        Copy-Item "$BuildDir/aprapipes.node" "$SdkDir/bin/" -Force
        Write-Host "  Copied: aprapipes.node"
    }

    # Copy non-CUDA DLLs from build directory
    # CUDA DLLs are delay-loaded and not required at startup
    $cudaDllPattern = "^(cudart|cublas|cufft|cudnn|npp|nvjpeg)"
    $copiedFromBuild = 0

    Get-ChildItem "$BuildDir/*.dll" -ErrorAction SilentlyContinue | Where-Object {
        $_.Name -notmatch $cudaDllPattern
    } | ForEach-Object {
        Copy-Item $_.FullName "$SdkDir/bin/" -Force
        $copiedFromBuild++
    }
    Write-Host "  Copied: $copiedFromBuild DLLs from build (excluding CUDA)"

    # Copy vcpkg runtime DLLs (OpenCV, FFmpeg, Boost, etc.)
    if ($VcpkgBinDir -and (Test-Path $VcpkgBinDir)) {
        Write-Host ""
        Write-Host "  vcpkg bin: $VcpkgBinDir"
        $vcpkgDllCount = (Get-ChildItem "$VcpkgBinDir/*.dll" -ErrorAction SilentlyContinue).Count
        Write-Host "  Available: $vcpkgDllCount DLLs"

        # Exclude CUDA DLLs and debug DLLs (*d.dll)
        $copiedFromVcpkg = 0
        Get-ChildItem "$VcpkgBinDir/*.dll" -ErrorAction SilentlyContinue | Where-Object {
            $_.Name -notmatch $cudaDllPattern -and
            $_.Name -notmatch "d\.dll$"  # Skip debug versions
        } | ForEach-Object {
            Copy-Item $_.FullName "$SdkDir/bin/" -Force
            $copiedFromVcpkg++
        }
        Write-Host "  Copied: $copiedFromVcpkg DLLs from vcpkg (excluding CUDA/debug)"
    } elseif ($VcpkgBinDir) {
        Write-Host "  WARNING: vcpkg bin directory not found: $VcpkgBinDir"
    }

    # Copy static libraries (.lib)
    $libFiles = Get-ChildItem "$BuildDir/*.lib" -ErrorAction SilentlyContinue
    foreach ($lib in $libFiles) {
        Copy-Item $lib.FullName "$SdkDir/lib/" -Force
    }
    Write-Host "  Copied: $($libFiles.Count) static libraries (.lib)"
}

# =============================================================================
# Copy Headers
# =============================================================================

Write-Host ""
Write-Host "=== Copying Headers ===" -ForegroundColor Cyan

if (Test-Path $includeDir) {
    Copy-Item "$includeDir/*" "$SdkDir/include/" -Recurse -Force -ErrorAction SilentlyContinue
    $headerCount = (Get-ChildItem "$SdkDir/include" -Recurse -File -ErrorAction SilentlyContinue).Count
    Write-Host "  Copied: $headerCount header files"
} else {
    Write-Host "  WARNING: Include directory not found: $includeDir"
}

# =============================================================================
# Copy Examples
# =============================================================================

Write-Host ""
Write-Host "=== Copying Examples ===" -ForegroundColor Cyan

# Basic examples (JSON pipelines)
$basicExamples = Join-Path $examplesDir "basic"
if (Test-Path $basicExamples) {
    $jsonFiles = Get-ChildItem "$basicExamples/*.json" -ErrorAction SilentlyContinue
    foreach ($json in $jsonFiles) {
        Copy-Item $json.FullName "$SdkDir/examples/basic/" -Force
    }
    Write-Host "  Copied: $($jsonFiles.Count) basic examples"
} else {
    Write-Host "  WARNING: Basic examples not found: $basicExamples"
}

# Node.js examples
$nodeExamples = Join-Path $examplesDir "node"
if (Test-Path $nodeExamples) {
    $jsFiles = Get-ChildItem "$nodeExamples/*.js" -ErrorAction SilentlyContinue
    foreach ($js in $jsFiles) {
        Copy-Item $js.FullName "$SdkDir/examples/node/" -Force
    }
    if (Test-Path "$nodeExamples/README.md") {
        Copy-Item "$nodeExamples/README.md" "$SdkDir/examples/node/" -Force
    }
    Write-Host "  Copied: $($jsFiles.Count) Node.js examples"
}

# CUDA examples (only for CUDA builds)
if ($Cuda -eq "ON") {
    $cudaExamples = Join-Path $examplesDir "cuda"
    if (Test-Path $cudaExamples) {
        New-Item -ItemType Directory -Path "$SdkDir/examples/cuda" -Force | Out-Null
        $cudaJsonFiles = Get-ChildItem "$cudaExamples/*.json" -ErrorAction SilentlyContinue
        foreach ($json in $cudaJsonFiles) {
            Copy-Item $json.FullName "$SdkDir/examples/cuda/" -Force
        }
        Write-Host "  Copied: $($cudaJsonFiles.Count) CUDA examples"
    }
}

# Jetson examples (ARM64 only)
if ($Jetson -eq "ON") {
    $jetsonExamples = Join-Path $examplesDir "jetson"
    if (Test-Path $jetsonExamples) {
        New-Item -ItemType Directory -Path "$SdkDir/examples/jetson" -Force | Out-Null
        $jetsonJsonFiles = Get-ChildItem "$jetsonExamples/*.json" -ErrorAction SilentlyContinue
        foreach ($json in $jetsonJsonFiles) {
            Copy-Item $json.FullName "$SdkDir/examples/jetson/" -Force
        }
        Write-Host "  Copied: $($jetsonJsonFiles.Count) Jetson examples"
    }
}

# =============================================================================
# Copy Sample Data
# =============================================================================

Write-Host ""
Write-Host "=== Copying Sample Data ===" -ForegroundColor Cyan

$dataFiles = @("frame.jpg", "faces.jpg")
$copiedData = 0

foreach ($file in $dataFiles) {
    $sourcePath = Join-Path $dataDir $file
    if (Test-Path $sourcePath) {
        Copy-Item $sourcePath "$SdkDir/data/" -Force
        Write-Host "  Copied: $file"
        $copiedData++
    }
}
Write-Host "  Total: $copiedData data files"

# =============================================================================
# Copy Documentation
# =============================================================================

Write-Host ""
Write-Host "=== Copying Documentation ===" -ForegroundColor Cyan

$sdkReadme = Join-Path $docsDir "SDK_README.md"
if (Test-Path $sdkReadme) {
    Copy-Item $sdkReadme "$SdkDir/README.md" -Force
    Write-Host "  Copied: SDK_README.md -> README.md"
} else {
    Write-Host "  WARNING: SDK_README.md not found at: $sdkReadme"
}

# =============================================================================
# Summary
# =============================================================================

Write-Host ""
Write-Host "=== SDK Contents ===" -ForegroundColor Green

$allFiles = Get-ChildItem $SdkDir -Recurse -File
foreach ($file in $allFiles) {
    $relativePath = $file.FullName.Replace("$SdkDir/", "").Replace("$SdkDir\", "")
    Write-Host "  $relativePath"
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Green
$binCount = (Get-ChildItem "$SdkDir/bin" -File -ErrorAction SilentlyContinue).Count
$libCount = (Get-ChildItem "$SdkDir/lib" -File -ErrorAction SilentlyContinue).Count
$exampleCount = (Get-ChildItem "$SdkDir/examples" -Recurse -File -ErrorAction SilentlyContinue).Count

Write-Host "  Binaries:  $binCount files"
Write-Host "  Libraries: $libCount files"
Write-Host "  Examples:  $exampleCount files"
Write-Host "  Total:     $($allFiles.Count) files"

# =============================================================================
# Debug Output (Optional)
# =============================================================================

if ($DebugOutput) {
    $debugFile = Join-Path $SourceDir "sdk_debug.txt"

    @"
SDK Debug Info
==============
Generated: $(Get-Date -Format "o")
Platform: $Platform
CUDA: $Cuda
Jetson: $Jetson
SDK Directory: $SdkDir
Build Directory: $BuildDir

Shared libraries in SDK bin:
"@ | Out-File $debugFile

    # List all shared libraries based on platform
    Get-ChildItem "$SdkDir/bin/*.dll", "$SdkDir/bin/*.so*", "$SdkDir/bin/*.dylib" -ErrorAction SilentlyContinue | ForEach-Object {
        "  $($_.Name)" | Out-File $debugFile -Append
    }

    $libCount = (Get-ChildItem "$SdkDir/bin/*.dll", "$SdkDir/bin/*.so*", "$SdkDir/bin/*.dylib" -ErrorAction SilentlyContinue).Count
    "Total shared libraries: $libCount" | Out-File $debugFile -Append

    Write-Host ""
    Write-Host "  Debug info written to: $debugFile"
}

Write-Host ""
Write-Host "SDK packaging complete!" -ForegroundColor Green
exit 0
