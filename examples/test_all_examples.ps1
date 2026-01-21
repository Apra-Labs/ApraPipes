<#
.SYNOPSIS
    Run ApraPipes SDK integration tests on Windows.

.DESCRIPTION
    Tests the ApraPipes CLI with JSON pipeline examples to verify SDK functionality.
    Generates a JSON report with pass/fail results.

    This is the Windows equivalent of test_all_examples.sh for Linux/macOS.

.PARAMETER SdkDir
    Path to the SDK directory containing bin/, examples/, data/.

.PARAMETER JsonReport
    Path where the JSON test report will be written.

.PARAMETER Basic
    Run only basic (CPU) examples.

.PARAMETER Cuda
    Run only CUDA (GPU) examples.

.PARAMETER CI
    CI mode: always exit 0, generate report regardless of failures.

.PARAMETER VcpkgBin
    Optional path to vcpkg bin directory for additional DLLs.

.PARAMETER Timeout
    Maximum seconds per test (default: 60). Tests exceeding this are killed and marked failed.

.EXAMPLE
    .\test_all_examples.ps1 -SdkDir "C:\sdk" -JsonReport "C:\report.json" -Basic

.EXAMPLE
    .\test_all_examples.ps1 -SdkDir "C:\sdk" -JsonReport "C:\report.json" -Cuda -CI
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$SdkDir,

    [Parameter(Mandatory=$true)]
    [string]$JsonReport,

    [Parameter(Mandatory=$false)]
    [switch]$Basic,

    [Parameter(Mandatory=$false)]
    [switch]$Cuda,

    [Parameter(Mandatory=$false)]
    [switch]$CI,

    [Parameter(Mandatory=$false)]
    [string]$VcpkgBin = "",

    [Parameter(Mandatory=$false)]
    [int]$Timeout = 60
)

$ErrorActionPreference = "Stop"

# Determine test type
if (-not $Basic -and -not $Cuda) {
    # Default to basic if nothing specified
    $Basic = $true
}

$testType = if ($Cuda) { "cuda" } else { "basic" }

# Validate SDK directory
if (-not (Test-Path $SdkDir)) {
    Write-Error "SDK directory not found: $SdkDir"
    exit 1
}

$sdkBin = Join-Path $SdkDir "bin"
$cli = Join-Path $sdkBin "aprapipes_cli.exe"

if (-not (Test-Path $cli)) {
    Write-Error "CLI not found: $cli"
    exit 1
}

# Setup PATH for DLL loading
$env:PATH = "$sdkBin;$env:PATH"

if ($VcpkgBin -and (Test-Path $VcpkgBin)) {
    $env:PATH = "$VcpkgBin;$env:PATH"
    Write-Host "Added vcpkg bin to PATH: $VcpkgBin"
}

if ($env:CUDA_PATH) {
    $cudaBin = Join-Path $env:CUDA_PATH "bin"
    if (Test-Path $cudaBin) {
        $env:PATH = "$cudaBin;$env:PATH"
        Write-Host "Added CUDA bin to PATH: $cudaBin"
    }
}

# Test CLI launch
Write-Host "=== Testing CLI Launch ==="
Write-Host "CLI path: $cli"

try {
    $output = & $cli list-modules 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Error "CLI failed to launch with exit code $LASTEXITCODE"
        Write-Host "Output: $output"
        exit 1
    }
    Write-Host "CLI launched successfully"
} catch {
    Write-Error "CLI launch failed: $_"
    exit 1
}

# Define test examples based on test type
$examples = @()
$examplesDir = ""

if ($Basic) {
    $examplesDir = Join-Path $SdkDir "examples\basic"
    $examples = @(
        "simple_source_sink",
        "three_module_chain",
        "split_pipeline",
        "bmp_converter_pipeline",
        "affine_transform_demo",
        "affine_transform_chain",
        "ptz_with_conversion",
        "transform_ptz_with_conversion"
    )
}

if ($Cuda) {
    $examplesDir = Join-Path $SdkDir "examples\cuda"
    if (Test-Path $examplesDir) {
        $examples = Get-ChildItem "$examplesDir\*.json" -ErrorAction SilentlyContinue |
                    ForEach-Object { $_.BaseName }
    }

    if ($examples.Count -eq 0) {
        Write-Host "No CUDA examples found in: $examplesDir"
        # Create empty report and exit successfully
        $report = @{
            timestamp = (Get-Date -Format "o")
            script = "test_all_examples.ps1"
            test_type = $testType
            summary = @{ passed = 0; failed = 0; skipped = 1; total = 0 }
            note = "No CUDA examples found"
        }
        $report | ConvertTo-Json -Depth 4 | Set-Content $JsonReport -Encoding UTF8
        exit 0
    }
}

# Run tests
Write-Host ""
Write-Host "=== Running $testType Integration Tests ==="
Write-Host "Examples directory: $examplesDir"
Write-Host "Examples to test: $($examples.Count)"

$passed = 0
$failed = 0
$skipped = 0
$results = @()

foreach ($example in $examples) {
    $jsonPath = Join-Path $examplesDir "$example.json"

    if (-not (Test-Path $jsonPath)) {
        Write-Host "[SKIP] $example (file not found)"
        $skipped++
        $results += @{ name = $example; status = "skipped" }
        continue
    }

    Write-Host "[TEST] $example (timeout: ${Timeout}s)"

    try {
        Push-Location $SdkDir

        # Run CLI with timeout using Start-Process
        $tempOut = [System.IO.Path]::GetTempFileName()
        $tempErr = [System.IO.Path]::GetTempFileName()

        $proc = Start-Process -FilePath $cli -ArgumentList "validate", $jsonPath `
            -NoNewWindow -PassThru `
            -RedirectStandardOutput $tempOut `
            -RedirectStandardError $tempErr

        $completed = $proc.WaitForExit($Timeout * 1000)

        if (-not $completed) {
            # Timeout - kill the process
            $proc.Kill()
            $proc.WaitForExit(5000)
            Pop-Location
            Write-Host "[FAIL] $example (timeout after ${Timeout}s)"
            $failed++
            $results += @{ name = $example; status = "failed"; reason = "timeout" }
            Remove-Item $tempOut, $tempErr -ErrorAction SilentlyContinue
            continue
        }

        $exitCode = $proc.ExitCode
        $output = Get-Content $tempOut -Raw -ErrorAction SilentlyContinue
        $errOutput = Get-Content $tempErr -Raw -ErrorAction SilentlyContinue
        Remove-Item $tempOut, $tempErr -ErrorAction SilentlyContinue
        Pop-Location

        if ($exitCode -eq 0) {
            Write-Host "[PASS] $example"
            $passed++
            $results += @{ name = $example; status = "passed" }
        } else {
            Write-Host "[FAIL] $example (exit code: $exitCode)"
            if ($errOutput) { Write-Host "  Error: $errOutput" }
            $failed++
            $results += @{ name = $example; status = "failed" }
        }
    } catch {
        Write-Host "[FAIL] $example (exception: $_)"
        $failed++
        $results += @{ name = $example; status = "failed" }
        Pop-Location -ErrorAction SilentlyContinue
    }
}

# Generate report
$report = @{
    timestamp = (Get-Date -Format "o")
    script = "test_all_examples.ps1"
    test_type = $testType
    summary = @{
        passed = $passed
        failed = $failed
        skipped = $skipped
        total = $passed + $failed + $skipped
    }
    results = $results
}

$report | ConvertTo-Json -Depth 4 | Set-Content $JsonReport -Encoding UTF8

Write-Host ""
Write-Host "=== Test Summary ==="
Write-Host "Passed: $passed, Failed: $failed, Skipped: $skipped"
Write-Host "Report: $JsonReport"

if ($failed -gt 0 -and -not $CI) {
    exit 1
}

exit 0
