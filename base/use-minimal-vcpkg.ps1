# Script to swap to minimal vcpkg.json for fast CI testing
# Usage: .\use-minimal-vcpkg.ps1

if (-not (Test-Path "base/vcpkg.json.full-backup")) {
    Write-Host "Backing up full vcpkg.json..."
    Copy-Item "base/vcpkg.json" "base/vcpkg.json.full-backup"
}

Write-Host "Switching to minimal vcpkg.json (glib only)..."
Copy-Item "base/vcpkg.json.minimal-test" "base/vcpkg.json" -Force

Write-Host "`nMinimal vcpkg.json activated. This will test:" -ForegroundColor Green
Write-Host "  - libxml2 hash (dependency of glib)" -ForegroundColor Cyan
Write-Host "  - Python distutils (required by glib build)" -ForegroundColor Cyan
Write-Host "`nTo restore: Copy-Item base/vcpkg.json.full-backup base/vcpkg.json -Force" -ForegroundColor Yellow
