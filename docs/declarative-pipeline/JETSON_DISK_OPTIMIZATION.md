# Jetson Disk Optimization Plan

> Created: 2026-01-17

## Problem

Jetson root partition (14GB eMMC) is at 67% capacity, leaving only ~4.3GB free. CI builds occasionally fail due to `/tmp` running out of space during compilation.

## Solution

Move CUDA toolkit from root partition to NVMe (`/data`) and create symlink.

## Current State

```
Filesystem      Size  Used Avail Use%
/dev/mmcblk0p1   14G  8.7G  4.3G  67%   (root - eMMC)
/dev/nvme0n1p1  117G   21G   90G  19%   (/data - NVMe)
```

**Large items on root:**
- `/usr/local/cuda-11.4`: 2.3GB (CUDA toolkit)
- `/usr/lib/aarch64-linux-gnu/libcudnn*`: ~2GB (keeping in place)

## Plan: Move CUDA Toolkit

**Prerequisites:**
- [ ] No CI build running on Jetson
- [ ] No active CUDA processes

**Steps:**

```bash
# 1. Verify no builds running
ps aux | grep -E 'cmake|ninja|gcc|nvcc' | grep -v grep

# 2. Create target directory on NVMe
sudo mkdir -p /data/usr/local

# 3. Move CUDA toolkit (mv preserves space, no double usage)
sudo mv /usr/local/cuda-11.4 /data/usr/local/cuda-11.4

# 4. Create symlink
sudo ln -s /data/usr/local/cuda-11.4 /usr/local/cuda-11.4

# 5. Verify symlinks (cuda and cuda-11 should still resolve)
ls -la /usr/local/cuda*

# 6. Verify CUDA works
nvcc --version
/usr/local/cuda/bin/nvcc --version

# 7. Check disk space
df -h /
```

**Expected Result:**
```
Filesystem      Size  Used Avail Use%
/dev/mmcblk0p1   14G  6.4G  6.6G  50%   (root - freed 2.3GB)
```

## Verification

After moving, trigger a test build:
```bash
gh workflow run CI-Linux-ARM64.yml --ref feat-declarative-pipeline-v2
```

## Rollback (if needed)

```bash
# Remove symlink
sudo rm /usr/local/cuda-11.4

# Move back
sudo mv /data/usr/local/cuda-11.4 /usr/local/cuda-11.4
```

## Notes

- `/data` is mounted via fstab, available at boot before any CUDA usage
- Docker is not installed on Jetson (placeholder dir only)
- cuDNN libraries left in place (more complex to move, many individual files)
