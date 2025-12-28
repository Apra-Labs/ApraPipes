# Docker Build Troubleshooting

Platform-specific troubleshooting for Docker containerized builds.

**Scope**: CI-Linux docker job running containerized builds.

---

## Docker-Specific Architecture

### Build Configuration
- **Workflow**: CI-Linux.yml
- **Job**: docker (nested within CI-Linux workflow)
- **Reusable**: build-test-lin-container.yml
- **Environment**: Docker container (Ubuntu 20.04+ base)
- **CUDA**: Optional (depends on container configuration)

### Workflow Files
- **Parent**: `.github/workflows/CI-Linux.yml`
- **Reusable**: `.github/workflows/build-test-lin-container.yml`

---

## Docker-Specific Architecture

### Build Configuration
- **Environment**: Docker container
- **Base Images**: Ubuntu, Alpine, or custom
- **Volumes**: Code mounted, caches mounted
- **Networking**: Container network access for vcpkg downloads

### Key Characteristics
- Isolated environment (clean state each run)
- Volume mounts for code and caches
- Network access required for vcpkg
- User permissions in containers

---

## Issue D1: Cache Mounting in Containers

**Symptom**:
```
vcpkg cache not persisting between builds
downloads re-fetched every time
```

**Root Cause**:
- vcpkg cache not mounted as volume
- Container destroyed after build, cache lost

**Fix**:

Mount vcpkg cache directory:
```yaml
# docker-compose.yml or docker run command
volumes:
  - ./vcpkg:/workspace/vcpkg
  - vcpkg-cache:/root/.cache/vcpkg  # Persist cache
```

**To Be Expanded**: Add specific Docker volume configuration examples.

---

## Issue D2: Container Cleanup Between Builds

**Symptom**:
- Old build artifacts remain
- Disk space issues in container

**Fix**:

Ensure containers are removed after each build:
```bash
docker-compose down -v  # Remove volumes
docker system prune -af  # Clean up
```

---

## Issue D3: Volume Permissions

**Symptom**:
```
Permission denied writing to /workspace
cannot create directory
```

**Root Cause**:
- User ID mismatch between host and container
- Volume mounted with wrong permissions

**Fix**:

Match container user to host user:
```dockerfile
RUN useradd -u 1000 -m builder
USER builder
```

Or run container with host user:
```bash
docker run --user $(id -u):$(id -g) ...
```

---

## Issue D4: GitHub Actions Node.js 20 GLIBC Compatibility (RESOLVED)

**Status**: As of November 2024, GitHub Actions requires GLIBC 2.28+. Ubuntu 18.04 containers are **no longer supported**.

**Symptom**:
```
/__e/node20/bin/node: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.28' not found
Error: The process '/__e/node20/bin/node' failed with exit code 1
```

**Root Cause**:
- GitHub Actions Node 16 reached EOL on November 12, 2024
- All actions now require Node.js 20, which needs GLIBC 2.28+
- Ubuntu 18.04 only has GLIBC 2.27 - **permanently incompatible**

**Required Solution**:
```dockerfile
# MUST use Ubuntu 20.04+ or Ubuntu 22.04
FROM ubuntu:22.04  # Has GLIBC 2.35 (recommended)
# OR
FROM ubuntu:20.04  # Has GLIBC 2.31 (minimum)
```

**Detection**:
```bash
# Check GLIBC version in container
ldd --version
# Output: ldd (Ubuntu GLIBC 2.31-0ubuntu9.12) 2.31  # OK for Node 20
```

**Best Practice**:
- Use Ubuntu 20.04+ for all GitHub Actions containers
- Ubuntu 18.04 workarounds (v3 actions) are deprecated and will stop working

**Invariant**: All GitHub Actions containers MUST use Ubuntu 20.04+ (GLIBC 2.28+)

---

## Docker Quick Fixes Checklist

- [ ] vcpkg cache mounted as volume
- [ ] Code directory mounted correctly
- [ ] Network access available
- [ ] User permissions correct
- [ ] Container cleanup after builds
- [ ] Ubuntu 20.04+ base image (GLIBC 2.28+)

---

## To Be Expanded

This guide will be expanded as Docker-specific issues are encountered:
- Dockerfile examples for ApraPipes builds
- docker-compose configuration for multi-stage builds
- Container registry integration
- Multi-platform Docker builds (x64 + ARM64)

**Cross-Platform Issues**: See other troubleshooting guides for platform-specific issues within containers.

---

**Applies to**: Docker containerized builds
**Related Guides**: reference.md, troubleshooting.linux.md
